"""Keep the solver on CPU; delegate each local denoiser to ComfyUI."""

import math
from numbers import Integral, Real
import time

import torch
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
from comfy.nested_tensor import NestedTensor

from .common import validate_sigmas
from .rolling import CPURollingFusion, RollingConfig
from .windows import CPUFusion, crop, pack, unpack


def _native_model_info(model):
    if not isinstance(getattr(model, "model", None), comfy.model_base.MiniMaxH3):
        raise ValueError("H3 window sampling requires a native MiniMax H3 model (official REF2VA or FastH3).")
    if model.get_additional_models_with_key("multigpu"):
        raise ValueError("H3 window sampling currently supports a single GPU model.")
    final_layer = getattr(model.model.diffusion_model, "final_layer", None)
    pdd_steps = getattr(final_layer, "pdd_steps", None)
    if isinstance(pdd_steps, bool) or not isinstance(pdd_steps, Integral) or pdd_steps < 1:
        raise ValueError("Cannot identify the native H3 output head's pdd_steps; check the model loader.")
    return int(pdd_steps)


def _validate_global_noise(noise, shapes):
    if not isinstance(noise, NestedTensor):
        raise ValueError("NOISE.generate_noise must return the native NestedTensor video/audio pair for H3.")
    streams = noise.unbind()
    if len(streams) != len(shapes):
        raise ValueError("NOISE returned the wrong number of H3 AV streams.")
    for index, (stream, shape) in enumerate(zip(streams, shapes)):
        if not isinstance(stream, torch.Tensor) or tuple(stream.shape) != tuple(shape):
            raise ValueError(f"NOISE stream {index} does not match the full global H3 latent shape {shape}.")
        if stream.device.type != "cpu" or stream.dtype != torch.float32:
            raise ValueError("NOISE must generate the full H3 timeline as CPU FP32 tensors.")
        if not bool(torch.isfinite(stream).all()):
            raise ValueError("NOISE generated non-finite values.")
    return noise


def _prepare_initial_latent(entries, fusion):
    """Fuse raw source latents on the same grid as denoised windows.

    Two branches are supported independently:

    * ``initial_video_latent`` — a raw video VAE latent per window. Either the
      VAE-encoded source video (first pass, ``video_encode``) or a slice of an
      already-generated joint latent (second pass).
    * ``initial_audio_latent`` — a slice of an existing audio latent. When
      present, the audio target starts from that content instead of silence,
      which is what a second pass needs so the soundtrack is not re-invented.

    Normalization into model/solver space belongs to process_latent_in in
    outer_sample, after model patches have been applied; noise mixing belongs
    exclusively to the native KSAMPLER.
    """
    source_flags = ["initial_video_latent" in entry for entry in entries]
    if any(source_flags) and not all(source_flags):
        raise ValueError("All H3 windows must use the same source-video initial latent mode.")
    if all(source_flags):
        for index, entry in enumerate(entries, 1):
            video = entry["initial_video_latent"]
            if not isinstance(video, torch.Tensor) or tuple(video.shape) != tuple(entry["window"].shapes[0]):
                raise ValueError(f"Window {index} initial_video_latent must match its native H3 video latent shape.")
            if video.device.type != "cpu" or video.dtype != torch.float32:
                raise ValueError("initial_video_latent must contain raw CPU FP32 video VAE latents.")
            if not bool(torch.isfinite(video).all()):
                raise ValueError(f"Window {index} initial_video_latent contains non-finite values.")

    audio_flags = ["initial_audio_latent" in entry for entry in entries]
    if any(audio_flags) and not all(audio_flags):
        raise ValueError("All H3 windows must use the same initial audio latent mode.")
    if all(audio_flags):
        for index, entry in enumerate(entries, 1):
            audio = entry["initial_audio_latent"]
            if not isinstance(audio, torch.Tensor) or tuple(audio.shape) != tuple(entry["window"].shapes[1]):
                raise ValueError(f"Window {index} initial_audio_latent must match its native H3 audio latent shape.")
            if audio.device.type != "cpu" or audio.dtype != torch.float32:
                raise ValueError("initial_audio_latent must contain raw CPU FP32 audio latents.")
            if not bool(torch.isfinite(audio).all()):
                raise ValueError(f"Window {index} initial_audio_latent contains non-finite values.")

    streams = [torch.zeros(shape, device="cpu", dtype=torch.float32) for shape in fusion.shapes]
    if all(source_flags):
        for entry, weights in zip(entries, fusion.weights):
            window = entry["window"]
            destination = streams[0][:, :, window.video_start:window.video_start + window.video_length]
            destination.addcmul_(entry["initial_video_latent"].detach(), weights[0])
        streams[0].div_(fusion.counts[0])
    if all(audio_flags):
        for entry, weights in zip(entries, fusion.weights):
            window = entry["window"]
            destination = streams[1][..., window.audio_start:window.audio_start + window.audio_length]
            destination.addcmul_(entry["initial_audio_latent"].detach(), weights[1])
        streams[1].div_(fusion.counts[1])

    if all(source_flags) and all(audio_flags):
        mode = "source_av"
    elif all(source_flags):
        mode = "source_video"
    else:
        mode = "empty_av"
    return NestedTensor(streams), mode


def _validate_sampler_tensor(value, expected_shape, label):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(expected_shape):
        raise ValueError(f"{label} must match the global packed AV shape {tuple(expected_shape)}.")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{label} must remain a CPU FP32 tensor.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values.")
    return value


class CPUWindowGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, entries, cfg, *, cfg_schedule=None, blend=0.0):
        super().__init__(model)
        self.entries = entries
        self.blend = float(blend)
        self.fusion = CPUFusion([entry["window"] for entry in entries], blend=self.blend)
        self.set_cfg(cfg)
        self.cfg_schedule = cfg_schedule
        # Always keep both groups. An early CFG=1 phase may skip its negative
        # forward pass, but later sigma values can still require that condition.
        self.inner_set_conds({f"{kind}_{i}": entry[kind]
                              for i, entry in enumerate(entries) for kind in ("positive", "negative")})
        self.window_calls = 0
        self.model_evaluations = 0
        self.sampler_callbacks = 0
        self.sampler_sigmas = None
        self.model_post_cfg_hooks = ()
        self.progress_value = 0
        self.window_progress = None
        # Progress is counted across the whole run: total = windows * steps, and
        # the bar advances once per finished window. Counting only the step and
        # taking max() left it stuck at full after window 1, which is how the
        # per-window progress disappeared.
        self.progress_total = 0
        self.window_conds = []

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
                     callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        if noise.device.type != "cpu" or latent_image.device.type != "cpu":
            raise ValueError("Global AV input unexpectedly moved off CPU.")
        if denoise_mask is not None:
            raise ValueError("This joint sampler supports unmasked REF2VA targets only.")
        device = self.model_patcher.load_device
        base = self.model_patcher.model
        old_shapes = base.latent_shapes
        self.inner_model = base
        self.window_calls = 0
        self.model_evaluations = 0
        self.sampler_callbacks = 0
        self.progress_value = 0
        self.sampler_sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32).clone()
        loaded = []
        try:
            largest_shape = max((entry["window"].shapes for entry in self.entries),
                                key=lambda shapes: sum(math.prod(shape) for shape in shapes))
            local_shape = (1, 1, sum(math.prod(shape) for shape in largest_shape))
            base.latent_shapes = largest_shape
            self.inner_model, self.conds, loaded = comfy.sampler_helpers.prepare_sampling(
                self.model_patcher, local_shape, self.conds, self.model_options)
            base = self.inner_model
            with comfy.model_management.cuda_device_context(device):
                comfy.samplers.cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())
                self.model_patcher.pre_run()
                # Match native CFGGuider.inner_sample: convert nonempty raw VAE
                # latents exactly once, with the full AV layout. MiniMaxH3 also
                # applies its audio solver scale here; do not apply it manually.
                base.latent_shapes = latent_shapes
                initial_streams = None
                if bool(torch.count_nonzero(latent_image)):
                    latent_image = base.process_latent_in(latent_image)
                    if latent_image.device.type != "cpu" or latent_image.dtype != torch.float32:
                        raise ValueError("Processed global initial latent must remain CPU FP32.")
                    initial_streams = unpack(latent_image, latent_shapes)
                for i, entry in enumerate(self.entries):
                    window = entry["window"]
                    base.latent_shapes = window.shapes
                    conds = {kind: [cond.copy() for cond in self.conds[f"{kind}_{i}"]]
                             for kind in ("positive", "negative")}
                    local_shape = (1, 1, sum(math.prod(shape) for shape in window.shapes))
                    local_zero = torch.zeros(local_shape, dtype=torch.float32, device="cpu")
                    local_initial = pack(crop(initial_streams, window)) if initial_streams is not None else local_zero
                    prepared = comfy.samplers.process_conds(
                        base, local_zero, conds, device, latent_image=local_initial,
                        seed=seed, latent_shapes=window.shapes)
                    # The native forward transfers this small refined text and only
                    # this window's visual references when evaluating the window.
                    for group in prepared.values():
                        for cond in group:
                            text = cond["model_conds"].get("c_crossattn")
                            if text is not None:
                                cond["model_conds"]["c_crossattn"] = text._copy_with(text.cond.cpu())
                            del text
                    self.window_conds.append(prepared)
                    del local_zero, local_initial

                base.latent_shapes = latent_shapes
                options = comfy.model_patcher.create_model_options_clone(self.model_options)
                # H3's PDD heads compare this vector with a GPU sigma. Only the
                # tiny model-side copy lives on GPU; the solver schedule stays CPU.
                options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas.to(device).clone()
                # Hooks installed on the model retain window-local behavior.
                # Hooks appended by the sampler (e.g. CFG++ history capture)
                # must observe one fused global prediction per model call.
                self.model_post_cfg_hooks = tuple(options.get("sampler_post_cfg_function", ()))
                args = {"model_options": options, "seed": seed}
                executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                    sampler.sample, sampler,
                    comfy.patcher_extension.get_all_wrappers(
                        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, options, is_model_options=True))
                def sampler_callback(*values):
                    self.sampler_callbacks += 1
                    if callback is not None:
                        callback(*values)

                # Some native algorithms edit their private schedule in place
                # (UniPC substitutes its own terminal epsilon). Keep the caller
                # grid and PDD's reference grid independent of that workspace.
                result = executor.execute(self, sigmas.clone(), args, sampler_callback, noise, latent_image, None, disable_pbar)
                _validate_sampler_tensor(result, noise.shape, "SAMPLER result")
                base.latent_shapes = latent_shapes
                output = base.process_latent_out(result)
                _validate_sampler_tensor(output, noise.shape, "Decoded-space sampler latent")
                if self.window_progress is not None:
                    self.window_progress.update_absolute(
                        self.progress_total or (len(sigmas) - 1))
                return output
        finally:
            base.latent_shapes = old_shapes
            self.window_conds.clear()
            self.model_patcher.cleanup()
            comfy.sampler_helpers.cleanup_models(self.conds, loaded)
            del self.inner_model
            self.sampler_sigmas = None
            self.model_post_cfg_hooks = ()

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        expected_shape = (1, 1, sum(math.prod(shape) for shape in self.fusion.shapes))
        _validate_sampler_tensor(x, expected_shape, "SAMPLER model input")
        if not isinstance(timestep, torch.Tensor) or timestep.numel() == 0:
            raise ValueError("SAMPLER must evaluate a finite H3 sigma in [0, 1].")
        if not bool(torch.isfinite(timestep).all()) or not bool(((timestep >= 0) & (timestep <= 1)).all()):
            raise ValueError("SAMPLER evaluated H3 outside its finite sigma domain [0, 1].")
        if not bool((timestep == timestep.reshape(-1)[0]).all()):
            raise ValueError("H3 joint AV sampling requires one shared sigma per model evaluation.")
        device = self.model_patcher.load_device
        base = self.inner_model
        old_shapes = base.latent_shapes
        if self.cfg_schedule is not None:
            high_cfg, cfg_split, low_cfg = self.cfg_schedule
            # Compare on the sigma tensor's dtype so an exact float32 boundary
            # is not displaced by converting the sigma to a Python float.
            active_cfg = high_cfg if bool(timestep.reshape(-1)[0] >= cfg_split) else low_cfg
            self.set_cfg(active_cfg)
        else:
            active_cfg = self.cfg
        sigma = timestep.to(device)
        supplied_options = model_options if model_options is not None else self.model_options
        options = dict(supplied_options)
        baseline_ids = {id(fn) for fn in self.model_post_cfg_hooks}
        hooks = list(options.get("sampler_post_cfg_function", ()))
        local_hooks = [fn for fn in hooks if id(fn) in baseline_ids]
        global_hooks = [fn for fn in hooks if id(fn) not in baseline_ids]
        options["sampler_post_cfg_function"] = local_hooks
        pdd_steps = getattr(getattr(getattr(base, "diffusion_model", None), "final_layer", None), "pdd_steps", 1)
        if pdd_steps > 1 and self.sampler_sigmas is not None:
            current = timestep.reshape(-1)[0].detach().to(device="cpu", dtype=torch.float64)
            comparison_grid = self.sampler_sigmas.to(dtype=torch.float64)
            # At original schedule points preserve the native head interval.
            # At intermediate evaluations, nearest-neighbor on the full grid
            # can select a future point and skip an interval. Give PDD the
            # exact current point and the next strictly lower schedule point.
            if not bool((comparison_grid == current).any()):
                lower = comparison_grid[comparison_grid < current]
                next_sigma = lower[0] if lower.numel() else current.new_zeros(())
                transformer_options = dict(options.get("transformer_options", {}))
                transformer_options["sample_sigmas"] = torch.stack((current, next_sigma)).to(device)
                options["transformer_options"] = transformer_options

        fused_branches = {}
        if global_hooks:
            fused_branches = {name: torch.zeros_like(x) for name in ("cond_denoised", "uncond_denoised")}
            branch_streams = {name: unpack(value, self.fusion.shapes) for name, value in fused_branches.items()}
            branch_conditions = {name: [] for name in ("cond", "uncond")}
        self.model_evaluations += 1

        def predict(index, local):
            comfy.model_management.throw_exception_if_processing_interrupted()
            window = self.entries[index]["window"]
            base.latent_shapes = window.shapes
            conds = self.window_conds[index]
            local_gpu = local.to(device)
            # Expose the window's timeline offset so model patches that read a
            # full-length control video (ControlNet and friends) can slice their
            # own frame range instead of always taking frames from index 0.
            local_options = dict(options)
            window_transformer_options = dict(options.get("transformer_options", {}))
            window_transformer_options["proyaya_window_start"] = int(window.start)
            window_transformer_options["proyaya_window_frames"] = int(window.frames)
            local_options["transformer_options"] = window_transformer_options
            if global_hooks:
                def capture_branches(args):
                    # Preserve the actual native CFG branch metadata. At CFG=1
                    # the optimized unconditional prediction may be a zero
                    # tensor, with uncond=None; history samplers use that None
                    # to fall back to the conditional prediction.
                    for name in branch_conditions:
                        branch_conditions[name].append(args[name])
                    for name, streams in branch_streams.items():
                        branch = args[name]
                        if branch is None:
                            # Native CFG=1 optimizations can omit unconditional
                            # evaluation unless the sampler requests it.
                            fused_branches[name] = None
                            continue
                        values = unpack(branch.to(device="cpu", dtype=torch.float32), self.entries[index]["window"].shapes)
                        for destination, value, weight in zip(crop(streams, self.entries[index]["window"]), values, self.fusion.weights[index]):
                            destination.addcmul_(value, weight)
                    return args["denoised"]

                local_options["sampler_post_cfg_function"] = local_hooks + [capture_branches]
            predicted = comfy.samplers.sampling_function(
                base, local_gpu, sigma, conds["negative"], conds["positive"], active_cfg,
                model_options=local_options, seed=seed)
            output = predicted.to(device="cpu", dtype=torch.float32)
            self.window_calls += 1
            del predicted, local_gpu
            if self.window_progress is not None and self.sampler_sigmas is not None:
                # True per-window progress: cell (step, window) = (step-1)*N + window + 1,
                # so the bar advances once per finished window instead of once per
                # schedule step (which made it hit full on the first window).
                reached = int((self.sampler_sigmas[:-1] >= float(sigma.reshape(-1)[0])).sum())
                windows = max(len(self.entries), 1)
                target = (max(reached, 1) - 1) * windows + index + 1
                if target > self.progress_value:
                    self.progress_value = target
                self.window_progress.update_absolute(self.progress_value)
            return output

        try:
            result = self.fusion.predict(x, predict)
            base.latent_shapes = self.fusion.shapes
            if global_hooks:
                for name, value in fused_branches.items():
                    if value is not None:
                        for total, count in zip(branch_streams[name], self.fusion.counts):
                            total.div_(count)
                        _validate_sampler_tensor(value, x.shape, f"Fused {name}")
                global_options = dict(supplied_options)
                global_options["transformer_options"] = options.get("transformer_options", {})
                global_conditions = {
                    name: ([condition for group in groups if group is not None for condition in group]
                           if any(group is not None for group in groups) else None)
                    for name, groups in branch_conditions.items()
                }
                for fn in global_hooks:
                    result = fn({"denoised": result, "cond_denoised": fused_branches["cond_denoised"],
                                 "uncond_denoised": fused_branches["uncond_denoised"],
                                 "cond": global_conditions["cond"],
                                 "uncond": global_conditions["uncond"],
                                 "cond_scale": active_cfg, "model": base, "sigma": timestep,
                                 "model_options": global_options, "input": x})
                    _validate_sampler_tensor(result, x.shape, "Global post-CFG prediction")
            return result
        finally:
            base.latent_shapes = old_shapes


class RollingWindowGuider(CPUWindowGuider):
    """CPUWindowGuider plus same-sigma previous-window prediction transfer."""

    def __init__(self, model, entries, cfg, *, cfg_schedule=None, rolling=None, blend=0.0):
        super().__init__(model, entries, cfg, cfg_schedule=cfg_schedule, blend=blend)
        self.fusion = CPURollingFusion([entry["window"] for entry in entries],
                                       rolling or RollingConfig(), blend=self.blend)

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
                     callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        sampling_settings = self.model_patcher.get_model_object("model_sampling")
        scale = getattr(sampling_settings, "noise_scale", 1.0)
        self.fusion.start_run(noise, noise_scale=scale)
        self.window_calls = 0
        try:
            return super().outer_sample(noise, latent_image, sampler, sigmas, denoise_mask,
                                        callback, disable_pbar, seed, latent_shapes)
        finally:
            self.fusion.finish_run()

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        self.fusion.sigma = float(timestep.reshape(-1)[0])
        return super().predict_noise(x, timestep, model_options=model_options, seed=seed)

    def continuity_report(self):
        return self.fusion.report()


def _checked_number(name, value, upper=None):
    ok = (not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
          and value >= 0 and (upper is None or value <= upper))
    if not ok:
        raise ValueError(f"{name} must be a finite number in [0, {upper}]." if upper is not None
                         else f"{name} must be a finite nonnegative number.")
    return float(value)


def sample_windows_native(noise, model, entries, sampler, sigmas, high_cfg, cfg_split, low_cfg,
                          rolling=None):
    """The only sampling entry point.

    Accepts the native NOISE / SAMPLER / SIGMAS interfaces and switches CFG at
    the current sigma. As in SamplerCustomAdvanced the NOISE object is called
    once for the full nested AV timeline; windows never generate their own
    noise. ``rolling`` (a RollingConfig) enables same-sigma window transfer;
    omitted, plain V1 fusion is used.
    """
    pdd_steps = _native_model_info(model)
    if not callable(getattr(noise, "generate_noise", None)):
        raise ValueError("Connect a native NOISE output, such as RandomNoise, to the noise input.")
    seed = getattr(noise, "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError("NOISE.seed must be an integer in [0, 18446744073709551615].")
    high_cfg = _checked_number("high_cfg", high_cfg)
    low_cfg = _checked_number("low_cfg", low_cfg)
    cfg_split = _checked_number("cfg_split", cfg_split, upper=1)
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("Connect at least one encoded H3 window before sampling.")
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not {"window", "positive", "negative"}.issubset(entry):
            raise ValueError(f"H3 window {index} is missing its window or conditioning data.")
    # SAMPLER owns its algorithm and options; device, sigma and packed AV
    # contracts are checked on the actual calls/results instead of by name.
    if not callable(getattr(sampler, "sample", None)):
        raise ValueError("SAMPLER must provide a callable sample method.")
    function = getattr(sampler, "sampler_function", None)
    sampler_name = getattr(function, "__name__", type(sampler).__name__).removeprefix("sample_")
    schedule = validate_sigmas(sigmas)
    cfg_schedule = (high_cfg, cfg_split, low_cfg)

    cloned = model.clone()
    if rolling is None:
        guider = CPUWindowGuider(cloned, entries, high_cfg, cfg_schedule=cfg_schedule)
    else:
        guider = RollingWindowGuider(cloned, entries, high_cfg, cfg_schedule=cfg_schedule, rolling=rolling)
    latent, initial_mode = _prepare_initial_latent(entries, guider.fusion)
    # Native RandomNoise prepares both streams with one global generator.
    global_noise = _validate_global_noise(noise.generate_noise({"samples": latent}), guider.fusion.shapes)
    steps = schedule.numel() - 1
    # Higher-order/adaptive solvers may evaluate more than once per interval.
    # Progress follows reached schedule boundaries, while actual calls are
    # counted separately; rejected trial steps never regress the UI.
    # One cell per (window, step) so the bar reports the real amount of work.
    guider.progress_total = len(entries) * max(steps, 1)
    guider.window_progress = comfy.utils.ProgressBar(guider.progress_total)

    device = model.load_device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    # No per-step preview callback: keep the full timeline and its previews off GPU.
    output = guider.sample(global_noise, latent, sampler, schedule, callback=None, seed=seed)
    report = {
        "windows": [{"start": e["window"].start, "frames": e["window"].frames} for e in entries],
        "frames": entries[-1]["window"].start + entries[-1]["window"].frames,
        "fps": 24, "global_state_device": "cpu", "window_calls": guider.window_calls,
        "model_evaluations": guider.model_evaluations,
        "sampler_callbacks": guider.sampler_callbacks,
        "progress_basis": "sigma_schedule_boundaries",
        "endpoint_policy": "native_sampler; schedule_zero_does_not_force_an_extra_terminal_step",
        "steps": steps, "sampler": sampler_name, "seed": int(seed),
        "initial_latent_mode": initial_mode, "initial_audio_mode": "empty",
        "start_sigma": float(schedule[0]),
        "sigmas": "external:SIGMAS",
        "schedule": [round(float(s), 6) for s in schedule],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "one_global_latent_MiB": round(sum(t.numel() * t.element_size() for t in output.unbind()) / 2**20, 2),
        "pdd_steps": pdd_steps,
        "model_variant": "pdd" if pdd_steps > 1 else "native_ref2va",
        "external_sampler_object": True,
        "noise_source": type(noise).__name__, "noise_generate_calls": 1,
        "noise_generated_for": "global_av_latent",
        "noise_generation_scope": "initial_global_noise_only; sampler_may_add_noise",
        "high_cfg": high_cfg, "cfg_split": cfg_split, "low_cfg": low_cfg,
        "cfg_rule": "sigma >= cfg_split: high_cfg; sigma < cfg_split: low_cfg",
    }
    if device.type == "cuda":
        report["peak_cuda_allocated_MiB"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        report["peak_cuda_reserved_MiB"] = round(torch.cuda.max_memory_reserved(device) / 2**20, 2)
    if hasattr(guider, "continuity_report"):
        report["continuity"] = guider.continuity_report()
    return {"samples": output.cpu()}, report