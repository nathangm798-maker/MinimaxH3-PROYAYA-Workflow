"""Masked sampling for PROYAYA H3 joint windows.

This pack adds a sampler that accepts a real denoise MASK on top of PROYAYA's
CPU window guider. PROYAYA itself is not modified: the masked path lives in a
subclass that overrides ``outer_sample`` (the only place PROYAYA refuses a
mask), and the rolling transfer is deliberately absent.

Why a subclass instead of an edit
---------------------------------
``comfy.samplers.Guider.sample`` already prepares a denoise mask for the
global packed AV latent and hands it to ``outer_sample``. PROYAYA's
``CPUWindowGuider.outer_sample`` raises on that argument. Overriding the single
method keeps the upstream pack byte-identical while giving the sampler both
mask views it needs:

* the packed global mask, handed to the sampler so preserved rows fall back to
  the source latent every step;
* a per-window crop of that mask, handed to ``process_conds`` so H3's
  ``_denoise_mask_conds`` can pin preserved rows to the conditioning timestep.

The ``outer_sample`` body below is adapted from PROYAYA ``runtime.py``
(``CPUWindowGuider.outer_sample``); see the backup folder
``custom_nodes/_backup_20261028_2240/PROYAYA`` for the reference version.

Mask presence decides the path
------------------------------
The node has no on/off switch. A connected ``mask`` means the masked path: the
mask is read, preserved rows fall back to the source latent, and every window
must carry its own target latent (``video_encode`` on the upstream encoder).
An unconnected ``mask`` means PROYAYA's stock window sampler: no mask cond, no
latent blending, and the source video only reaches the model through the
``<Video n>`` reference channel.

The mask polarity is fixed at white = regenerate. There is no inverted mode: a
caller that needs white = preserve must invert the mask upstream.
"""

from __future__ import annotations

import json
import math
import time

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
from comfy.nested_tensor import NestedTensor

# 本文件已并入 PROYAYA 包，直接走包内相对导入。
# （历史：曾作为独立的 ComfyUI-PROYAYA-MaskedSampler 包存在，需要 import PROYAYA
# 顶层模块并给 sys.path 打补丁；合并后这层兜底已无必要。）
from . import runtime as _RUNTIME
from .rolling import CPURollingFusion, ROLLING_MODES, RollingConfig

MaskedGuiderBase = _RUNTIME.CPUWindowGuider
_prepare_initial_latent = _RUNTIME._prepare_initial_latent
_validate_global_noise = _RUNTIME._validate_global_noise
_checked_number = _RUNTIME._checked_number
validate_sigmas = _RUNTIME.validate_sigmas
unpack = _RUNTIME.unpack
pack = _RUNTIME.pack
crop = _RUNTIME.crop


def _normalize_mask(mask):
    """Reduce a ComfyUI MASK to a float [frames, H, W] tensor."""
    if mask is None:
        raise ValueError("需要连接遮罩：白色表示要重绘的区域。")
    m = mask.float()
    if m.ndim == 4:
        m = m[..., 0] if m.shape[-1] == 1 else m[:, :, :, 0]
    if m.ndim == 3:
        pass
    elif m.ndim == 2:
        m = m[None]
    else:
        raise ValueError(f"无法识别的遮罩维度 {m.ndim}；期望 [H,W]、[1,H,W] 或 [N,H,W]。")
    if m.shape[0] == 0 or m.shape[1] == 0 or m.shape[2] == 0:
        raise ValueError("遮罩为空。")
    if not bool(torch.isfinite(m).all()):
        raise ValueError("遮罩包含非有限值。")
    return m.clamp(0.0, 1.0)


def _video_mask_to_latent(mask, latent_t, latent_h, latent_w):
    """Map a pixel-space mask onto H3's video latent grid.

    The temporal axis is always a proportional stretch: the mask timeline is
    resampled onto the latent timeline with nearest-exact. That is exact for a
    static mask (one frame replicated), and for a per-frame mask it keeps every
    latent frame tied to the same proportional position in the mask timeline.

    Spatial reduction uses adaptive max pooling so a hard mask stays hard and
    covers the same region after the VAE's 16x downscale.
    """
    # [F,H,W] -> [1,1,F,H,W] so interpolate works on the (D,H,W) axes, then drop
    # both batch axes back to a plain [latent_t,H,W] frame stack.
    stretched = F.interpolate(mask[None, None], size=(latent_t, mask.shape[1], mask.shape[2]),
                              mode="nearest-exact")[0, 0]
    pooled = F.adaptive_max_pool2d(stretched.unsqueeze(1), (latent_h, latent_w))
    return pooled.permute(1, 0, 2, 3).contiguous()


def _audio_mask_to_latent(audio_length, lock=False):
    """Audio denoise mask.

    ``lock=False`` (first pass): H3's audio target starts empty, so the only
    useful policy is "generate all of it".
    ``lock=True`` (second pass): the audio already exists and must survive the
    refine pass untouched, so every audio row is preserved. A preserved row is
    blended back to the initial latent at every step, which is bit-exact.
    """
    if lock:
        return torch.zeros((1, audio_length), dtype=torch.float32)
    return torch.ones((1, audio_length), dtype=torch.float32)


def build_denoise_mask(mask, shapes, lock_audio=False):
    """Build the nested [video, audio] denoise mask for one sampling run."""
    normalized = _normalize_mask(mask)

    video_shape = shapes[0]
    video_mask = _video_mask_to_latent(normalized, video_shape[2], video_shape[3],
                                       video_shape[4]).clamp(0.0, 1.0)
    if float(video_mask.max()) <= 0.0:
        raise ValueError("遮罩全黑，没有可重绘的区域。白色 = 要重绘，检查遮罩图是否全黑或上下级取反了。")
    audio_mask = _audio_mask_to_latent(shapes[1][-1], lock=lock_audio)
    nested = NestedTensor([video_mask, audio_mask])
    return nested, {
        "enabled": True,
        "mask_frames": int(normalized.shape[0]),
        "mask_pixels": [int(normalized.shape[1]), int(normalized.shape[2])],
        "latent_video": [int(video_shape[2]), int(video_shape[3]), int(video_shape[4])],
        "video_regenerate_ratio": round(float(video_mask.mean()), 6),
        "audio_regenerate_ratio": round(float(audio_mask.mean()), 6),
        "audio_lock": bool(lock_audio),
        "temporal_map": "proportional stretch",
    }


def _lock_audio_mask(shapes):
    """Second pass without an external mask: redraw all video, keep audio as-is.

    The video side is a plain ones tensor, so the sampler behaves exactly like
    the unmasked path for the picture while the audio rows are pinned to the
    initial audio latent on every step.
    """
    video_shape = shapes[0]
    return NestedTensor([
        torch.ones((video_shape[2], video_shape[3], video_shape[4]), dtype=torch.float32),
        torch.zeros((1, shapes[1][-1]), dtype=torch.float32),
    ])


class MaskedCPUWindowGuider(MaskedGuiderBase):
    """PROYAYA's plain CPU window guider, plus denoise-mask support.

    Rolling transfer stays off while a denoise mask is active: the same-sigma
    prediction handoff has no meaningful behaviour inside a masked region and
    would bleed preserved pixels across the mask border. Without a mask it is
    available as an opt-in, because the masked path otherwise falls through to
    the stock implementation where rolling is well defined.
    """

    def __init__(self, model, entries, cfg, *, cfg_schedule=None, blend=0.0, rolling=None):
        super().__init__(model, entries, cfg, cfg_schedule=cfg_schedule, blend=blend)
        self.rolling = rolling if (rolling is not None and rolling.enabled) else None
        if self.rolling is not None:
            # Same-sigma handoff of the previous window's post-CFG x0 into the
            # next window's overlap slice, before that window is evaluated.
            self.fusion = CPURollingFusion([entry["window"] for entry in entries],
                                           self.rolling, blend=self.blend)

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if self.rolling is not None and isinstance(timestep, torch.Tensor) and timestep.numel():
            self.fusion.sigma = float(timestep.reshape(-1)[0])
        return super().predict_noise(x, timestep, model_options=model_options, seed=seed)

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
                     callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        if self.rolling is None:
            return self._outer_sample_body(noise, latent_image, sampler, sigmas, denoise_mask,
                                           callback, disable_pbar, seed, latent_shapes)
        scale = getattr(self.model_patcher.get_model_object("model_sampling"), "noise_scale", 1.0)
        self.fusion.start_run(noise, noise_scale=scale)
        try:
            return self._outer_sample_body(noise, latent_image, sampler, sigmas, denoise_mask,
                                           callback, disable_pbar, seed, latent_shapes)
        finally:
            self.fusion.finish_run()

    def _outer_sample_body(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
                           callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        if denoise_mask is None:
            return super().outer_sample(noise, latent_image, sampler, sigmas, denoise_mask,
                                        callback, disable_pbar, seed, latent_shapes)
        if noise.device.type != "cpu" or latent_image.device.type != "cpu":
            raise ValueError("Global AV input unexpectedly moved off CPU.")
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
                comfy.samplers.cast_to_load_options(self.model_options, device=device,
                                                    dtype=self.model_patcher.model_dtype())
                self.model_patcher.pre_run()
                base.latent_shapes = latent_shapes
                initial_streams = None
                if bool(torch.count_nonzero(latent_image)):
                    latent_image = base.process_latent_in(latent_image)
                    if latent_image.device.type != "cpu" or latent_image.dtype != torch.float32:
                        raise ValueError("Processed global initial latent must remain CPU FP32.")
                    initial_streams = unpack(latent_image, latent_shapes)
                # ComfyUI already packed the mask onto the global AV latent.
                mask_streams = unpack(denoise_mask, latent_shapes)
                for i, entry in enumerate(self.entries):
                    window = entry["window"]
                    base.latent_shapes = window.shapes
                    conds = {kind: [cond.copy() for cond in self.conds[f"{kind}_{i}"]]
                             for kind in ("positive", "negative")}
                    local_shape = (1, 1, sum(math.prod(shape) for shape in window.shapes))
                    local_zero = torch.zeros(local_shape, dtype=torch.float32, device="cpu")
                    local_initial = (pack(crop(initial_streams, window))
                                     if initial_streams is not None else local_zero)
                    local_mask = pack(crop(mask_streams, window))
                    prepared = comfy.samplers.process_conds(
                        base, local_zero, conds, device, latent_image=local_initial,
                        denoise_mask=local_mask, seed=seed, latent_shapes=window.shapes)
                    for group in prepared.values():
                        for cond in group:
                            text = cond["model_conds"].get("c_crossattn")
                            if text is not None:
                                cond["model_conds"]["c_crossattn"] = text._copy_with(text.cond.cpu())
                            del text
                    self.window_conds.append(prepared)
                    del local_zero, local_initial, local_mask
                del mask_streams

                base.latent_shapes = latent_shapes
                options = comfy.model_patcher.create_model_options_clone(self.model_options)
                options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas.to(device).clone()
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

                # ComfyUI hands us the mask on load_device, which is right for the
                # model-side conditioning below. The sampler itself runs on the
                # CPU global state that PROYAYA keeps (see _prepare_initial_latent
                # and CPUWindowGuider.predict_noise): KSamplerX0Inpaint blends
                # ``x * denoise_mask + latent_image * (1 - mask)`` on that state,
                # so this copy must stay CPU FP32 or the blend mixes devices.
                sampler_mask = denoise_mask.detach().to(device="cpu", dtype=torch.float32)
                result = executor.execute(self, sigmas.clone(), args, sampler_callback,
                                          noise, latent_image, sampler_mask, disable_pbar)
                _RUNTIME._validate_sampler_tensor(result, noise.shape, "SAMPLER result")
                base.latent_shapes = latent_shapes
                output = base.process_latent_out(result)
                _RUNTIME._validate_sampler_tensor(output, noise.shape, "Decoded-space sampler latent")
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


def sample_masked_windows(noise, model, entries, sampler, sigmas, mask,
                          high_cfg, cfg_split, low_cfg, blend=0.0, rolling=None):
    """Window sampling with an optional denoise mask.

    ``mask is None`` turns this into PROYAYA's unmasked window sampler: the mask
    is never read, ``outer_sample`` falls through to the stock implementation,
    and the source video only reaches the model through the ``<Video n>``
    reference channel.
    """
    from .runtime import _native_model_info

    mask_enabled = mask is not None
    # 二采：windows 由二采节点从一采 latent 重建时会带上 lock_audio，
    # 表示音频已经是成品、不许在精修里被重新发明。此时即使没有外部遮罩也走遮罩路径。
    lock_audio = bool(entries) and all(bool(entry.get("lock_audio")) for entry in entries)
    mask_path = mask_enabled or lock_audio
    pdd_steps = _native_model_info(model)
    if not callable(getattr(noise, "generate_noise", None)):
        raise ValueError("Connect a native NOISE output, such as RandomNoise, to the noise input.")
    seed = getattr(noise, "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError("NOISE.seed must be an integer in [0, 18446744073709551615].")
    high_cfg = _checked_number("high_cfg", high_cfg)
    low_cfg = _checked_number("low_cfg", low_cfg)
    cfg_split = _checked_number("cfg_split", cfg_split, upper=1)
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("Connect at least one encoded H3 window before sampling.")
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not {"window", "positive", "negative"}.issubset(entry):
            raise ValueError(f"H3 window {index} is missing its window or conditioning data.")
    if mask_path:
        missing = [index for index, entry in enumerate(entries, 1)
                   if not isinstance(entry.get("initial_video_latent"), torch.Tensor)]
        if missing:
            raise ValueError(
                "遮罩/二采需要源片初始 latent，但窗口 %s 没有。请检查上游 "
                "PROYY MiniMax H3 Reference to Window：① 一采打开 video_encode，"
                "它在 widgets 里位于 keep_final 之后；② 二采接 source_latent（一采放大后的 latent）；"
                "③ 确认 vae 已连接（用同一路视频 VAE）；"
                "④ preview_only 必须保持关闭，否则该节点的 windows 会被 ExecutionBlocker 拦住。"
                % ", ".join(map(str, missing)))
    if not callable(getattr(sampler, "sample", None)):
        raise ValueError("SAMPLER must provide a callable sample method.")
    function = getattr(sampler, "sampler_function", None)
    sampler_name = getattr(function, "__name__", type(sampler).__name__).removeprefix("sample_")
    schedule = validate_sigmas(sigmas)
    cfg_schedule = (high_cfg, cfg_split, low_cfg)

    # The masked path replaces preserved rows every step, so a same-sigma
    # prediction handoff would smear preserved pixels past the mask border.
    rolling_used = rolling if (rolling is not None and rolling.enabled and not mask_path) else None
    rolling_note = ("disabled: 遮罩/音频锁定时不使用同sigma交接"
                    if (rolling is not None and rolling.enabled and mask_path)
                    else ("enabled" if rolling_used is not None else "disabled"))

    cloned = model.clone()
    guider = MaskedCPUWindowGuider(cloned, entries, high_cfg, cfg_schedule=cfg_schedule,
                                   blend=blend, rolling=rolling_used)
    latent, initial_mode = _prepare_initial_latent(entries, guider.fusion)
    if mask_enabled:
        denoise_mask, mask_report = build_denoise_mask(mask, guider.fusion.shapes,
                                                       lock_audio=lock_audio)
    elif lock_audio:
        denoise_mask = _lock_audio_mask(guider.fusion.shapes)
        mask_report = {
            "enabled": True,
            "source": "二采（windows 自带 lock_audio）",
            "video_regenerate_ratio": 1.0,
            "audio_regenerate_ratio": 0.0,
            "audio_lock": True,
            "note": "视频整段按 sigmas 重绘；音频逐位保留一采结果，不参与采样。",
        }
    else:
        denoise_mask = None
        mask_report = {
            "enabled": False,
            "note": "未连接遮罩：不读遮罩，不产生遮罩cond，也不做latent混合。"
                    "整段画面重新生成，源片仅通过 <Video n> 参考通道影响生成。",
        }
    global_noise = _validate_global_noise(noise.generate_noise({"samples": latent}),
                                          guider.fusion.shapes)
    steps = schedule.numel() - 1
    # One cell per (window, step): the bar has to cover every window, otherwise
    # it fills up on the first one and the remaining windows show no movement.
    guider.progress_total = len(entries) * max(steps, 1)
    guider.window_progress = comfy.utils.ProgressBar(guider.progress_total)

    device = model.load_device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = guider.sample(global_noise, latent, sampler, schedule,
                           denoise_mask=denoise_mask, callback=None, seed=seed)
    report = {
        "windows": [{"start": e["window"].start, "frames": e["window"].frames} for e in entries],
        "frames": entries[-1]["window"].start + entries[-1]["window"].frames,
        "fps": 24, "global_state_device": "cpu",
        "window_calls": guider.window_calls, "model_evaluations": guider.model_evaluations,
        "sampler_callbacks": guider.sampler_callbacks,
        "steps": steps, "sampler": sampler_name, "seed": int(seed),
        "initial_latent_mode": initial_mode, "initial_audio_mode":
            "source_latent" if initial_mode == "source_av" else "empty",
        "audio_lock": lock_audio,
        "rolling": rolling_note,
        "rolling_strength": (round(float(rolling_used.strength), 3) if rolling_used is not None else None),
        "rolling_mode": (rolling_used.mode if rolling_used is not None else None),
        "fusion_preserve": round(float(guider.blend), 3),
        "fusion_rule": "-10: 上一窗占满整个重叠区; 0: 原生线性交叉渐变; "
                       "+10: 下一窗占满整个重叠区",
        "mask": mask_report,
        "start_sigma": float(schedule[0]), "sigmas": "external:SIGMAS",
        "schedule": [round(float(s), 6) for s in schedule],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "pdd_steps": pdd_steps,
        "model_variant": "pdd" if pdd_steps > 1 else "native_ref2va",
        "high_cfg": high_cfg, "cfg_split": cfg_split, "low_cfg": low_cfg,
        "cfg_rule": "sigma >= cfg_split: high_cfg; sigma < cfg_split: low_cfg",
    }
    if rolling_used is not None and hasattr(guider.fusion, "report"):
        rrep = guider.fusion.report()
        report["rolling_curve"] = rrep.get("curve")
        report["rolling_steps"] = rrep.get("steps")
        report["rolling_video_latent_positions"] = rrep.get("video_latent_positions")
    if device.type == "cuda":
        report["peak_cuda_allocated_MiB"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        report["peak_cuda_reserved_MiB"] = round(torch.cuda.max_memory_reserved(device) / 2**20, 2)
    return {"samples": output.cpu()}, report


class PROYYMaskedSampler:
    """PROYAYA H3 window sampler with an optional per-window denoise mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise": ("NOISE", {"lazy": True,
                "tooltip": "官方 RandomNoise；只为完整全局音视频 latent 生成一次初始噪声。"}),
            "model": ("MODEL", {"lazy": True}),
            "windows": ("H3_JOINT_WINDOWS", {
                "tooltip": "PROYY MiniMax H3 Reference to Window 的 windows；接入遮罩时每个窗口必须带源片初始 latent。"}),
            "sampler": ("SAMPLER", {"lazy": True}),
            "sigmas": ("SIGMAS", {"lazy": True,
                "tooltip": "首值在 (0,1]，严格递减并以 0 结尾。"}),
            "high_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                "tooltip": "当前 sigma >= cfg_split 时使用。"}),
            "cfg_split": ("FLOAT", {"default": 0.79, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "按实际采样 sigma 分界，不是采样进度百分比。"}),
            "low_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                "tooltip": "当前 sigma < cfg_split 时使用。"}),
        }, "optional": {
            "mask": ("MASK", {
                "tooltip": "白色 = 要重绘的区域，逐帧遮罩按窗口时间切片，静态遮罩请只给一帧。"
                           "不连接时本节点等同无遮罩的窗口采样器：整段重新生成，源片仅走 <Video n> 参考通道。"
                           "需要反相请在外部节点处理，本节点固定白色=重绘。"}),
            "fusion_preserve": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                "tooltip": "窗间重叠区保留哪一窗的倾向，只影响重叠区。"
                           "0 = 原生线性交叉渐变；"
                           "-10 = 重叠区整段由上一窗（先进内容）保留 100%，下一窗不参与；"
                           "+10 = 重叠区整段由下一窗（后进内容）保留 100%，上一窗不参与。"
                           "两条权重始终互补，和恒为 1。"}),
            # 默认关闭，保持既有工作流逐位可复现；开启后才走同 sigma 交接。
            "rolling_enabled": ("BOOLEAN", {"default": False,
                "label_on": "同sigma交接 开",
                "label_off": "同sigma交接 关",
                "tooltip": "同一步内把上一窗的 post-CFG x0 预测写进下一窗的重叠段，"
                           "再评估下一窗，让后段直接沿用前段的结论（如衣物滑落后的状态）。"
                           "仅在遮罩未接入时可用；接入遮罩时本项被忽略。"}),
            "rolling_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "交接强度。β 在每个 sigma 上都是这个常数（flat 曲线），"
                           "σ=1 时也会执行（resample 模式下那里没有信息可传）。"}),
            "rolling_mode": (list(ROLLING_MODES), {"default": ROLLING_MODES[0],
                "tooltip": "resample = 按 (1−σ)·x0+σ·ε 把上一窗还原到当前噪声位，信息上限 1−σ，"
                           "4 步时最高只有 20%；hard = 直接注入上一窗的 x0，不走重采样，"
                           "传输量等于 strength，但高 σ 阶段会强行压入未收敛的 x0（重叠区会发亮）。"
                           "front = 同 hard，但只在高 σ 生效；"
                           "front_resample = 同 resample，但只在 [low, high] 区间生效。"}),
            "rolling_front_high": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "front / front_resample 曲线的满强度上限：σ ≥ 该值时 β = strength。"}),
            "rolling_front_low": ("FLOAT", {"default": 0.63, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "front / front_resample 曲线的归零下限：σ ≤ 该值时 β = 0，"
                           "两者之间线性过渡。必须小于上限。"}),
            "unload": ("BOOLEAN", {"default": True,
                "tooltip": "采样结束后卸载模型与克隆。"}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("joint_latent", "run_report")
    FUNCTION = "sample"
    CATEGORY = "PROYAYA/H3"
    DESCRIPTION = ("H3 joint CPU window sampling with an optional denoise mask. Preserved rows fall "
                   "back to the source latent every step; rolling transfer is disabled while a mask "
                   "is connected.")

    def check_lazy_status(self, noise=None, model=None, sampler=None, sigmas=None, **kwargs):
        """只请求四个重输入。

        ``mask`` 是可选输入且不 lazy：连了就由 ComfyUI 正常求值，没连就根本不会出现，
        所以这里不能再返回它 —— 对没有连线的输入执行 make_input_strong_link 会直接抛错。
        """
        return [name for name, value in (("noise", noise), ("model", model),
                                         ("sampler", sampler), ("sigmas", sigmas))
                if value is None]

    def sample(self, noise, model, windows, sampler, sigmas, high_cfg, cfg_split, low_cfg,
               mask=None, fusion_preserve=0.0, rolling_enabled=False, rolling_strength=1.0,
               rolling_mode=ROLLING_MODES[0], rolling_front_high=0.9, rolling_front_low=0.63,
               unload=True):
        try:
            latent, report = sample_masked_windows(
                noise, model, windows, sampler, sigmas, mask,
                high_cfg, cfg_split, low_cfg,
                blend=fusion_preserve,
                rolling=RollingConfig(bool(rolling_enabled), float(rolling_strength),
                                      str(rolling_mode), float(rolling_front_high),
                                      float(rolling_front_low)))
        finally:
            if unload:
                comfy.model_management.unload_model_and_clones(model)
        report["unloaded"] = bool(unload)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"text": [text]}, "result": (latent, text)}


NODE_CLASS_MAPPINGS = {"PROYYMaskedSampler": PROYYMaskedSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"PROYYMaskedSampler": "۞▷Window Sampler◁۞"}
