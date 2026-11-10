"""Real CPU solver coverage for freely selected H3 AV samplers.

Neural-model evaluation/loading are replaced; native KSAMPLER, CFG, Euler,
multistage, multistep, stochastic, Brownian and CFG++ implementations are real.
"""

from functools import partial
import importlib
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_source_latent_runtime as source_tests

continuity_tests = source_tests.continuity_tests
torch = source_tests.torch
comfy = source_tests.comfy
runtime = source_tests.runtime
windows = source_tests.windows
continuity = source_tests.continuity
rolling = source_tests.rolling
sampling = importlib.import_module("comfy.k_diffusion.sampling")


class SamplerHarness(source_tests.InitialHarness):
    def __init__(self, testcase, schedule):
        super().__init__(testcase)
        self.original_schedule = schedule.clone()
        self.pdd_selections = []

    def predict(self, base, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
        sigma = timestep.reshape(-1)[0]
        exposed = model_options["transformer_options"]["sample_sigmas"]
        # Exercise the actual installed FinalLayer's nearest-point/next-point
        # lookup contract, not a helper that repeats the new adapter formula.
        selected = int((exposed - sigma).abs().argmin())
        next_sigma = exposed[min(selected + 1, len(exposed) - 1)]
        lower = self.original_schedule[self.original_schedule < sigma.cpu()]
        expected_next = lower[0] if len(lower) else torch.tensor(0.0)
        self.testcase.assertAlmostEqual(float(next_sigma), float(expected_next), places=6)
        self.pdd_selections.append((float(sigma), float(next_sigma), exposed.clone()))
        return super().predict(base, x, timestep, uncond, cond, cond_scale, model_options, seed)


class SamplerChoiceTests(unittest.TestCase):
    schedule = torch.tensor([0.9, 0.6, 0.3, 0.1, 0.0])

    def run_sampler(self, sampler, *, enabled=True, seed=381, schedule=None,
                    high_cfg=2.0, low_cfg=1.25):
        schedule = self.schedule if schedule is None else schedule
        entries = source_tests.source_entries()
        originals = [entry["initial_video_latent"].clone() for entry in entries]
        model, noise = source_tests.SourcePatcher(), continuity_tests.CountingNoise()
        noise.seed = seed
        self.assertEqual(model.model.diffusion_model.final_layer.pdd_steps, 32)
        config = rolling.RollingConfig(enabled, 0.45)
        original_sigmas = schedule.clone()
        with SamplerHarness(self, schedule) as harness:
            latent, report = runtime.sample_windows_native(
                noise, model, entries, sampler, schedule, high_cfg, 0.5, low_cfg,
                rolling=config)
        self.assertEqual(noise.calls, 1)
        self.assertTrue(torch.equal(schedule, original_sigmas))
        for entry, original in zip(entries, originals):
            self.assertTrue(torch.equal(entry["initial_video_latent"], original))
        expected_shapes = [(1, 24, 97, 2, 2), (1, 32, 2, 547)]
        self.assertEqual([tuple(t.shape) for t in latent["samples"].unbind()], expected_shapes)
        self.assertTrue(all(t.device.type == "cpu" and t.dtype == torch.float32
                            and bool(torch.isfinite(t).all()) for t in latent["samples"].unbind()))
        self.assertEqual(report["noise_generate_calls"], 1)
        self.assertEqual(report["pdd_steps"], 32)
        self.assertEqual(report["initial_latent_mode"], "source_video")
        self.assertEqual(report["initial_audio_mode"], "empty")
        self.assertEqual(report["steps"], len(schedule) - 1)
        self.assertEqual(report["window_calls"], len(harness.calls))
        self.assertEqual([call["marker"] for call in harness.calls], [1, 2, 3] * (len(harness.calls) // 3))
        return latent, report, harness, noise, model

    def test_native_solver_families_run_source_09_with_rolling_on_and_off(self):
        expected_evaluations = {
            "euler": 4, "heun": 7, "dpm_2": 7, "dpmpp_2m": 4,
            "euler_ancestral": 4, "dpmpp_2m_sde": 4, "res_multistep": 4,
            "euler_cfg_pp": 4, "dpmpp_2m_cfg_pp": 4,
        }
        for sampler_name, count in expected_evaluations.items():
            for enabled in (False, True):
                with self.subTest(sampler=sampler_name, rolling=enabled):
                    _, report, harness, _, _ = self.run_sampler(
                        comfy.samplers.sampler_object(sampler_name), enabled=enabled)
                    self.assertEqual(report["window_calls"], 3 * count)
                    self.assertEqual(report["model_evaluations"], count)
                    self.assertEqual(len(report["continuity"]["steps"]), count)
                    if enabled:
                        self.assertEqual(report["continuity"]["video_latent_positions"], 14 * count)
                    else:
                        self.assertEqual(report["continuity"]["video_latent_positions"], 0)
                    for call in harness.calls:
                        self.assertEqual(call["cfg"], 2.0 if call["sigma"] >= 0.5 else 1.25)

    def test_dpm2_midpoint_uses_next_lower_schedule_without_skipping_interval(self):
        _, report, harness, _, _ = self.run_sampler(comfy.samplers.sampler_object("dpm_2"))
        actual_sigmas = [point[0] for point in harness.pdd_selections[::3]]
        expected_sigmas = [0.9, math.sqrt(0.9 * 0.6), 0.6,
                           math.sqrt(0.6 * 0.3), 0.3, math.sqrt(0.3 * 0.1), 0.1]
        for actual, expected in zip(actual_sigmas, expected_sigmas):
            self.assertAlmostEqual(actual, expected, places=6)
        selected_next = [point[1] for point in harness.pdd_selections[::3]]
        for actual, expected in zip(selected_next, [0.6, 0.6, 0.3, 0.3, 0.1, 0.1, 0.0]):
            self.assertAlmostEqual(actual, expected, places=6)
        # The original nearest+next rule fails at the first midpoint because
        # sqrt(.9*.6) is nearer .6, then it incorrectly selects .3.
        midpoint = torch.tensor(expected_sigmas[1])
        old_next = self.schedule[int((self.schedule - midpoint).abs().argmin()) + 1]
        self.assertAlmostEqual(float(old_next), 0.3, places=6)
        self.assertAlmostEqual(selected_next[1], 0.6, places=6)
        self.assertEqual(report["steps"], 4)
        self.assertEqual(report["model_evaluations"], 7)

    def test_cfgpp_receives_fused_global_unconditional_prediction_once_per_evaluation(self):
        hook_calls = []
        original_register = comfy.model_patcher.set_model_options_post_cfg_function
        active_harness = []

        def register(options, function, **kwargs):
            if getattr(function, "__module__", "") == sampling.__name__:
                def inspect_global(args, original_function=function):
                    harness = active_harness[0]
                    shape = (1, 1, 24 * 97 * 2 * 2 + 32 * 2 * 547)
                    self.assertEqual(tuple(args["input"].shape), shape)
                    self.assertEqual(tuple(args["denoised"].shape), shape)
                    self.assertEqual(tuple(args["uncond_denoised"].shape), shape)
                    self.assertEqual(args["uncond_denoised"].device.type, "cpu")
                    latest = harness.calls[-3:]
                    self.assertEqual([call["marker"] for call in latest], [1, 2, 3])
                    fusion = windows.CPUFusion([entry["window"] for entry in source_tests.source_entries()])
                    values = [0.125 * call["input"] - call["marker"] * 0.01 + call["sigma"] * 0.02
                              for call in latest]
                    expected = fusion.predict(torch.zeros(shape), lambda index, local: values[index])
                    torch.testing.assert_close(args["uncond_denoised"], expected)
                    hook_calls.append((float(args["sigma"][0]), shape))
                    return original_function(args)
                function = inspect_global
            return original_register(options, function, **kwargs)

        # Real native CFG++ installs its own callback during sampler execution.
        # CFG=1 additionally proves that CFG++ keeps unconditional evaluation.
        model, noise = source_tests.SourcePatcher(), continuity_tests.CountingNoise()
        with SamplerHarness(self, self.schedule) as harness, \
                mock.patch.object(comfy.model_patcher, "set_model_options_post_cfg_function", side_effect=register):
            active_harness.append(harness)
            _, report = runtime.sample_windows_native(
                noise, model, source_tests.source_entries(), comfy.samplers.sampler_object("euler_cfg_pp"),
                self.schedule, 1.0, 0.5, 1.0,
                rolling=rolling.RollingConfig(True, 0.45))
        self.assertEqual(len(hook_calls), 4)
        self.assertEqual(report["window_calls"], 12)
        self.assertEqual(noise.calls, 1)

    def test_stochastic_solvers_repeat_seed_and_consume_solver_noise_separately(self):
        for sampler_name in ("euler_ancestral", "dpmpp_2m_sde"):
            for enabled in (False, True):
                with self.subTest(sampler=sampler_name, rolling=enabled):
                    first = self.run_sampler(comfy.samplers.sampler_object(sampler_name), enabled=enabled, seed=37)
                    second = self.run_sampler(comfy.samplers.sampler_object(sampler_name), enabled=enabled, seed=37)
                    other = self.run_sampler(comfy.samplers.sampler_object(sampler_name), enabled=enabled, seed=38)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                        first[0]["samples"].unbind(), second[0]["samples"].unbind())))
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                        first[0]["samples"].unbind(), other[0]["samples"].unbind())))
        solver_noise_calls = []
        original_default = sampling.default_noise_sampler

        def capture_noise(x, seed=None):
            inner = original_default(x, seed)

            def draw(sigma, sigma_next):
                result = inner(sigma, sigma_next)
                solver_noise_calls.append(tuple(result.shape))
                return result
            return draw

        with mock.patch.object(sampling, "default_noise_sampler", side_effect=capture_noise):
            _, report, _, noise, _ = self.run_sampler(comfy.samplers.sampler_object("euler_ancestral"))
        self.assertGreater(len(solver_noise_calls), 1)
        self.assertTrue(all(shape == (1, 1, 24 * 97 * 2 * 2 + 32 * 2 * 547) for shape in solver_noise_calls))
        self.assertEqual(noise.calls, 1)
        self.assertEqual(report["noise_generate_calls"], 1)

    def test_custom_sampler_object_and_extra_options_reach_connected_callable_unchanged(self):
        observed = []
        marker = object()

        def selected_function(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            observed.append(kwargs)
            self.assertIs(kwargs["marker"], marker)
            self.assertEqual(kwargs["custom_amount"], 0.375)
            return sampling.sample_heun(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

        class SelectedSampler(comfy.samplers.KSAMPLER):
            def sample(self, *args, **kwargs):
                self.seen_self = self
                self.options_at_call = self.extra_options
                return super().sample(*args, **kwargs)

        options = {"marker": marker, "custom_amount": 0.375}
        sampler = SelectedSampler(selected_function, extra_options=options)
        _, report, _, _, _ = self.run_sampler(sampler)
        self.assertIs(sampler.seen_self, sampler)
        self.assertIs(sampler.options_at_call, options)
        self.assertIs(sampler.extra_options, options)
        self.assertEqual(len(observed), 1)
        self.assertEqual(set(options), {"marker", "custom_amount"})
        self.assertEqual(report["window_calls"], 21)

    def test_user_step_counts_and_partial_sigma_schedules_are_preserved(self):
        for steps in (1, 2, 8, 16):
            with self.subTest(steps=steps):
                # Deliberately nonuniform and unlike the default H3 scheduler.
                schedule = 0.9 * torch.linspace(1.0, 0.0, steps + 1).square()
                _, report, harness, _, _ = self.run_sampler(
                    comfy.samplers.sampler_object("euler"), schedule=schedule)
                self.assertEqual(report["steps"], steps)
                self.assertEqual(report["model_evaluations"], steps)
                self.assertEqual(report["window_calls"], 3 * steps)
                self.assertEqual(report["schedule"], [round(float(s), 6) for s in schedule])
                observed = torch.tensor([call["sigma"] for call in harness.calls[::3]])
                torch.testing.assert_close(observed, schedule[:-1], rtol=0, atol=0)

    def test_custom_sampler_wrong_shape_or_nonfinite_output_is_rejected_and_cleaned(self):
        for defect in ("shape", "nonfinite"):
            with self.subTest(defect=defect):
                def invalid_result(model, x, sigmas, **kwargs):
                    return x[..., :-1] if defect == "shape" else torch.full_like(x, float("nan"))

                self.assert_sampler_failure_cleanup(comfy.samplers.KSAMPLER(invalid_result),
                                                    "shape|finite")

    def test_unipc_working_schedule_mutation_does_not_change_pdd_grid_or_report(self):
        # Installed UniPC assigns .001 into a view of sigmas[-1]. Its internal
        # endpoint is native solver behavior; the original PDD grid and report
        # must retain the supplied zero endpoint and not alias that work tensor.
        schedule = torch.tensor([0.9, 0.6, 0.3, 0.0])
        for sampler_name in ("uni_pc", "uni_pc_bh2"):
            with self.subTest(sampler=sampler_name):
                _, report, harness, _, _ = self.run_sampler(
                    comfy.samplers.sampler_object(sampler_name), schedule=schedule)
                self.assertEqual(report["schedule"], [0.9, 0.6, 0.3, 0.0])
                self.assertEqual(float(schedule[-1]), 0.0)
                self.assertGreater(report["model_evaluations"], 0)
                # SamplerHarness checks next-lower PDD selection at *every*
                # actual call, including UniPC's .001 endpoint evaluation.
                self.assertTrue(any(abs(sigma - 0.3) < 1e-6 and next_sigma == 0.0
                                    for sigma, next_sigma, _ in harness.pdd_selections))

    def test_custom_sampler_actual_sigma_outside_model_domain_is_rejected_and_cleaned(self):
        for sigma in (1.01, -0.01, float("nan")):
            with self.subTest(sigma=sigma):
                def invalid_sigma(model, x, sigmas, extra_args=None, **kwargs):
                    return model(x, x.new_tensor([sigma]), **extra_args)

                self.assert_sampler_failure_cleanup(comfy.samplers.KSAMPLER(invalid_sigma),
                                                    "sigma domain|finite H3 sigma|sigma in")

    def assert_sampler_failure_cleanup(self, sampler, message):
        model, noise = source_tests.SourcePatcher(), continuity_tests.CountingNoise()
        guiders = []

        real_guider = runtime.RollingWindowGuider

        def make_guider(*args, **kwargs):
            guider = real_guider(*args, **kwargs)
            guiders.append(guider)
            return guider

        old_shapes = model.model.latent_shapes
        with SamplerHarness(self, self.schedule), \
                mock.patch.object(runtime, "RollingWindowGuider", side_effect=make_guider), \
                self.assertRaisesRegex(ValueError, message):
            runtime.sample_windows_native(noise, model, source_tests.source_entries(), sampler,
                                          self.schedule, 2.0, 0.5, 1.25,
                                          rolling=rolling.RollingConfig(True, 0.45))
        self.assertEqual(noise.calls, 1)
        self.assertIs(model.model.latent_shapes, old_shapes)
        self.assertEqual(model.clone_calls[0].cleanups, 1)
        guider = guiders[0]
        self.assertIsNone(guider.fusion.noise)
        self.assertIsNone(guider.fusion.sigma)
        self.assertEqual(guider.window_conds, [])
        self.assertFalse(hasattr(guider, "inner_model"))

    def test_model_hooks_remain_local_and_multiple_solver_hooks_run_globally_in_order(self):
        events, pending = [], []
        global_shape = (1, 1, 24 * 97 * 2 * 2 + 32 * 2 * 547)
        local_shape = (1, 1, 24 * 37 * 2 * 2 + 32 * 2 * 207)

        def local_hook(args):
            self.assertEqual(tuple(args["denoised"].shape), local_shape)
            events.append("local")
            return args["denoised"] + 0.3

        def global_first(args):
            self.assertEqual(tuple(args["denoised"].shape), global_shape)
            events.append("global_first")
            pending.append(args["denoised"].clone() + 2.0)
            return args["denoised"] + 2.0

        def global_second(args):
            self.assertEqual(tuple(args["denoised"].shape), global_shape)
            events.append("global_second")
            torch.testing.assert_close(args["denoised"], pending.pop(), rtol=0, atol=0)
            return args["denoised"] * 0.5

        def custom_hooks(model, x, sigmas, extra_args=None, callback=None, disable=None):
            extra_args = dict(extra_args)
            options = dict(extra_args["model_options"])
            options = comfy.model_patcher.set_model_options_post_cfg_function(options, global_first)
            options = comfy.model_patcher.set_model_options_post_cfg_function(options, global_second)
            extra_args["model_options"] = options
            return sampling.sample_euler(model, x, sigmas, extra_args, callback, disable)

        model, noise = source_tests.SourcePatcher(), continuity_tests.CountingNoise()
        model.model_options["sampler_post_cfg_function"] = [local_hook]
        with SamplerHarness(self, self.schedule):
            _, report = runtime.sample_windows_native(
                noise, model, source_tests.source_entries(), comfy.samplers.KSAMPLER(custom_hooks),
                self.schedule, 2.0, 0.5, 1.25,
                rolling=rolling.RollingConfig(True, 0.45))
        self.assertEqual(events, ["local", "local", "local", "global_first", "global_second"] * 4)
        self.assertEqual(pending, [])
        self.assertEqual(report["model_evaluations"], 4)
        self.assertEqual(report["sampler_callbacks"], 4)
        self.assertEqual(model.model_options["sampler_post_cfg_function"], [local_hook])

    def test_cfgpp_ud10_single_window_matches_native_with_cfg1_and_cfg_transitions(self):
        entry = source_tests.source_entries(1)[0]
        shapes = entry["window"].shapes
        raw = source_tests.NestedTensor([entry["initial_video_latent"].clone(), torch.zeros(shapes[1])])
        generator = torch.Generator(device="cpu").manual_seed(381)
        noise = source_tests.NestedTensor([torch.randn(shape, generator=generator) for shape in shapes])
        for high, low in ((1.0, 1.0), (1.0, 2.5), (2.5, 1.0)):
            with self.subTest(high_cfg=high, low_cfg=low):
                class ScheduledStock(comfy.samplers.CFGGuider):
                    def predict_noise(self, x, timestep, model_options=None, seed=None):
                        self.set_cfg(high if bool(timestep.reshape(-1)[0] >= 0.5) else low)
                        return super().predict_noise(x, timestep, model_options=model_options, seed=seed)

                stock = ScheduledStock(source_tests.SourcePatcher())
                stock.set_conds(entry["positive"], entry["negative"])
                stock.set_cfg(high)
                joint = continuity.RollingWindowGuider(
                    source_tests.SourcePatcher(), [entry], high, cfg_schedule=(high, 0.5, low),
                    rolling=rolling.RollingConfig(True, 0.45))
                with SamplerHarness(self, self.schedule):
                    expected = stock.sample(noise, raw, comfy.samplers.sampler_object("cfgpp_ud10_ab"),
                                            self.schedule, disable_pbar=True, seed=381)
                    actual = joint.sample(noise, raw, comfy.samplers.sampler_object("cfgpp_ud10_ab"),
                                          self.schedule, disable_pbar=True, seed=381)
                for left, right in zip(actual.unbind(), expected.unbind()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_cfgpp_ud10_global_branch_metadata_tracks_cfg1_optimization(self):
        original_register = comfy.model_patcher.set_model_options_post_cfg_function
        for high, low in ((1.0, 1.0), (1.0, 2.5), (2.5, 1.0)):
            with self.subTest(high_cfg=high, low_cfg=low):
                seen = []

                def register(options, function, **kwargs):
                    if getattr(function, "__module__", "") == sampling.__name__:
                        def inspect_branch(args, native_function=function):
                            cfg = args["cond_scale"]
                            expected_absent = cfg == 1.0
                            self.assertEqual(args["uncond"] is None, expected_absent)
                            self.assertIsNotNone(args["cond"])
                            # Native calc_cond_batch can return a zero tensor
                            # for an omitted branch. The None metadata tells
                            # cfgpp_ud10_ab to select cond_denoised instead.
                            self.assertIsInstance(args["uncond_denoised"], torch.Tensor)
                            if expected_absent:
                                self.assertEqual(torch.count_nonzero(args["uncond_denoised"]).item(), 0)
                            latest = harness.calls[-3:]
                            fusion = windows.CPUFusion([entry["window"] for entry in source_tests.source_entries()])
                            predicted = [0.125 * call["input"] + call["marker"] * 0.01 + call["sigma"] * 0.02
                                         for call in latest]
                            expected_cond = fusion.predict(torch.zeros_like(args["input"]),
                                                           lambda index, local: predicted[index])
                            torch.testing.assert_close(args["cond_denoised"], expected_cond)
                            seen.append((cfg, args["uncond"] is None))
                            return native_function(args)
                        function = inspect_branch
                    return original_register(options, function, **kwargs)

                model, noise = source_tests.SourcePatcher(), continuity_tests.CountingNoise()
                with SamplerHarness(self, self.schedule) as harness, \
                        mock.patch.object(comfy.model_patcher, "set_model_options_post_cfg_function", side_effect=register):
                    latent, report = runtime.sample_windows_native(
                        noise, model, source_tests.source_entries(), comfy.samplers.sampler_object("cfgpp_ud10_ab"),
                        self.schedule, high, 0.5, low,
                        rolling=rolling.RollingConfig(True, 0.45))
                expected_cfgs = [high, high, low, low]
                self.assertEqual(seen, [(cfg, cfg == 1.0) for cfg in expected_cfgs])
                self.assertEqual(report["model_evaluations"], 4)
                self.assertEqual(report["window_calls"], 12)
                self.assertTrue(all(bool(torch.isfinite(stream).all()) for stream in latent["samples"].unbind()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
