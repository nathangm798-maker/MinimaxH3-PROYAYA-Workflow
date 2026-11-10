"""V1.1 contracts using installed ComfyUI Euler/CFGGuider and real CFG math.

Only checkpoint evaluation and model loading are replaced. No checkpoint,
CUDA allocation, source video, or generated media is required.
"""

import copy
import importlib
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime as base_tests

torch = base_tests.torch
comfy = base_tests.comfy
NestedTensor = base_tests.NestedTensor
runtime = base_tests.runtime
windows = base_tests.windows
continuity = runtime  # RollingWindowGuider lives in runtime
# PROYYSamplerCustomAdvanced 已移除；这些用例直接测 runtime.sample_windows_native。
rolling = importlib.import_module(base_tests.package_name + ".rolling")
native_sampling_function = comfy.samplers.sampling_function


class CloneablePatcher(base_tests.FakePatcher):
    def __init__(self):
        super().__init__()
        self.clone_calls = []

    def clone(self):
        cloned = copy.copy(self)
        cloned.model_options = copy.deepcopy(self.model_options)
        self.clone_calls.append(cloned)
        return cloned


class RealCFGHarness(base_tests.NativeHarness):
    def __init__(self, testcase, fail_marker=None):
        super().__init__(testcase, fail_marker=fail_marker)
        self.calls = []

    def __enter__(self):
        super().__enter__()
        self.stack.enter_context(mock.patch("comfy.samplers.calc_cond_batch", side_effect=self.cond_batch))
        return self

    def cond_batch(self, model, conds, x, timestep, model_options):
        return [torch.zeros_like(x) if group is None else
                0.125 * x + group[0]["marker"] * 0.01 + timestep.reshape(-1, 1, 1) * 0.02
                for group in conds]

    def predict(self, base, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
        super().predict(base, x, timestep, uncond, cond, cond_scale, model_options, seed)
        output = native_sampling_function(base, x, timestep, uncond, cond, cond_scale,
                                          model_options=model_options, seed=seed)
        self.calls.append({"marker": cond[0]["marker"], "sigma": float(timestep[0]),
                           "cfg": cond_scale, "input": x.clone(), "output": output.clone()})
        return output


class CountingNoise:
    seed = 381

    def __init__(self):
        self.calls = 0
        self.generated = None

    def generate_noise(self, latent):
        self.calls += 1
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.generated = NestedTensor([torch.randn(stream.shape, generator=generator)
                                       for stream in latent["samples"].unbind()])
        return self.generated


class ContinuityRuntimeTests(unittest.TestCase):
    sigmas = torch.tensor([1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0])

    def inputs(self, shapes):
        generator = torch.Generator(device="cpu").manual_seed(123)
        return (NestedTensor([torch.randn(s, generator=generator) for s in shapes]),
                NestedTensor([torch.zeros(s) for s in shapes]))

    def test_three_windows_eight_steps_real_cfg_and_same_sigma_propagation(self):
        entries = base_tests.entries(3)
        patcher = CloneablePatcher()
        # Verify commits include post-CFG hooks, not only positive or raw CFG.
        patcher.model_options["sampler_post_cfg_function"] = [lambda args: args["denoised"] + 0.75]
        patcher.model.model_sampling.noise_scale = 1.7
        guider = continuity.RollingWindowGuider(patcher, entries, 4.0,
                    cfg_schedule=(4.0, 0.5, 1.25), rolling=rolling.RollingConfig(True, 0.45))
        noise, latent = self.inputs(guider.fusion.shapes)
        original_noise = [part.clone() for part in noise.unbind()]
        callbacks = []
        with RealCFGHarness(self) as harness:
            output = guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), self.sigmas,
                                   callback=lambda *args: callbacks.append(args), disable_pbar=True, seed=123)
        self.assertEqual(guider.window_calls, 24)
        self.assertEqual([c["marker"] for c in harness.calls], [1, 2, 3] * 8)
        self.assertEqual([c["cfg"] for c in harness.calls], [4.0] * 15 + [1.25] * 9)
        self.assertEqual(len(callbacks), 8)
        self.assertEqual(len(harness.prepared), 3)
        self.assertTrue(harness.solver_values)
        for step in range(1, 8):
            sigma = float(self.sigmas[step])
            # Flat curve: every sigma transfers the same fraction.
            beta = 0.45
            for index in (1, 2):
                donor, receiver = harness.calls[step * 3 + index - 1:step * 3 + index + 1]
                donor_x = windows.unpack(donor["input"], entries[index - 1]["window"].shapes)
                donor_x0 = windows.unpack(donor["output"], entries[index - 1]["window"].shapes)
                received = windows.unpack(receiver["input"], entries[index]["window"].shapes)
                video_start, audio_start = entries[index]["window"].video_start, entries[index]["window"].audio_start
                for actual, old, clean, epsilon in (
                    (received[0][:, :, :7], donor_x[0][:, :, -7:], donor_x0[0][:, :, -7:],
                     original_noise[0][:, :, video_start:video_start + 7]),
                    (received[1][..., :37], donor_x[1][..., -37:], donor_x0[1][..., -37:],
                     original_noise[1][..., audio_start:audio_start + 37]),
                ):
                    expected = (1 - beta) * old + beta * ((1 - sigma) * clean + sigma * 1.7 * epsilon)
                    torch.testing.assert_close(actual, expected)
        for actual, original in zip(noise.unbind(), original_noise):
            self.assertTrue(torch.equal(actual, original))
        self.assertEqual([tuple(part.shape) for part in output.unbind()], guider.fusion.shapes)
        self.assertTrue(all(part.device.type == "cpu" for part in output.unbind()))
        self.assertIsNone(guider.fusion.noise)
        self.assertIsNone(guider.fusion.sigma)
        report = guider.continuity_report()
        self.assertEqual(len(report["steps"]), 8)
        # A flat curve transfers at every sigma, including the first step.
        self.assertEqual(report["video_latent_positions"], 8 * 14)
        self.assertEqual(report["audio_latent_positions"], 8 * 74)
        self.assertEqual(report["steps"][0]["effective_strength"], 0.45)

    def test_progress_bar_spans_every_window_not_just_the_steps(self):
        """The bar must cover windows * steps, or it fills up on window 1."""
        entries = base_tests.entries(3)
        guider = continuity.RollingWindowGuider(CloneablePatcher(), entries, 4.0)
        noise, latent = self.inputs(guider.fusion.shapes)
        steps = len(self.sigmas) - 1
        seen = []

        class Recorder:
            def update_absolute(self, value, *args, **kwargs):
                seen.append(int(value))

        guider.window_progress = Recorder()
        guider.progress_total = len(entries) * steps
        with RealCFGHarness(self):
            guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), self.sigmas,
                          disable_pbar=True, seed=123)

        self.assertTrue(seen, "progress bar was never updated")
        self.assertEqual(seen, sorted(seen), "progress must never move backwards")
        self.assertEqual(seen[-1], guider.progress_total, "bar must end exactly full")
        # The final cell is pushed to 100% when outer_sample returns, so inspect
        # the cells before it: with the old bug they never exceeded one window.
        self.assertGreater(
            max(seen[:-1] or [0]), steps,
            "progress never left the first window: %d cells for %d windows"
            % (max(seen[:-1] or [0]), len(entries)))

    def test_disabled_strength_zero_and_single_window_match_v1_bit_exact(self):
        for count, config in ((3, rolling.RollingConfig(False, 0.45)),
                              (3, rolling.RollingConfig(True, 0.0)),
                              (1, rolling.RollingConfig(True, 0.45))):
            with self.subTest(count=count, config=config):
                entries = base_tests.entries(count)
                baseline = runtime.CPUWindowGuider(CloneablePatcher(), entries, 3.0, cfg_schedule=(3.0, 0.5, 1.2))
                candidate = continuity.RollingWindowGuider(CloneablePatcher(), entries, 3.0,
                              cfg_schedule=(3.0, 0.5, 1.2), rolling=config)
                noise, latent = self.inputs(baseline.fusion.shapes)
                with RealCFGHarness(self):
                    expected = baseline.sample(noise, latent, comfy.samplers.sampler_object("euler"), self.sigmas,
                                               disable_pbar=True, seed=123)
                    actual = candidate.sample(noise, latent, comfy.samplers.sampler_object("euler"), self.sigmas,
                                              disable_pbar=True, seed=123)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual.unbind(), expected.unbind())))

    def test_failure_cleans_rolling_state_and_runtime_and_can_retry(self):
        patcher = CloneablePatcher()
        guider = continuity.RollingWindowGuider(patcher, base_tests.entries(3), 2.0)
        noise, latent = self.inputs(guider.fusion.shapes)
        original_shapes, original_options = patcher.model.latent_shapes, guider.model_options
        with RealCFGHarness(self, fail_marker=2):
            with self.assertRaisesRegex(RuntimeError, "injected window prediction failure"):
                guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), self.sigmas,
                              disable_pbar=True, seed=123)
        self.assertIsNone(guider.fusion.noise)
        self.assertIsNone(guider.fusion.sigma)
        self.assertEqual(guider.fusion.steps, [])
        self.assertEqual(guider.window_conds, [])
        self.assertFalse(hasattr(guider, "inner_model"))
        self.assertIs(patcher.model.latent_shapes, original_shapes)
        self.assertIs(guider.model_options, original_options)
        self.assertEqual(patcher.cleanups, 1)
        with RealCFGHarness(self):
            actual = guider.sample(noise, latent, comfy.samplers.sampler_object("euler"), self.sigmas,
                                   disable_pbar=True, seed=123)
            fresh = continuity.RollingWindowGuider(CloneablePatcher(), base_tests.entries(3), 2.0)
            expected = fresh.sample(noise, latent, comfy.samplers.sampler_object("euler"), self.sigmas,
                                    disable_pbar=True, seed=123)
        self.assertEqual(guider.window_calls, 24)
        self.assertEqual(len(guider.fusion.steps), 8)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual.unbind(), expected.unbind())))

    def test_native_sampling_generates_noise_once_returns_av_and_reports(self):
        source, model = CountingNoise(), CloneablePatcher()
        with RealCFGHarness(self) as harness, \
                mock.patch.object(runtime, "_native_model_info", return_value=1):
            latent, report = runtime.sample_windows_native(
                source, model, base_tests.entries(3), comfy.samplers.sampler_object("euler"), self.sigmas,
                3.0, 0.5, 1.2, rolling=rolling.RollingConfig(True, 0.45))
        self.assertEqual(source.calls, 1)
        self.assertEqual(len(model.clone_calls), 1)
        self.assertEqual(len(harness.predictions), 24)
        self.assertEqual([tuple(part.shape) for part in latent["samples"].unbind()],
                         [(1, 24, 97, 2, 2), (1, 32, 2, 547)])
        self.assertEqual(report["noise_generate_calls"], 1)
        self.assertEqual(report["window_calls"], 24)
        self.assertEqual(report["steps"], 8)
        self.assertEqual(report["frames"], 328)
        self.assertEqual(report["continuity"]["mechanism"], "intra_step_x0_rolling")
        self.assertEqual(len(report["continuity"]["steps"]), 8)
        self.assertEqual(report["high_cfg"], 3.0)
        self.assertEqual(report["low_cfg"], 1.2)

    def test_cloned_sampling_override_sets_same_noise_scale_for_solver_and_rolling(self):
        class OverriddenPatcher(CloneablePatcher):
            def __init__(self):
                super().__init__()
                self.model.model_sampling.noise_scale = 1.0
                self.effective_sampling = copy.deepcopy(self.model.model_sampling)
                self.effective_sampling.noise_scale = 2.3

            def get_model_object(self, name):
                if name == "model_sampling":
                    return self.effective_sampling
                return super().get_model_object(name)

        source, model = CountingNoise(), OverriddenPatcher()
        unpatched_sampling = model.model.model_sampling
        entries = base_tests.entries(3)
        applied_to = []
        with RealCFGHarness(self) as harness:
            def apply_effective_sampling(patcher, shape, conds, model_options):
                # ModelPatcher object overrides are visible through
                # get_model_object before prepare_sampling applies them to the
                # underlying base model. The node must read the effective value
                # even though the base still has noise_scale=1 at start_run.
                self.assertIs(patcher, model.clone_calls[0])
                self.assertIs(patcher.model.model_sampling, unpatched_sampling)
                self.assertEqual(patcher.model.model_sampling.noise_scale, 1.0)
                patcher.model.model_sampling = patcher.get_model_object("model_sampling")
                applied_to.append(patcher)
                return harness.prepare_sampling(patcher, shape, conds, model_options)

            with mock.patch.object(runtime, "_native_model_info", return_value=1), \
                    mock.patch.object(model.model, "model_sampling", unpatched_sampling), \
                    mock.patch("comfy.sampler_helpers.prepare_sampling", side_effect=apply_effective_sampling):
                _latent, report = runtime.sample_windows_native(
                    source, model, entries, comfy.samplers.sampler_object("euler"),
                    torch.tensor([1.0, 0.5, 0.0]), 3.0, 0.5, 1.2,
                    rolling=rolling.RollingConfig(True, 0.6))

        self.assertEqual(len(applied_to), 1)
        self.assertIs(model.model.model_sampling, unpatched_sampling)
        self.assertEqual(source.calls, 1)
        self.assertEqual(report["continuity"]["noise_scale"], 2.3)
        self.assertEqual(len(harness.calls), 6)
        original_noise = source.generated.unbind()
        initial_local_noise = windows.pack(windows.crop(original_noise, entries[0]["window"]))
        # The real Euler solver initializes from the *applied* object override.
        torch.testing.assert_close(harness.calls[0]["input"], initial_local_noise * 2.3)
        for index in (1, 2):
            donor, receiver = harness.calls[3 + index - 1:3 + index + 1]
            donor_x = windows.unpack(donor["input"], entries[index - 1]["window"].shapes)
            donor_x0 = windows.unpack(donor["output"], entries[index - 1]["window"].shapes)
            received = windows.unpack(receiver["input"], entries[index]["window"].shapes)
            video_start, audio_start = entries[index]["window"].video_start, entries[index]["window"].audio_start
            for actual, old, clean, epsilon in (
                (received[0][:, :, :7], donor_x[0][:, :, -7:], donor_x0[0][:, :, -7:],
                 original_noise[0][:, :, video_start:video_start + 7]),
                (received[1][..., :37], donor_x[1][..., -37:], donor_x0[1][..., -37:],
                 original_noise[1][..., audio_start:audio_start + 37]),
            ):
                expected = 0.4 * old + 0.6 * (0.5 * clean + 0.5 * 2.3 * epsilon)
                stale_base_scale = 0.4 * old + 0.6 * (0.5 * clean + 0.5 * epsilon)
                torch.testing.assert_close(actual, expected)
                self.assertFalse(torch.allclose(actual, stale_base_scale))

    def test_active_rolling_accepts_native_res_multistep(self):
        source = CountingNoise()
        with RealCFGHarness(self), mock.patch.object(runtime, "_native_model_info", return_value=1):
            _latent, report = runtime.sample_windows_native(
                source, CloneablePatcher(), base_tests.entries(3), comfy.samplers.sampler_object("res_multistep"),
                self.sigmas, 3.0, 0.5, 1.2, rolling=rolling.RollingConfig(True, 0.45))
        self.assertEqual(source.calls, 1)
        self.assertEqual(report["window_calls"], 24)
        self.assertTrue(report["continuity"]["enabled"])
        # A flat curve transfers at every sigma, including the first step.
        self.assertEqual(report["continuity"]["video_latent_positions"], 112)

    def test_disabled_rolling_accepts_v1_native_res_multistep(self):
        source = CountingNoise()
        with RealCFGHarness(self), mock.patch.object(runtime, "_native_model_info", return_value=1):
            _latent, report = runtime.sample_windows_native(
                source, CloneablePatcher(), base_tests.entries(3), comfy.samplers.sampler_object("res_multistep"),
                self.sigmas, 3.0, 0.5, 1.2, rolling=rolling.RollingConfig(False))
        self.assertEqual(source.calls, 1)
        self.assertEqual(report["window_calls"], 24)
        self.assertFalse(report["continuity"]["enabled"])
        self.assertEqual(report["continuity"]["video_latent_positions"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
