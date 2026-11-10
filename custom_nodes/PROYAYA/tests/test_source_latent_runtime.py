"""Source-initialized H3 sampling through real CFGGuider, CONST and Euler.

Model loading/evaluation is replaced, but H3's native input/output transforms,
packed AV audio scaling and the sampler's initial noise mixture remain real.
"""

from functools import partial
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
import test_continuity_runtime as continuity_tests

base_tests = continuity_tests.base_tests
torch = base_tests.torch
comfy = base_tests.comfy
runtime = base_tests.runtime
windows = base_tests.windows
NestedTensor = base_tests.NestedTensor
continuity = continuity_tests.continuity
validate_sigmas = importlib.import_module(base_tests.package_name + ".common").validate_sigmas
rolling = continuity_tests.rolling


class ScaledFormat:
    """Nonidentity VAE normalization exposes missing/double process_latent_in."""

    def process_in(self, latent):
        return latent * 2.0

    def process_out(self, latent):
        return latent / 2.0


class NativeInputBase(comfy.model_base.MiniMaxH3):
    def __init__(self):
        # Avoid constructing transformer weights while retaining the actual
        # MiniMaxH3 process_latent_in/out and _scale_audio_slice implementations.
        torch.nn.Module.__init__(self)
        self.model_sampling = base_tests.AVSampling()
        self.model_sampling.set_parameters(shift=12.0, audio_shift=3.0)
        self.latent_format = ScaledFormat()
        self.latent_shapes = [("original",)]
        self.diffusion_model = types.SimpleNamespace(final_layer=types.SimpleNamespace(pdd_steps=32))
        self.input_shapes = []
        self.output_shapes = []

    def process_latent_in(self, latent):
        self.input_shapes.append(tuple(tuple(shape) for shape in self.latent_shapes))
        return super().process_latent_in(latent)

    def process_latent_out(self, latent):
        self.output_shapes.append(tuple(tuple(shape) for shape in self.latent_shapes))
        return super().process_latent_out(latent)


class SourcePatcher(continuity_tests.CloneablePatcher):
    def __init__(self):
        super().__init__()
        self.model = NativeInputBase()

    def get_additional_models_with_key(self, key):
        return []


class InitialHarness(continuity_tests.RealCFGHarness):
    def __init__(self, testcase):
        super().__init__(testcase)
        self.schedules = []
        self.condition_initials = []

    def predict(self, base, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
        self.schedules.append(model_options["transformer_options"]["sample_sigmas"].clone())
        return super().predict(base, x, timestep, uncond, cond, cond_scale, model_options, seed)

    def process_conds(self, base, noise, conds, device, latent_image=None,
                      denoise_mask=None, seed=None, latent_shapes=None):
        self.condition_initials.append(latent_image.clone())
        return super().process_conds(base, noise, conds, device, latent_image=latent_image,
                                    denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)


def source_entries(count=3):
    entries = base_tests.entries(count)
    for index, entry in enumerate(entries):
        entry["initial_video_latent"] = torch.full(entry["window"].shapes[0], float(1 + index * 2))
    return entries


class SourceLatentRuntimeTests(unittest.TestCase):
    schedule = torch.tensor([0.9, 0.6, 0.0])

    def run_native(self, entries, schedule=None, config=None, noise_scale=1.0):
        model, noise = SourcePatcher(), continuity_tests.CountingNoise()
        model.model.model_sampling.noise_scale = noise_scale
        config = config or rolling.RollingConfig(False, 0.45)
        schedule = self.schedule if schedule is None else schedule
        with InitialHarness(self) as harness:
            latent, report = runtime.sample_windows_native(
                noise, model, entries, comfy.samplers.sampler_object("euler"), schedule,
                high_cfg=1.0, cfg_split=0.5, low_cfg=1.0,
                rolling=config)
        return latent, report, model, noise, harness

    def test_raw_source_latents_use_existing_overlap_weights_without_mutation(self):
        entries = source_entries()
        originals = [entry["initial_video_latent"].clone() for entry in entries]
        fusion = windows.CPUFusion([entry["window"] for entry in entries])
        latent, mode = runtime._prepare_initial_latent(entries, fusion)
        video, audio = latent.unbind()
        expected = torch.cat((torch.ones(30), 1 + 2 * torch.arange(1, 8) / 8,
                              torch.full((23,), 3.0), 3 + 2 * torch.arange(1, 8) / 8,
                              torch.full((30,), 5.0)))
        torch.testing.assert_close(video, expected.view(1, 1, -1, 1, 1).expand_as(video))
        self.assertEqual(mode, "source_video")
        self.assertEqual(tuple(video.shape), (1, 24, 97, 2, 2))
        self.assertEqual(tuple(audio.shape), (1, 32, 2, 547))
        self.assertEqual(torch.count_nonzero(audio), 0)
        for entry, original in zip(entries, originals):
            self.assertTrue(torch.equal(entry["initial_video_latent"], original))

    def test_source_sigma_09_is_noised_once_after_native_normalization(self):
        entries = source_entries()
        latent, report, model, noise, harness = self.run_native(entries, noise_scale=1.7)
        raw, _ = runtime._prepare_initial_latent(entries, windows.CPUFusion([e["window"] for e in entries]))
        # Actual MiniMaxH3 process_latent_in doubles our synthetic VAE format;
        # audio remains zero, while the noise is never audio-scaled.
        processed = [raw.unbind()[0] * 2.0, raw.unbind()[1]]
        initial = [self.schedule[0] * 1.7 * eps + (1 - self.schedule[0]) * clean
                   for eps, clean in zip(noise.generated.unbind(), processed)]
        for index, entry in enumerate(entries):
            torch.testing.assert_close(harness.calls[index]["input"], windows.pack(windows.crop(initial, entry["window"])))
            torch.testing.assert_close(harness.condition_initials[index], windows.pack(windows.crop(processed, entry["window"])))
        self.assertEqual(noise.calls, 1)
        self.assertEqual(model.model.input_shapes, [tuple(tuple(t.shape) for t in raw.unbind())])
        self.assertTrue(all(torch.equal(s, self.schedule) for s in harness.schedules))
        self.assertEqual(report["initial_latent_mode"], "source_video")
        self.assertEqual(report["initial_audio_mode"], "empty")
        self.assertAlmostEqual(report["start_sigma"], 0.9, places=6)
        self.assertEqual(report["pdd_steps"], 32)
        self.assertEqual(report["window_calls"], 6)
        self.assertEqual(report["noise_generate_calls"], 1)
        self.assertEqual([tuple(t.shape) for t in latent["samples"].unbind()],
                         [tuple(t.shape) for t in raw.unbind()])

    def test_sigma_09_with_empty_target_keeps_native_scaled_noise(self):
        entries = base_tests.entries(3)
        _, report, model, noise, harness = self.run_native(entries)
        for index, entry in enumerate(entries):
            expected = windows.pack(windows.crop(noise.generated.unbind(), entry["window"])) * self.schedule[0]
            torch.testing.assert_close(harness.calls[index]["input"], expected)
        self.assertEqual(model.model.input_shapes, [])
        self.assertEqual(report["initial_latent_mode"], "empty_av")
        self.assertAlmostEqual(report["start_sigma"], 0.9, places=6)
        self.assertEqual(noise.calls, 1)

    def test_sigma_one_source_mode_is_bit_exact_empty_mode_with_rolling(self):
        schedule = torch.tensor([1.0, 0.7, 0.0])
        config = rolling.RollingConfig(True, 0.45)
        source = self.run_native(source_entries(), schedule, config)
        empty = self.run_native(base_tests.entries(3), schedule, config)
        for a, b in zip(source[0]["samples"].unbind(), empty[0]["samples"].unbind()):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(source[4].calls, empty[4].calls):
            self.assertTrue(torch.equal(a["input"], b["input"]))
        self.assertEqual(source[3].calls, 1)
        self.assertEqual(empty[3].calls, 1)

    def test_rolling_at_initial_sigma_09_uses_original_noise_and_processed_state(self):
        entries = source_entries()
        _, report, _, noise, harness = self.run_native(entries, config=rolling.RollingConfig(True, 0.45))
        raw, _ = runtime._prepare_initial_latent(entries, windows.CPUFusion([e["window"] for e in entries]))
        sigma = float(self.schedule[0])
        beta = 0.45  # flat curve: strength is constant across sigma
        global_state = [sigma * eps + (1 - sigma) * raw_value * 2.0
                        for eps, raw_value in zip(noise.generated.unbind(), raw.unbind())]
        for index in (1, 2):
            donor = windows.unpack(harness.calls[index - 1]["output"], entries[index - 1]["window"].shapes)
            actual = windows.unpack(harness.calls[index]["input"], entries[index]["window"].shapes)
            original = windows.crop(global_state, entries[index]["window"])
            eps = windows.crop(noise.generated.unbind(), entries[index]["window"])
            for got, old, clean, epsilon in ((actual[0][:, :, :7], original[0][:, :, :7], donor[0][:, :, -7:], eps[0][:, :, :7]),
                                             (actual[1][..., :37], original[1][..., :37], donor[1][..., -37:], eps[1][..., :37])):
                torch.testing.assert_close(got, (1 - beta) * old + beta * ((1 - sigma) * clean + sigma * epsilon))
        self.assertAlmostEqual(report["continuity"]["steps"][0]["effective_strength"], beta)
        self.assertEqual(report["continuity"]["steps"][0]["video_latent_positions"], 14)

    def test_single_window_matches_stock_h3_av_normalization_and_audio_scale_once(self):
        entry = base_tests.entries(1)
        shape = entry[0]["window"].shapes
        raw = NestedTensor([torch.full(shape[0], 0.25), torch.full(shape[1], 0.75)])
        noise = NestedTensor([torch.full(s, 0.5) for s in shape])
        stock_model, joint_model = SourcePatcher(), SourcePatcher()
        stock = comfy.samplers.CFGGuider(stock_model)
        stock.set_conds(entry[0]["positive"], entry[0]["negative"])
        stock.set_cfg(1.0)
        joint = runtime.CPUWindowGuider(joint_model, entry, 1.0)
        with InitialHarness(self) as harness:
            expected = stock.sample(noise, raw, comfy.samplers.sampler_object("euler"), self.schedule, disable_pbar=True)
            actual = joint.sample(noise, raw, comfy.samplers.sampler_object("euler"), self.schedule, disable_pbar=True)
        for left, right in zip(actual.unbind(), expected.unbind()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(len(joint_model.model.input_shapes), 1)
        initial_video, initial_audio = windows.unpack(harness.calls[2]["input"], shape)
        torch.testing.assert_close(initial_video, torch.full_like(initial_video, 0.5))
        # 0.9*noise + 0.1*(raw_audio * format_scale=2 * audio_scale=4).
        torch.testing.assert_close(initial_audio, self.schedule[0] * noise.unbind()[1] +
                                   (1 - self.schedule[0]) * raw.unbind()[1] * 8.0)

    def test_invalid_or_mixed_source_initials_fail_before_noise_generation(self):
        bad_values = [None, torch.zeros(1), torch.zeros((1, 24, 37, 2, 2), dtype=torch.float64),
                      torch.full((1, 24, 37, 2, 2), float("nan")), torch.full((1, 24, 37, 2, 2), float("inf"))]
        for value in bad_values:
            with self.subTest(value=type(value).__name__):
                entries = source_entries(1)
                entries[0]["initial_video_latent"] = value
                noise = continuity_tests.CountingNoise()
                with self.assertRaisesRegex(ValueError, "initial_video_latent"):
                    runtime.sample_windows_native(noise, SourcePatcher(), entries,
                        comfy.samplers.sampler_object("euler"), self.schedule, 1.0, 0.5, 1.0)
                self.assertEqual(noise.calls, 0)
        entries = source_entries()
        entries[1].pop("initial_video_latent")
        with self.assertRaisesRegex(ValueError, "same source-video"):
            runtime._prepare_initial_latent(entries, windows.CPUFusion([e["window"] for e in entries]))

    def test_external_sigmas_accept_positive_partial_start_but_require_zero_end(self):
        original = torch.tensor([0.9, 0.6, 0.0], dtype=torch.float64)
        saved = original.clone()
        converted = validate_sigmas(original)
        self.assertEqual(converted.dtype, torch.float32)
        self.assertTrue(torch.equal(original, saved))
        self.assertNotEqual(original.data_ptr(), converted.data_ptr())
        for values in ([0.0, 0.0], [1.1, 0.0], [0.9, 0.9, 0.0], [0.9, 0.1],
                       [0.9, -0.1, 0.0], [float("nan"), 0.0], [1e-50, 0.0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_sigmas(torch.tensor(values, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
