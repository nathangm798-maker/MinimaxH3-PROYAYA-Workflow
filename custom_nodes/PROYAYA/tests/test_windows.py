"""CPU contracts for H3 window timing, shared state, and overlap fusion."""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "windows.py"
SPEC = importlib.util.spec_from_file_location("h3_window_sampler_windows_test", MODULE_PATH)
windows_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = windows_module
SPEC.loader.exec_module(windows_module)

Window = windows_module.Window
CPUFusion = windows_module.CPUFusion
pack = windows_module.pack
unpack = windows_module.unpack
crop = windows_module.crop
shapes_for_windows = windows_module.shapes_for_windows


def make_windows(frames=124, overlap=22, height=2, width=3):
    stride = frames - overlap
    return [Window(index * stride, frames, height, width) for index in range(3)]


def noise_for(fusion):
    generator = torch.Generator(device="cpu").manual_seed(719)
    return pack([
        torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        for shape in fusion.shapes
    ])


class NativeGridTests(unittest.TestCase):
    def test_exact_native_av_grids(self):
        cases = [
            # frames, overlap, video starts, global T, audio starts, local A, global A
            (124, 22, [0, 30, 60], 97, [0, 170, 340], 207, 547),
            (158, 5, [0, 45, 90], 137, [0, 255, 510], 263, 773),
            (158, 56, [0, 30, 60], 107, [0, 170, 340], 263, 603),
        ]
        for frames, overlap, video_starts, video_t, audio_starts, audio_t, global_audio_t in cases:
            with self.subTest(frames=frames, overlap=overlap):
                windows = make_windows(frames, overlap)
                self.assertEqual([window.video_start for window in windows], video_starts)
                self.assertEqual([window.audio_start for window in windows], audio_starts)
                self.assertTrue(all(window.video_start % 5 == 0 for window in windows))
                self.assertTrue(all(window.audio_length == audio_t for window in windows))
                self.assertEqual(
                    shapes_for_windows(windows),
                    [(1, 24, video_t, 2, 3), (1, 32, 2, global_audio_t)],
                )
                self.assertEqual(video_t % 5, 2)
                fusion = CPUFusion(windows)
                self.assertTrue(all(bool((count > 0).all()) for count in fusion.counts))

    def test_158_with_22_overlap_rejects_fractional_audio_origin(self):
        with self.assertRaisesRegex(ValueError, "multiple of 51"):
            Window(136, 158, 2, 3)

    def test_rejects_non_native_lengths(self):
        for frames in (0, 4, 123, 168):
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):
                    Window(0, frames, 2, 3)

    def test_long_windows_keep_native_grid_without_quality_cap(self):
        window = Window(0, 175, 2, 3)
        self.assertEqual(window.video_length, 52)
        self.assertEqual(window.audio_length, 292)

    def test_rejects_missing_head_gap_and_resolution_change(self):
        invalid = [
            [],
            [Window(51, 124, 2, 3)],
            [Window(0, 124, 2, 3), Window(153, 124, 2, 3)],
            [Window(0, 124, 2, 3), Window(102, 124, 3, 3)],
        ]
        for windows in invalid:
            with self.subTest(windows=windows):
                with self.assertRaises(ValueError):
                    shapes_for_windows(windows)

    def test_pack_unpack_and_crop_preserve_stream_coordinates(self):
        windows = make_windows()
        shapes = shapes_for_windows(windows)
        video = torch.arange(24 * 97 * 2 * 3, dtype=torch.float32).view(shapes[0])
        audio = torch.arange(32 * 2 * 547, dtype=torch.float32).view(shapes[1]) + 100000
        streams = unpack(pack([video, audio]), shapes)
        self.assertTrue(torch.equal(streams[0], video))
        self.assertTrue(torch.equal(streams[1], audio))
        local_video, local_audio = crop(streams, windows[1])
        self.assertTrue(torch.equal(local_video, video[:, :, 30:67]))
        self.assertTrue(torch.equal(local_audio, audio[..., 170:377]))
        self.assertEqual([tuple(value.shape) for value in (local_video, local_audio)], windows[1].shapes)


class SharedStateTests(unittest.TestCase):
    def test_shared_noise_overlap_is_identical_and_identity_fuses_back(self):
        windows = make_windows()
        fusion = CPUFusion(windows)
        state = noise_for(fusion)
        seen = {}

        def identity(index, local):
            self.assertEqual(local.device.type, "cpu")
            self.assertEqual(local.dtype, torch.float32)
            seen[index] = [value.clone() for value in unpack(local, windows[index].shapes)]
            return local

        result = fusion.predict(state, identity)
        torch.testing.assert_close(result, state, rtol=1e-6, atol=1e-6)
        for index in (0, 1):
            self.assertTrue(torch.equal(seen[index][0][:, :, 30:37], seen[index + 1][0][:, :, :7]))
            self.assertTrue(torch.equal(seen[index][1][..., 170:207], seen[index + 1][1][..., :37]))

    def test_order_independence_with_three_way_overlap(self):
        windows = [Window(start, 158, 2, 3) for start in (0, 51, 102)]
        fusion = CPUFusion(windows)
        state = noise_for(fusion)

        def predictor(index, local):
            return local * (index + 1) + index

        forward = fusion.predict(state, predictor)
        reverse = fusion.predict(state, predictor, order=[2, 1, 0])
        alternate = fusion.predict(state, predictor, order=[1, 0, 2])
        torch.testing.assert_close(forward, reverse, rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(forward, alternate, rtol=1e-6, atol=2e-6)

    def test_each_window_routes_its_own_prediction(self):
        fusion = CPUFusion(make_windows())
        state = noise_for(fusion)
        calls = []

        def predictor(index, local):
            calls.append(index)
            return torch.full_like(local, float(index + 1))

        video, audio = unpack(fusion.predict(state, predictor), fusion.shapes)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(float(video[0, 0, 0, 0, 0]), 1.0)
        self.assertEqual(float(video[0, 0, 40, 0, 0]), 2.0)
        self.assertEqual(float(video[0, 0, -1, 0, 0]), 3.0)
        self.assertAlmostEqual(float(video[0, 0, 30, 0, 0]), 1.125, places=6)
        self.assertAlmostEqual(float(video[0, 0, 36, 0, 0]), 1.875, places=6)
        self.assertAlmostEqual(float(audio[0, 0, 0, 170]), 1.0 + 1.0 / 38.0, places=6)
        self.assertEqual(float(audio[0, 0, 0, -1]), 3.0)

    def test_predictor_cannot_mutate_global_snapshot_or_later_inputs(self):
        windows = make_windows()
        fusion = CPUFusion(windows)
        state = noise_for(fusion)
        original = state.clone()
        original_streams = unpack(original, fusion.shapes)
        seen = []

        def destructive_predictor(index, local):
            self.assertTrue(torch.equal(local, pack(crop(original_streams, windows[index]))))
            seen.append(index)
            local.fill_(7.0)
            return local

        result = fusion.predict(state, destructive_predictor)
        self.assertEqual(seen, [0, 1, 2])
        self.assertTrue(torch.equal(state, original))
        torch.testing.assert_close(result, torch.full_like(result, 7.0), rtol=1e-6, atol=1e-6)

    def test_counts_and_weights_are_positive_cpu_temporal_buffers(self):
        small = CPUFusion(make_windows(height=2, width=3))
        large = CPUFusion(make_windows(height=48, width=84))
        self.assertEqual([tuple(count.shape) for count in large.counts], [(1, 1, 97, 1, 1), (1, 1, 1, 547)])
        self.assertEqual(sum(count.numel() for count in large.counts), 97 + 547)
        self.assertEqual(sum(count.numel() for count in large.counts), sum(count.numel() for count in small.counts))
        for tensor in large.counts + [weight for pair in large.weights for weight in pair]:
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(bool((tensor > 0).all()))
        for video_weight, audio_weight in large.weights:
            self.assertEqual(video_weight.numel(), 37)
            self.assertEqual(audio_weight.numel(), 207)


class PredictionValidationTests(unittest.TestCase):
    def setUp(self):
        self.fusion = CPUFusion(make_windows())
        self.state = noise_for(self.fusion)

    def test_nonfinite_predictions_fail(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                def predictor(index, local):
                    local[..., -1] = invalid
                    return local

                with self.assertRaisesRegex(ValueError, "non-finite"):
                    self.fusion.predict(self.state, predictor)

    def test_wrong_prediction_shape_fails(self):
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            self.fusion.predict(self.state, lambda index, local: local[..., :-1])

    def test_wrong_prediction_dtype_fails(self):
        with self.assertRaisesRegex(ValueError, "FP32 CPU"):
            self.fusion.predict(self.state, lambda index, local: local.double())

    def test_non_fp32_state_fails_before_predictor(self):
        def unexpected_predictor(index, local):
            self.fail("An invalid global state reached the predictor")

        with self.assertRaisesRegex(ValueError, "FP32 on CPU"):
            self.fusion.predict(self.state.double(), unexpected_predictor)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable; all numerical tests use CPU")
    def test_cuda_state_is_rejected_before_predictor(self):
        def unexpected_predictor(index, local):
            self.fail("A CUDA global state reached the predictor")

        with self.assertRaisesRegex(ValueError, "FP32 on CPU"):
            self.fusion.predict(self.state.to("cuda"), unexpected_predictor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
