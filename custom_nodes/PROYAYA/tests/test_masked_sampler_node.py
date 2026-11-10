"""PROYYMaskedSampler 的接口契约。

三条硬约定：
1. 遮罩是可选输入 —— 没有 mask_enabled 这类开关，接了就是重绘模式，没接就是原生窗口采样；
2. 极性固定白色=重绘，音频恒 generate all；
3. 遮罩时间轴一律等比拉伸到潜帧数。

只跑纯函数与 schema，不加载模型。
"""

import importlib
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime as base_tests

torch = base_tests.torch
masked = importlib.import_module(base_tests.package_name + ".masked_sampler")
SamplerNode = masked.PROYYMaskedSampler

# 按出现顺序排列的控件；unload 必须在最末。
EXISTING_WIDGETS = ["high_cfg", "cfg_split", "low_cfg", "fusion_preserve",
                    "rolling_enabled", "rolling_strength", "rolling_mode",
                    "rolling_front_high", "rolling_front_low", "unload"]


class SchemaTests(unittest.TestCase):
    def test_mask_is_optional_and_path_switches_are_gone(self):
        schema = SamplerNode.INPUT_TYPES()
        self.assertIn("mask", schema["optional"])
        self.assertNotIn("mask", schema["required"])
        for gone in ("mask_enabled", "mask_mode", "audio_mode", "temporal_resample"):
            self.assertNotIn(gone, schema["required"])
            self.assertNotIn(gone, schema["optional"])

    def test_widgets_keep_declared_order_and_unload_is_last(self):
        schema = SamplerNode.INPUT_TYPES()
        order = [name for group in ("required", "optional") for name in schema[group]]
        positions = [order.index(name) for name in EXISTING_WIDGETS]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(order[-1], "unload")

    def test_sample_treats_a_missing_mask_as_no_mask(self):
        signature = inspect.signature(SamplerNode.sample)
        self.assertIsNone(signature.parameters["mask"].default)
        self.assertTrue(signature.parameters["unload"].default)


class LazyTests(unittest.TestCase):
    def test_lazy_check_never_requests_the_mask(self):
        node = SamplerNode()
        # 没连遮罩时绝不能请求它：ComfyUI 会对无连线输入抛 NodeInputError。
        self.assertEqual(set(node.check_lazy_status(noise=None, model=None,
                                                    sampler=None, sigmas=None)),
                         {"noise", "model", "sampler", "sigmas"})
        self.assertEqual(node.check_lazy_status(noise=1, model=1, sampler=1, sigmas=1, mask=None), [])
        self.assertEqual(node.check_lazy_status(noise=1, model=1, sampler=1, sigmas=1,
                                                mask=torch.ones(1, 4, 4)), [])


class MaskMappingTests(unittest.TestCase):
    def test_static_mask_replicates_across_latent_frames(self):
        # 曾经的 stretch 分支把 5D 张量直接喂给 adaptive_max_pool2d，静态遮罩必崩。
        result = masked._video_mask_to_latent(torch.ones(1, 8, 8), 5, 2, 2)
        self.assertEqual(tuple(result.shape), (1, 5, 2, 2))
        self.assertTrue(bool((result == 1.0).all()))

    def test_per_frame_mask_keeps_its_temporal_structure(self):
        mask = torch.zeros(4, 4, 4)
        mask[0] = 1.0
        result = masked._video_mask_to_latent(mask, 4, 2, 2)
        self.assertEqual(tuple(result.shape), (1, 4, 2, 2))
        self.assertEqual(float(result[0, 0].min()), 1.0)
        self.assertEqual(float(result[0, 1].max()), 0.0)

    def test_audio_stream_is_always_full_regenerate(self):
        audio = masked._audio_mask_to_latent(37)
        self.assertEqual(tuple(audio.shape), (1, 37))
        self.assertTrue(bool((audio == 1.0).all()))


class DenoiseMaskTests(unittest.TestCase):
    shapes = [(1, 24, 5, 2, 2), (1, 32, 2, 37)]

    def test_nested_mask_covers_video_and_regenerates_all_audio(self):
        nested, report = masked.build_denoise_mask(torch.ones(1, 8, 8), self.shapes)
        video, audio = nested.unbind()
        self.assertEqual(tuple(video.shape), (1, 5, 2, 2))
        self.assertEqual(tuple(audio.shape), (1, 37))
        self.assertTrue(bool((audio == 1.0).all()))
        self.assertEqual(report["video_regenerate_ratio"], 1.0)
        self.assertEqual(report["audio_regenerate_ratio"], 1.0)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["temporal_map"], "proportional stretch")
        self.assertNotIn("temporal_resample", report)

    def test_black_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "全黑"):
            masked.build_denoise_mask(torch.zeros(1, 8, 8), self.shapes)

    def test_mask_polarity_is_never_inverted(self):
        # 左半边白：positive 区域必须留在左边，不允许出现 white=preserve 的反相路径。
        mask = torch.zeros(1, 8, 8)
        mask[:, :, :4] = 1.0
        nested, _ = masked.build_denoise_mask(mask, self.shapes)
        video, _ = nested.unbind()
        self.assertEqual(float(video[..., 0].max()), 1.0)
        self.assertEqual(float(video[..., -1].max()), 0.0)


class UnmaskedPathTests(unittest.TestCase):
    """遮罩缺失时必须与 PROYAYA 原生窗口采样逐位等价。"""

    sigmas = torch.tensor([1.0, 0.5, 0.0])

    def run_entries(self, sampler_call):
        cont = importlib.import_module("test_continuity_runtime")
        with cont.RealCFGHarness(self), \
                mock.patch.object(masked._RUNTIME, "_native_model_info", return_value=1):
            return sampler_call(cont)

    def test_no_mask_is_bit_exact_against_the_plain_window_sampler(self):
        def native(cont):
            return masked._RUNTIME.sample_windows_native(
                cont.CountingNoise(), cont.CloneablePatcher(), cont.base_tests.entries(3),
                base_tests.comfy.samplers.sampler_object("euler"), self.sigmas, 3.0, 0.5, 1.2)

        def without_mask(cont):
            return masked.sample_masked_windows(
                cont.CountingNoise(), cont.CloneablePatcher(), cont.base_tests.entries(3),
                base_tests.comfy.samplers.sampler_object("euler"), self.sigmas, None, 3.0, 0.5, 1.2)

        plain_latent, plain_report = self.run_entries(native)
        masked_latent, report = self.run_entries(without_mask)
        self.assertFalse(report["mask"]["enabled"])
        self.assertIn("未连接遮罩", report["mask"]["note"])
        self.assertEqual(report["window_calls"], plain_report["window_calls"])
        for actual, expected in zip(masked_latent["samples"].unbind(), plain_latent["samples"].unbind()):
            self.assertTrue(torch.equal(actual, expected))

    def test_missing_source_latent_only_matters_when_a_mask_is_connected(self):
        cont = importlib.import_module("test_continuity_runtime")
        entries = [{key: value for key, value in entry.items() if key != "initial_video_latent"}
                   for entry in cont.base_tests.entries(3)]
        with mock.patch.object(masked._RUNTIME, "_native_model_info", return_value=1):
            with self.assertRaisesRegex(ValueError, "需要源片初始 latent"):
                masked.sample_masked_windows(
                    cont.CountingNoise(), None, entries, None, self.sigmas,
                    torch.ones(1, 4, 4), 3.0, 0.5, 1.2)


if __name__ == "__main__":
    unittest.main()
