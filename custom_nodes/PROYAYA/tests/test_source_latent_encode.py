"""Source-latent encoding contracts with real VIDEO reading and ImageScale.

VAE/model inference is replaced with deterministic CPU probes. The tests keep
the actual 24-fps reader, time trims, resize/crop, native grid and node assembly.
"""

from fractions import Fraction
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime as base_tests

torch = base_tests.torch
NestedTensor = base_tests.NestedTensor
windows = base_tests.windows
video_module = importlib.import_module(base_tests.package_name + ".native_video")
encode_nodes = importlib.import_module(base_tests.package_name + ".encode_nodes")
from comfy_api.latest import InputImpl, Types, io
import nodes as comfy_nodes


def source_video(count=328, rate=24):
    # Spatial details distinguish an original-frame crop from stretching or
    # enlarging a previously reduced 32x32 reference image.
    pixels = torch.zeros(count, 64, 96, 3)
    pixels[..., 0] = torch.arange(96).remainder(3).float()[None, None, :] / 2
    pixels[..., 1] = torch.arange(64).float()[None, :, None] / 63
    pixels[..., 2] = torch.arange(count).float()[:, None, None] / max(count, 1)
    components = Types.VideoComponents(images=pixels, frame_rate=Fraction(rate))
    return InputImpl.VideoFromComponents(components), pixels


class ProbeVAE:
    patcher = "video-vae-patcher"

    def __init__(self, failure=None):
        self.failure = failure
        self.frames = []
        self.frame_refs = []
        self.returned = []

    def encode(self, frames):
        self.frames.append(frames.clone())
        self.frame_refs.append(weakref.ref(frames))
        if self.failure == "raise":
            raise RuntimeError("injected target VAE failure")
        shape = windows.Window(0, len(frames), frames.shape[1] // 16, frames.shape[2] // 16).shapes[0]
        value = torch.full(shape, len(self.frames) + 0.25, dtype=torch.float16)
        if self.failure == "shape":
            value = value[:, :, :-1]
        elif self.failure == "nonfinite":
            value[..., 0, 0, 0] = float("nan")
        self.returned.append(value)
        return value


class SourceLatentEncodeTests(unittest.TestCase):
    def setUp(self):
        self.node = encode_nodes.PROYYMiniMaxH3ReferenceToVideo
        self.clip = types.SimpleNamespace(
            patcher="clip-patcher", tokenize=lambda text: text,
            encode_from_tokens_scheduled=lambda tokens: [[torch.ones(1, 2, 3), {}]],
        )
        self.vae = ProbeVAE()
        self.reference_calls = []
        self.reference_latents = []
        self.progress_patch = mock.patch.object(self.node, "_progress_text")
        self.progress_patch.start()
        self.addCleanup(self.progress_patch.stop)
        self.encoder_patch = mock.patch.object(encode_nodes, "encode_weighted_references", side_effect=self.reference_encode)
        self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        self.unload_patch = mock.patch.object(encode_nodes.model_management, "unload_model_and_clones")
        self.unload = self.unload_patch.start()
        self.addCleanup(self.unload_patch.stop)

    def reference_encode(self, **kwargs):
        recorded = {**kwargs, "ref_videos": dict(kwargs["ref_videos"])}
        self.reference_calls.append(recorded)
        window = windows.Window(0, kwargs["length"], kwargs["height"] // 16, kwargs["width"] // 16)
        empty = {"samples": NestedTensor([torch.zeros(shape) for shape in window.shapes])}
        self.reference_latents.append(empty)
        refs = []
        if kwargs["video_weight"] and kwargs["ref_videos"]:
            refs.append({"kind": "video", "latent": torch.zeros(window.shapes[0])})
        return io.NodeOutput([[torch.ones(1, 2, 3), {"minimax_refs": refs}]], empty)

    def args(self, video, **overrides):
        result = dict(
            auto_segment=True, auto_prompt=False, window_frames=124, manual_lengths="124,124,124",
            overlap=22, keep_final=40, width=64, height=32, length=328,
            text_weight=1.0, image_weight=1.0, video_weight=1.0,
            unload=True, preview_only=False, clip=self.clip, vae=self.vae,
            ref_images={"ref_image_0": torch.zeros(1, 32, 32, 3)},
            main_ref_videos={"main_ref_video_0": video},
            prompts={f"prompt_{i}": f"window {i}" for i in range(3)},
            video_encode=True,
        )
        result.update(overrides)
        return result

    def test_enabled_reads_original_pixels_and_owns_raw_cpu_latents(self):
        source, pixels = source_video()
        low_reference = video_module.WindowResizedVideo(source, 32, 32, "area", "center")
        output = self.node.execute(**self.args(low_reference))
        entries, frame_count, report = output.result
        self.assertEqual(frame_count, 328)
        self.assertEqual([entry["window"].start for entry in entries], [0, 102, 204])
        self.assertEqual(len(self.vae.frames), 3)
        for index, start in enumerate((0, 102, 204)):
            expected_pixels = comfy_nodes.ImageScale().upscale(
                pixels[start:start + 124], "lanczos", 64, 32, "center")[0]
            torch.testing.assert_close(self.vae.frames[index], expected_pixels, rtol=0, atol=0)
            value = entries[index]["initial_video_latent"]
            self.assertEqual(tuple(value.shape), (1, 24, 37, 2, 4))
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
            self.assertNotEqual(value.untyped_storage().data_ptr(), self.vae.returned[index].untyped_storage().data_ptr())
            self.assertTrue(bool((value == index + 1.25).all()))
            # The reference path still receives its separately reduced window.
            self.assertEqual(tuple(self.reference_calls[index]["ref_videos"]["ref_video_0"].shape), (124, 32, 32, 3))
            self.assertEqual(self.reference_calls[index]["prompt"], f"window {index}")
            self.assertTrue(all(torch.count_nonzero(part).item() == 0
                                for part in self.reference_latents[index]["samples"].unbind()))
        self.vae.returned[0].fill_(99)
        self.assertTrue(bool((entries[0]["initial_video_latent"] == 1.25).all()))
        self.assertTrue(all(ref() is None for ref in self.vae.frame_refs))
        self.assertIn("绕过 1 层", report)
        self.assertIn("96×64 → 64×32", report)
        self.assertEqual(self.unload.call_args_list, [mock.call("clip-patcher"), mock.call("video-vae-patcher")])

    def test_video_reference_weight_zero_does_not_disable_initial_video_latent(self):
        source, _ = source_video()
        output = self.node.execute(**self.args(source, video_weight=0.0))
        self.assertEqual(len(self.vae.frames), 3)
        self.assertTrue(all("initial_video_latent" in entry for entry in output.result[0]))
        self.assertTrue(all(call["ref_videos"] == {} for call in self.reference_calls))

    def test_disabled_and_omitted_keep_original_entries_and_never_target_encode(self):
        source, _ = source_video()
        for omit in (False, True):
            with self.subTest(omitted=omit):
                kwargs = self.args(source, video_encode=False)
                if omit:
                    del kwargs["video_encode"]
                with mock.patch.object(encode_nodes, "initial_video_target", side_effect=AssertionError("disabled target path")):
                    output = self.node.execute(**kwargs)
                self.assertTrue(all(set(entry) == {"window", "positive", "negative"} for entry in output.result[0]))
                self.assertNotIn("主视频初始latent", output.result[2])
        self.assertEqual(self.vae.frames, [])

    def test_short_source_plans_actual_frames_and_pads_only_inside_window_grid(self):
        source, pixels = source_video(count=110)
        output = self.node.execute(**self.args(
            source, video_weight=0.0, auto_prompt=True))
        entries = output.result[0]
        self.assertEqual(len(entries), 1)  # length=328 > 110 never invents windows
        self.assertEqual(output.result[1], 110)
        self.assertEqual([len(frames) for frames in self.vae.frames], [124])
        last = comfy_nodes.ImageScale().upscale(pixels[-1:], "lanczos", 64, 32, "center")[0]
        torch.testing.assert_close(self.vae.frames[0][110:], last.expand(14, -1, -1, -1))
        self.assertEqual(tuple(entries[0]["initial_video_latent"].shape), (1, 24, 37, 2, 4))
        report = output.result[2]
        self.assertIn("⚠⚠⚠", report)
        self.assertIn("218", report)  # shortfall 328-110

    def test_target_longer_than_source_260_gives_two_windows_and_warning(self):
        source, _ = source_video(count=260)
        output = self.node.execute(**self.args(
            source, length=1000, auto_prompt=True, video_weight=0.0))
        entries, frames, report = output.result
        self.assertEqual(len(entries), 2)
        self.assertEqual(frames, 226)
        self.assertEqual(len(self.vae.frames), 2)
        self.assertTrue(report.startswith("!" * 60))
        self.assertIn("740", report)

    def test_no_warning_when_length_fits_source(self):
        source, _ = source_video(count=328)
        output = self.node.execute(**self.args(source, video_weight=0.0))
        self.assertNotIn("⚠", output.result[2])
        self.assertEqual(output.result[1], 328)

    def test_preview_only_with_shortfall_warns_and_blocks_generation(self):
        source, _ = source_video(count=260)
        output = self.node.execute(**self.args(
            source, length=1000, auto_prompt=True, preview_only=True, clip=None, vae=None))
        self.assertEqual(type(output.result[0]).__name__, "ExecutionBlocker")
        self.assertIn("⚠⚠⚠", output.result[2])
        self.assertEqual(len(output.result), 3)  # windows / length / plan_report，不再有 images

    def test_short_target_tail_pads_target_boundary_not_later_source_content(self):
        source, pixels = source_video(count=328)
        output = self.node.execute(**self.args(
            source, length=141, auto_prompt=True, video_weight=0.0,
            keep_final=17))
        entries = output.result[0]
        self.assertEqual(output.result[1], 141)
        self.assertEqual([(entry["window"].start, entry["window"].frames) for entry in entries],
                         [(0, 124), (102, 90)])
        self.assertEqual([len(frames) for frames in self.vae.frames], [124, 90])
        valid = comfy_nodes.ImageScale().upscale(pixels[102:141], "lanczos", 64, 32, "center")[0]
        torch.testing.assert_close(self.vae.frames[1][:39], valid, rtol=0, atol=0)
        torch.testing.assert_close(self.vae.frames[1][39:], valid[-1:].expand(51, -1, -1, -1), rtol=0, atol=0)
        # The still-available future source must not enter the padded context.
        future = comfy_nodes.ImageScale().upscale(pixels[141:192], "lanczos", 64, 32, "center")[0]
        self.assertFalse(torch.equal(self.vae.frames[1][39:], future))

    def test_preview_does_not_request_or_use_clip_vae_even_with_switch_enabled(self):
        source, _ = source_video()
        kwargs = self.args(source, preview_only=True, clip=None, vae=None)
        lazy = dict(kwargs)
        lazy["main_ref_videos"] = {"main_ref_video_0": (source, "main_ref_videos.main_ref_video_0")}
        lazy["prompts"] = {f"prompt_{i}": (None, f"prompts.prompt_{i}") for i in range(3)}
        self.assertEqual(self.node.check_lazy_status(**lazy), [])
        output = self.node.execute(**kwargs)
        self.assertEqual(type(output.result[0]).__name__, "ExecutionBlocker")
        self.assertEqual(len(output.result), 3)  # 只有 windows / length / plan_report
        self.assertEqual(self.reference_calls, [])
        self.assertEqual(self.vae.frames, [])
        self.unload.assert_not_called()

    def test_target_vae_errors_and_invalid_shapes_always_unload_encoders(self):
        source, _ = source_video()
        for failure, expected_error in (("raise", RuntimeError), ("shape", ValueError), ("nonfinite", ValueError)):
            with self.subTest(failure=failure):
                self.unload.reset_mock()
                broken = ProbeVAE(failure)
                with self.assertRaises(expected_error):
                    self.node.execute(**self.args(source, vae=broken, video_weight=0.0))
                self.assertEqual(self.unload.call_args_list,
                                 [mock.call("clip-patcher"), mock.call("video-vae-patcher")])

    def test_schema_has_disabled_by_default_switch_and_keep_final(self):
        schema = self.node.define_schema()
        switch = next(item for item in schema.inputs if item.id == "video_encode")
        self.assertEqual(schema.node_id, "PROYYMiniMaxH3ReferenceToVideo")
        self.assertFalse(switch.default)
        # 中文显示名已去掉，界面直接显示接口名 video_encode。
        self.assertIsNone(switch.display_name)
        self.assertTrue(any(item.id == "keep_final" for item in schema.inputs))
        self.assertEqual(len(schema.outputs), 3)

    def test_nested_resize_time_trims_survive_spatial_bypass_and_keep_24fps(self):
        source, pixels = source_video(count=200, rate=30)
        first = video_module.WindowResizedVideo(
            source, 32, 32, "area", "center", start_offset=Fraction(10, 24), duration=Fraction(100, 24))
        second = video_module.WindowResizedVideo(
            first, 64, 64, "bilinear", "disabled", start_offset=Fraction(20, 24), duration=Fraction(40, 24))
        target, report = video_module.initial_video_target(second, 64, 32)
        self.assertIs(target.video, source)
        self.assertEqual(report["bypassed_resize_recipes"], 2)
        self.assertEqual(video_module.video_info(target)["total_frames"], 40)
        self.assertEqual(target.start_offset, Fraction(30, 24))
        actual = video_module.read_video_window(target, 5, 7, 64, 32)
        # Original frame indices floor(((30+5+k)/24)*30), not 24-fps
        # indices from a materialized intermediate resize.
        indices = [((Fraction(35 + k, 24) * 30).numerator //
                    (Fraction(35 + k, 24) * 30).denominator) for k in range(7)]
        expected = comfy_nodes.ImageScale().upscale(pixels[indices], "lanczos", 64, 32, "center")[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class PlanNoticeTests(SourceLatentEncodeTests):
    """Wiring problems are reported in the plan text, not raised as exceptions."""

    def test_prompt_count_mismatch_reports_instead_of_raising(self):
        source, _ = source_video()
        prompts = {"prompt_0": "slip", "prompt_1": "hold"}
        output = self.node.execute(**self.args(source, prompts=prompts, video_weight=0.0))
        entries, frames, report = output.result
        self.assertEqual(type(entries).__name__, "ExecutionBlocker")
        self.assertEqual(frames, 328)
        # auto_segment=true 时重叠求解总是开启，2 个提示词在 328 帧下无对齐布局，
        # 因此先报「无法求解重叠」并列出可行段数，而不是笼统的「数量不匹配」。
        self.assertIn("无法为 2 个提示词求解重叠", report)
        self.assertIn("可行的段数：1、3、4、5", report)
        self.assertIn("（缺提示词）", report)   # 报告表格仍按当前参数列出每一窗
        self.assertEqual(self.vae.frames, [])
        self.assertEqual(self.reference_calls, [])

    def test_mismatch_does_not_raise_during_lazy_discovery(self):
        source, _ = source_video()
        lazy = self.args(source, prompts={"prompt_0": "only"})
        lazy["main_ref_videos"] = {"main_ref_video_0": (source, "main_ref_videos.main_ref_video_0")}
        lazy["prompts"] = {"prompt_0": (None, "prompts.prompt_0")}
        self.assertEqual(self.node.check_lazy_status(**lazy), [])

    def test_auto_segment_solves_one_cheap_identical_prompt_boundary(self):
        source, _ = source_video()
        prompts = {"prompt_0": "slip", "prompt_1": "hold", "prompt_2": "hold",
                   "prompt_3": "restore"}
        output = self.node.execute(**self.args(
            source, prompts=prompts, video_weight=0.0))
        entries, frames, report = output.result
        self.assertNotEqual(type(entries).__name__, "ExecutionBlocker")
        self.assertEqual([entry["window"].start for entry in entries], [0, 51, 153, 204])
        self.assertEqual(frames, 328)
        self.assertIn("重叠自动求解：3 条边界 → 73、22、73", report)
        self.assertIn("符合提示词偏好 3/3", report)
        self.assertIn("共采样 496 帧", report)
        self.assertNotIn("⚠", report)
        # The report table carries the per-window overlap the runtime uses.
        self.assertIn("| [51, 175) | 124 | 73 |", report)
        self.assertIn("| [153, 277) | 124 | 22 |", report)

    def test_auto_segment_keeps_every_boundary_wide_when_prompts_all_differ(self):
        source, _ = source_video()
        prompts = {f"prompt_{i}": f"shot {i}" for i in range(5)}
        output = self.node.execute(**self.args(
            source, prompts=prompts, video_weight=0.0))
        entries, _, report = output.result
        self.assertEqual([entry["window"].start for entry in entries], [0, 51, 102, 153, 204])
        self.assertIn("4 条边界 → 73、73、73、73", report)
        self.assertIn("符合提示词偏好 4/4", report)

    def test_auto_segment_reports_an_impossible_prompt_count(self):
        source, _ = source_video()
        prompts = {f"prompt_{i}": f"shot {i}" for i in range(6)}
        output = self.node.execute(**self.args(
            source, prompts=prompts, video_weight=0.0))
        entries, _, report = output.result
        self.assertEqual(type(entries).__name__, "ExecutionBlocker")
        self.assertIn("无法为 6 个提示词求解重叠", report)
        # Two windows cannot span 328 frames; six has no aligned layout either.
        self.assertIn("可行的段数：1、3、4、5", report)
        self.assertEqual(self.reference_calls, [])

    def test_auto_prompt_uses_only_the_first_prompt_for_every_window(self):
        # auto_prompt 默认 true：整段共用第一个提示词，多接的提示词不参与，
        # 也不会因为数量对不上而拦截生成。
        schema = self.node.define_schema()
        switch = next(item for item in schema.inputs if item.id == "auto_prompt")
        self.assertTrue(switch.default)
        source, _ = source_video()
        prompts = {f"prompt_{i}": f"shot {i}" for i in range(3)}
        output = self.node.execute(**self.args(
            source, prompts=prompts, auto_segment=True, auto_prompt=True, video_weight=0.0))
        entries, frames, report = output.result
        self.assertNotEqual(type(entries).__name__, "ExecutionBlocker")
        self.assertEqual(frames, 328)
        self.assertNotIn("提示词数量不匹配", report)
        self.assertNotIn("无法为", report)
        # 三窗都取自同一个提示词接口。
        self.assertIn("| prompt_0", report)
        self.assertNotIn("| prompt_1", report)

    def test_manual_mode_does_not_auto_solve(self):
        # 重叠自动求解已并入 auto_segment：手动模式不再求解，也不再抛错，
        # 提示词数量对不上时改为在报告里说明。
        source, _ = source_video()
        prompts = {"prompt_0": "slip", "prompt_1": "hold", "prompt_2": "hold",
                   "prompt_3": "restore"}
        output = self.node.execute(**self.args(
            source, prompts=prompts, auto_segment=False, video_weight=0.0))
        entries, _, report = output.result
        self.assertEqual(type(entries).__name__, "ExecutionBlocker")
        self.assertIn("提示词数量不匹配", report)
        self.assertNotIn("重叠自动求解", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
