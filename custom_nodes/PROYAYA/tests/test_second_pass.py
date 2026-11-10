"""二采（second pass）契约：时间轴复刻 + 初始 latent 直通 + 音频锁定。

三条硬约定：
1. 二采的分段完全照抄一采：起点/长度/补帧/重叠逐帧重合，不重新 solve；
2. 初始 latent 从 source_latent 裁出来，视频与音频都带，不经过 VAE 往返；
3. 音频在二采里被锁死（audio mask = 0），画面照常按 sigmas 重绘。

只跑纯函数，不加载模型。
"""

import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime as base_tests

torch = base_tests.torch
NestedTensor = base_tests.NestedTensor
encode_nodes = importlib.import_module(base_tests.package_name + ".encode_nodes")
masked = importlib.import_module(base_tests.package_name + ".masked_sampler")
planner = importlib.import_module(base_tests.package_name + ".planner")
runtime = importlib.import_module(base_tests.package_name + ".runtime")
windows_mod = importlib.import_module(base_tests.package_name + ".windows")

LATENT_H, LATENT_W = 45, 80


def _entries(starts=(0, 51, 102), frames=124):
    return [{"window": windows_mod.Window(start, frames, LATENT_H, LATENT_W)} for start in starts]


def _source_latent(entries, video_frames=None, audio_steps=None):
    last = entries[-1]["window"]
    covered = last.video_start + last.video_length
    audio_total = last.audio_start + last.audio_length
    video = torch.zeros((1, 24, video_frames or covered, LATENT_H, LATENT_W), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, audio_steps or audio_total), dtype=torch.float32)
    return {"samples": NestedTensor([video, audio])}


class SecondPassPlanTests(unittest.TestCase):
    def test_plan_is_copied_from_the_first_pass(self):
        entries = _entries()
        plan, streams = encode_nodes._second_pass_plan(
            entries, _source_latent(entries), 226, 40)
        self.assertEqual([w.start for w in plan.windows], [0, 51, 102])
        self.assertEqual([w.frames for w in plan.windows], [124, 124, 124])
        self.assertEqual([w.overlap for w in plan.windows], [0, 73, 73])
        self.assertEqual(plan.pad_frames, 0)
        self.assertEqual(plan.output_frames, 226)
        self.assertEqual(plan.mode, "reuse")
        self.assertEqual(tuple(streams[0].shape)[2], 67)
        self.assertEqual(tuple(streams[1].shape)[-1], 377)

    def test_short_length_only_pads_the_last_window(self):
        entries = _entries()
        plan, _ = encode_nodes._second_pass_plan(entries, _source_latent(entries), 200, 40)
        last = plan.windows[-1]
        self.assertEqual((last.real_frames, last.pad_frames), (98, 26))
        self.assertEqual(plan.output_frames, 200)

    def test_timeline_mismatch_is_rejected(self):
        entries = _entries()
        with self.assertRaisesRegex(ValueError, "时间轴对不上"):
            encode_nodes._second_pass_plan(entries, _source_latent(entries, video_frames=66), 226, 40)

    def test_missing_source_latent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "需要 source_latent"):
            encode_nodes._second_pass_plan(_entries(), None, 226, 40)

    def test_non_joint_latent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "joint AV latent"):
            encode_nodes._second_pass_plan(_entries(), {"samples": torch.zeros(1, 24, 67, 45, 80)}, 226, 40)


class InitialLatentTests(unittest.TestCase):
    def _fuse(self, entries, audio_init=True):
        fusion = windows_mod.CPUFusion([entry["window"] for entry in entries])
        for index, entry in enumerate(entries):
            shape_v, shape_a = entry["window"].shapes
            entry["initial_video_latent"] = torch.full(shape_v, 0.5, dtype=torch.float32)
            if audio_init:
                entry["initial_audio_latent"] = torch.full(shape_a, 0.25, dtype=torch.float32)
        latent, mode = runtime._prepare_initial_latent(entries, fusion)
        return latent.unbind(), mode

    def test_audio_initial_latent_is_fused(self):
        streams, mode = self._fuse(_entries())
        self.assertEqual(mode, "source_av")
        self.assertTrue(torch.allclose(streams[1], torch.full_like(streams[1], 0.25)),
                        "重叠区的加权平均必须还原成同一个常量")
        self.assertTrue(torch.allclose(streams[0], torch.full_like(streams[0], 0.5)))

    def test_audio_stays_empty_without_it(self):
        streams, mode = self._fuse(_entries(), audio_init=False)
        self.assertEqual(mode, "source_video")
        self.assertEqual(float(streams[1].abs().max()), 0.0)

    def test_partial_audio_init_is_rejected(self):
        entries = _entries()
        fusion = windows_mod.CPUFusion([entry["window"] for entry in entries])
        for entry in entries:
            entry["initial_video_latent"] = torch.zeros(entry["window"].shapes[0])
        entries[0]["initial_audio_latent"] = torch.zeros(entries[0]["window"].shapes[1])
        with self.assertRaisesRegex(ValueError, "same initial audio latent mode"):
            runtime._prepare_initial_latent(entries, fusion)


class AudioLockTests(unittest.TestCase):
    SHAPES = [(1, 24, 37, 4, 4), (1, 32, 2, 207)]

    def test_lock_mask_keeps_video_and_freezes_audio(self):
        nested = masked._lock_audio_mask(self.SHAPES)
        video, audio = nested.unbind()
        self.assertEqual(tuple(video.shape), (37, 4, 4))
        self.assertEqual(float(video.min()), 1.0)
        self.assertEqual(tuple(audio.shape), (1, 207))
        self.assertEqual(float(audio.max()), 0.0)

    def test_build_denoise_mask_reports_audio_lock(self):
        mask = torch.ones(1, 32, 32, dtype=torch.float32)
        nested, report = masked.build_denoise_mask(mask, self.SHAPES, lock_audio=True)
        self.assertTrue(report["audio_lock"])
        self.assertEqual(report["audio_regenerate_ratio"], 0.0)
        self.assertEqual(report["video_regenerate_ratio"], 1.0)
        _, audio = nested.unbind()
        self.assertEqual(float(audio.max()), 0.0)

    def test_first_pass_audio_is_still_fully_redrawn(self):
        mask = torch.ones(1, 32, 32, dtype=torch.float32)
        _, report = masked.build_denoise_mask(mask, self.SHAPES)
        self.assertFalse(report["audio_lock"])
        self.assertEqual(report["audio_regenerate_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
