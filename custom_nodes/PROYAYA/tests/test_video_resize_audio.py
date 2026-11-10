"""PROYYImageScale 的 AUDIO 输出契约。

三条硬约定：
1. 音频输出紧跟视频输出（视频、音频、目标视频宽、目标视频高）；
2. 音频取源视频的音频轨，且与官方 ``GetVideoComponents`` 抽出来的是**同一份波形**；
3. 源片没有音频轨时输出 ``None``，但节点本身不能报错 —— 视频输出照常可用。

测试自造两个探针片（带/不带音频），不依赖 input 目录里现成的素材。
"""

import importlib
import tempfile
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import av

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime as base_tests

torch = base_tests.torch
video_resize = importlib.import_module(base_tests.package_name + ".video_resize")
native_video = importlib.import_module(base_tests.package_name + ".native_video")
InputImpl = importlib.import_module("comfy_api.latest").InputImpl

SIZE = (360, 640)
FPS = 24
SECONDS = 1
RATE = 44100


def _write_probe(path, with_audio, fps=FPS, seconds=SECONDS):
    container = av.open(path, mode="w")
    vstream = container.add_stream("libx264", rate=fps)
    vstream.width, vstream.height = SIZE
    vstream.pix_fmt = "yuv420p"
    vstream.options = {"crf": "30"}
    astream = container.add_stream("aac", rate=RATE, layout="stereo") if with_audio else None

    for index in range(round(fps * seconds)):
        frame = av.VideoFrame.from_ndarray(
            np.full((SIZE[1], SIZE[0], 3), 40 + index, dtype=np.uint8), format="rgb24"
        )
        frame.pts = index
        container.mux(vstream.encode(frame))
    container.mux(vstream.encode(None))

    if astream is not None:
        tone = (0.2 * np.sin(2 * np.pi * 440.0 * np.arange(RATE * SECONDS) / RATE)).astype(np.float32)
        audio = av.AudioFrame.from_ndarray(np.ascontiguousarray(np.stack([tone, tone])),
                                           format="fltp", layout="stereo")
        audio.sample_rate = RATE
        audio.pts = 0
        container.mux(astream.encode(audio))
        container.mux(astream.encode(None))
    container.close()


class AudioTrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.with_audio = str(Path(cls._dir.name) / "with.mp4")
        cls.without_audio = str(Path(cls._dir.name) / "without.mp4")
        _write_probe(cls.with_audio, True)
        _write_probe(cls.without_audio, False)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_shape_and_rate_follow_the_source(self):
        audio = native_video.read_audio_track(self.with_audio)
        self.assertIsNotNone(audio)
        self.assertEqual(audio["sample_rate"], RATE)
        waveform = audio["waveform"]
        self.assertEqual(waveform.ndim, 3, "waveform 必须是 (B, C, T)")
        self.assertEqual(waveform.shape[0], 1)
        self.assertEqual(waveform.shape[1], 2, "源是立体声，不能塌成单声道")
        self.assertGreater(waveform.shape[2], 0)
        # AAC 有编码填充，时长只要求接近源片
        self.assertAlmostEqual(waveform.shape[2] / float(RATE), SECONDS, delta=0.1)
        self.assertGreater(float(waveform.abs().max()), 0.05, "解出来不该是静音")

    def test_silent_source_returns_none_instead_of_raising(self):
        self.assertIsNone(native_video.read_audio_track(self.without_audio))

    def test_matches_the_official_components_audio_sample_for_sample(self):
        ours = native_video.read_audio_track(self.with_audio)["waveform"]
        theirs = InputImpl.VideoFromFile(self.with_audio).get_components().audio["waveform"]
        self.assertEqual(tuple(ours.shape), tuple(theirs.shape))
        self.assertTrue(torch.equal(ours, theirs), "与 GetVideoComponents 的音频必须逐样本相同")


class NodeOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.with_audio = str(Path(cls._dir.name) / "with.mp4")
        cls.without_audio = str(Path(cls._dir.name) / "without.mp4")
        _write_probe(cls.with_audio, True)
        _write_probe(cls.without_audio, False)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_audio_sits_right_below_video_and_ints_are_renamed(self):
        schema = video_resize.PROYYImageScale.define_schema()
        self.assertEqual([out.io_type for out in schema.outputs], ["VIDEO", "AUDIO", "INT", "INT"])
        self.assertEqual([out.display_name for out in schema.outputs],
                         ["视频", "音频", "目标视频宽", "目标视频高"])

    def _execute(self, path):
        with mock.patch.object(video_resize.folder_paths, "get_annotated_filepath", return_value=path):
            return video_resize.PROYYImageScale.execute(
                file=str(path), width=720, height=1280,
                upscale_method="area", crop="center", single_side=False, target_mp=0.0,
            )

    def test_execute_returns_four_outputs_with_audio(self):
        video, audio, gen_w, gen_h = self._execute(self.with_audio).result
        # 输出宽高统一按 32 向下对齐：720 -> 704
        self.assertEqual(video.get_dimensions(), (704, 1280))
        self.assertIsNotNone(audio)
        self.assertEqual(audio["waveform"].shape[1], 2)
        # 采样宽高跟随裁切后的输出尺寸（32 对齐）：720 -> 704
        self.assertEqual((gen_w, gen_h), (704, 1280))

    def test_execute_survives_a_silent_source(self):
        video, audio, gen_w, gen_h = self._execute(self.without_audio).result
        # 输出宽高统一按 32 向下对齐：720 -> 704
        self.assertEqual(video.get_dimensions(), (704, 1280))
        self.assertIsNone(audio, "没有音频轨时输出空值，而不是抛错")
        # 采样宽高跟随裁切后的输出尺寸（32 对齐）：720 -> 704
        self.assertEqual((gen_w, gen_h), (704, 1280))

    def test_read_frames_reaches_the_wrapper(self):
        with mock.patch.object(video_resize.folder_paths, "get_annotated_filepath",
                               return_value=self.with_audio):
            video, _audio, _w, _h = video_resize.PROYYImageScale.execute(
                file=str(self.with_audio), width=128, height=224,
                upscale_method="area", crop="center", single_side=False,
                target_mp=0.0, read_frames=5).result
        self.assertEqual(video.read_limit, 5)
        self.assertEqual(video.get_components().images.shape[0], 5)

    def test_skip_frames_shifts_the_timeline(self):
        """跳过 N 帧 = 把起点沿 24fps 时间轴后移 N/24 秒，音频跟着一起移。"""
        with mock.patch.object(video_resize.folder_paths, "get_annotated_filepath",
                               return_value=self.with_audio):
            plain, plain_audio, _w, _h = video_resize.PROYYImageScale.execute(
                file=str(self.with_audio), width=128, height=224,
                upscale_method="area", crop="center", single_side=False,
                target_mp=0.0).result
            skipped, skipped_audio, _w2, _h2 = video_resize.PROYYImageScale.execute(
                file=str(self.with_audio), width=128, height=224,
                upscale_method="area", crop="center", single_side=False,
                target_mp=0.0, skip_frames=6).result

        self.assertEqual(plain.start_offset, 0)
        self.assertEqual(skipped.start_offset, Fraction(6, 24))
        # 探针片正好 1 秒 24 帧：跳过 6 帧后剩 18 帧，音频短掉 0.25 秒的采样
        self.assertEqual(plain.get_frame_count(), 24)
        self.assertEqual(skipped.get_frame_count(), 18)
        rate = plain_audio["sample_rate"]
        self.assertEqual(skipped_audio["sample_rate"], rate)
        self.assertEqual(
            plain_audio["waveform"].shape[-1] - skipped_audio["waveform"].shape[-1],
            rate // 4,
        )

    def test_skip_frames_counts_source_frames_at_30fps(self):
        """skip 按**源片帧率**数帧：30fps 片子跳 24 帧 = 从第 25 帧（索引 24）开始。

        探针帧亮度是 40+源索引，所以跳过后第一帧的亮度能直接对出源帧号。
        """
        probe = str(Path(self._dir.name) / "thirty.mp4")
        _write_probe(probe, True, fps=30, seconds=1.5)   # 45 帧
        with mock.patch.object(video_resize.folder_paths, "get_annotated_filepath",
                               return_value=probe):
            video, audio, _w, _h = video_resize.PROYYImageScale.execute(
                file=str(probe), width=128, height=224,
                upscale_method="area", crop="center", single_side=False,
                target_mp=0.0, skip_frames=24).result

        self.assertEqual(video.start_offset, Fraction(24, 30))
        components = video.get_components()
        # 1.5s - 0.8s = 0.7s -> ceil(0.7*24) = 17 帧
        self.assertEqual(components.images.shape[0], 17)
        # 第一帧 = 源片索引 24（亮度 64）
        self.assertAlmostEqual(float(components.images[0].mean()), 64 / 255, delta=0.01)
        # 音频也裁掉了 0.8s（AAC 编码器会截整帧，所以基准用完整轨的实际长度）
        rate = audio["sample_rate"]
        full_length = native_video.read_audio_track(probe)["waveform"].shape[-1]
        self.assertEqual(audio["waveform"].shape[-1], full_length - round(0.8 * rate))

    def test_skip_frames_rejects_negative_values(self):
        with mock.patch.object(video_resize.folder_paths, "get_annotated_filepath",
                               return_value=self.with_audio):
            with self.assertRaises(ValueError):
                video_resize.PROYYImageScale.execute(
                    file=str(self.with_audio), width=128, height=224,
                    upscale_method="area", crop="center", single_side=False,
                    target_mp=0.0, skip_frames=-3)


class ReadLimitTests(unittest.TestCase):
    """读取上限：整片 materialize 时只解前 N 帧，且不改窗口计划的边界。

    背景：``#17`` 输出的 VIDEO 是个「延迟读取配方」，``GetVideoComponents`` 一调
    ``get_components()`` 就会把整片解出来。源片几千帧时这一步纯属白工 ——
    下游（YOLO）根本只用前 N 帧。
    """

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.path = str(Path(cls._dir.name) / "limit.mp4")
        _write_probe(cls.path, True)                     # 24 帧 / 1 秒

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _wrap(self, read_limit):
        return native_video.WindowResizedVideo(
            InputImpl.VideoFromFile(self.path), 128, 224, "area", "center",
            read_limit=read_limit)

    def test_zero_means_the_whole_clip(self):
        video = self._wrap(0)
        self.assertEqual(native_video.video_info(video)["total_frames"], FPS)
        self.assertEqual(video.get_components().images.shape[0], FPS)

    def test_limit_cuts_the_decoded_frame_count(self):
        self.assertEqual(self._wrap(6).get_components().images.shape[0], 6)

    def test_limit_above_the_source_is_clamped(self):
        self.assertEqual(self._wrap(999).get_components().images.shape[0], FPS)

    def test_limit_does_not_change_the_reported_length(self):
        """窗口计划仍按源片真实长度校验：设小了要报越界，而不是悄悄截断。"""
        video = self._wrap(6)
        self.assertEqual(native_video.video_info(video)["total_frames"], FPS)
        with self.assertRaises(ValueError):
            native_video.read_video_window(video, FPS - 2, 4, 128, 224)

    def test_audio_is_cut_to_the_same_span(self):
        audio = self._wrap(6).get_components().audio
        self.assertIsNotNone(audio)
        self.assertAlmostEqual(audio["waveform"].shape[-1] / audio["sample_rate"],
                               6 / FPS, delta=0.05)

    def test_trim_keeps_the_limit(self):
        self.assertEqual(self._wrap(6).as_trimmed(duration=0.5).read_limit, 6)

    def test_negative_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            self._wrap(-1)


if __name__ == "__main__":
    unittest.main()
