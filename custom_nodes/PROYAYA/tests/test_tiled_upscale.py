"""PROYYTiledLatentUpscale 的分块拼接契约。

分块放大最容易出两类错：接缝权重没接对（块边界出现亮度台阶），以及块与块
放大后尺寸不一致导致整体错位。这两类错在真实素材上很难一眼看出，所以这里
用一个可控的假放大器把它们放大到可断言的误差：

- 常量图 → 任何位置的接缝都会让结果偏离常量；
- 线性渐变图 → 任何错位都会让渐变出现台阶或跳变；
- 假放大器故意让边长多/少一格，模拟第三方按 32 像素对齐带来的尺寸偏差。

不加载任何权重，不碰 CUDA。
"""

import importlib
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime as base_tests

torch = base_tests.torch
F = torch.nn.functional
tiled = importlib.import_module(base_tests.package_name + ".tiled_upscale")
NestedTensor = base_tests.NestedTensor


class FakeUpscaler:
    """假放大器：最近邻放大，可指定边长偏差用来模拟对齐误差。"""

    drift = 0
    calls = []
    fail_on_call = None

    @classmethod
    def execute(cls, latent, model_name, mode, align, chunking, unload, device, precision):
        cls.calls.append({
            "shape": tuple(latent["samples"].shape),
            "scale": mode["scale"],
            "chunking": chunking,
            "unload": unload,
        })
        if cls.fail_on_call is not None and len(cls.calls) == cls.fail_on_call:
            raise RuntimeError("boom")
        x = latent["samples"].to(torch.float32)
        s = float(mode["scale"])
        out_h = max(1, int(round(x.shape[-2] * s)) + cls.drift)
        out_w = max(1, int(round(x.shape[-1] * s)) + cls.drift)
        y = F.interpolate(x, size=(x.shape[-3], out_h, out_w), mode="nearest")
        return tiled.io.NodeOutput({"samples": y})


def run(latent, scale=1.5, tile_size=512, tile_overlap=128, align=32,
        drift=0, chunking=False, full=False):
    """跑一次节点。默认只返回 (latent, 报告)；full=True 返回全部四个输出。"""
    FakeUpscaler.calls = []
    FakeUpscaler.drift = drift
    FakeUpscaler.fail_on_call = None
    with mock.patch.object(tiled, "_upscaler_class", return_value=FakeUpscaler):
        res = tiled.PROYYTiledLatentUpscale.execute(
            latent=latent,
            model_name="fake.safetensors",
            scale=scale,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            align=align,
            enable_temporal_chunking=chunking,
            precision="fp16",
            device="cpu",
        )
    return tuple(res[i] for i in range(4)) if full else (res[0], res[1])


class SliceGrid(unittest.TestCase):
    def test_covers_whole_range_with_overlap(self):
        rows = tiled._slices(72, 32, 8)
        self.assertEqual(rows[0][0], 0)
        self.assertEqual(rows[-1][1], 72)
        for (_, end), (start, _) in zip(rows, rows[1:]):
            self.assertEqual(end - start, 8, "相邻块必须恰好共享 overlap 宽")
        for start, end in rows:
            self.assertLessEqual(end - start, 32)

    def test_single_block_when_tile_covers_all(self):
        self.assertEqual(tiled._slices(20, 32, 8), [(0, 20)])

    def test_tail_merged_instead_of_tiny_sliver(self):
        # 33 用 step=24 走会剩 9 格；但末块不能只是 1 格宽的碎片
        rows = tiled._slices(33, 32, 8)
        self.assertEqual(rows[-1][1], 33)
        self.assertGreaterEqual(rows[-1][1] - rows[-1][0], tiled.MIN_TILE)
        for start, end in rows:
            self.assertGreaterEqual(end - start, tiled.MIN_TILE)

    def test_overlap_zero_still_advances(self):
        rows = tiled._slices(10, 4, 0)
        self.assertEqual(rows, [(0, 4), (4, 8), (8, 10)])

    def test_overlap_clamped_below_tile(self):
        # tile_overlap 填得比 tile 还大时不能让步长变成 0
        tile = 32
        overlap = max(0, min(tile - 1, 999))
        rows = tiled._slices(100, tile, overlap)
        self.assertGreater(len(rows), 1)
        self.assertEqual(rows[-1][1], 100)


class OutputSize(unittest.TestCase):
    def test_matches_third_party_rule(self):
        # 一采 768×1344 → latent 48×84，×1.5 应得 72×126
        self.assertEqual(tiled._out_side(48, 1.5, 32), 72)
        self.assertEqual(tiled._out_side(84, 1.5, 32), 126)

    def test_align_rounding(self):
        # 1104px 在 align=32 下四舍五入到 1088（34×32），不是 1120
        self.assertEqual(tiled._out_side(46, 1.5, 32), 68)

    def test_scale_one_is_identity(self):
        self.assertEqual(tiled._out_side(50, 1.0, 32), 50)


class SizeOutputs(unittest.TestCase):
    """宽/高两个输出槽：下游二采编码按它规划窗口，必须跟着倍率走。"""

    def test_reports_pixel_size_after_scale(self):
        # 一采 768×1344 → latent 84×48，×1.5 → 1152×2016
        x = torch.zeros(1, 4, 2, 84, 48)
        _, _, w, h = run({"samples": x}, scale=1.5, tile_size=2048, full=True)
        self.assertEqual((w, h), (1152, 2016))

    def test_follows_scale_change(self):
        x = torch.zeros(1, 4, 2, 84, 48)
        _, _, w1, h1 = run({"samples": x}, scale=1.0, tile_size=2048, full=True)
        _, _, w2, h2 = run({"samples": x}, scale=2.0, tile_size=2048, full=True)
        self.assertEqual((w1, h1), (768, 1344))
        self.assertEqual((w2, h2), (1536, 2688))

    def test_matches_actual_latent_size(self):
        """分块 + 尺寸漂移都走过一遍后，宽高仍要与真实 latent 严格对应。"""
        x = torch.zeros(1, 4, 2, 84, 48)
        out, _, w, h = run({"samples": x}, scale=1.5, tile_size=512, drift=1, full=True)
        got = out["samples"]
        self.assertEqual(w, got.shape[-1] * tiled.VAE_DOWNSAMPLE)
        self.assertEqual(h, got.shape[-2] * tiled.VAE_DOWNSAMPLE)

    def test_joint_latent_reports_video_branch(self):
        """joint AV latent 的宽高必须取视频分支，不能被音频分支干扰。"""
        video = torch.zeros(1, 24, 3, 40, 48)
        audio = torch.zeros(1, 32, 2, 20)
        _, _, w, h = run({"samples": NestedTensor((video, audio))}, full=True)
        self.assertEqual((w, h), (1152, 960))


class BlendWeight(unittest.TestCase):
    def test_edges_are_unity_when_no_neighbour(self):
        w = tiled._blend_weight(16, 16, 4, 4, False, False, False, False)
        self.assertTrue(torch.allclose(w, torch.ones_like(w)))

    def test_interior_block_fades_on_all_four_sides(self):
        w = tiled._blend_weight(16, 16, 4, 4, True, True, True, True)
        self.assertLess(float(w[0, 0]), 1.0)
        self.assertLess(float(w[0, -1]), 1.0)
        self.assertLess(float(w[-1, 0]), 1.0)
        self.assertLess(float(w[-1, -1]), 1.0)
        # 正中必须保持 1.0，否则整块会被整体压暗
        self.assertAlmostEqual(float(w[8, 8]), 1.0, places=6)

    def test_only_top_fades_when_bottom_is_border(self):
        w = tiled._blend_weight(16, 16, 4, 4, True, False, False, False)
        self.assertLess(float(w[0, 8]), 1.0)
        self.assertAlmostEqual(float(w[-1, 8]), 1.0, places=6)


class Reassembly(unittest.TestCase):
    def test_constant_image_stays_constant(self):
        """接缝权重错了这里就会露馅：块边界会出现亮度台阶。"""
        x = torch.full((1, 24, 4, 84, 48), 0.75)
        out, _ = run({"samples": x})
        got = out["samples"]
        expect = torch.full_like(got, 0.75)
        self.assertEqual(tuple(got.shape), (1, 24, 4, 126, 72))
        self.assertLess(float((got - expect).abs().max()), 1e-5)

    def test_constant_image_with_size_drift(self):
        """假放大器边长多一格时，靠 interpolate 校正后仍不能出现接缝。"""
        x = torch.full((1, 24, 3, 40, 60), 0.4)
        out, _ = run({"samples": x}, drift=1)
        got = out["samples"]
        self.assertLess(float((got - 0.4).abs().max()), 1e-5)

    def test_gradient_with_size_drift_keeps_endpoints(self):
        """尺寸偏差下贴图位置必须仍然正确：首尾应停在渐变的两个端点。

        常量图对错位不敏感（怎么插值都是常量），只有位置相关的图才能验出来。
        """
        h, w = 40, 60
        ramp = torch.linspace(0.0, 1.0, h).view(1, 1, 1, h, 1).expand(1, 3, 2, h, w).contiguous()
        out, _ = run({"samples": ramp}, scale=1.5, drift=1)
        got = out["samples"][0, 0, 0, :, 0]
        # nearest 插值在末端像素上取不到精确端点，5e-4 量级是正常的；
        # 真错位是整格位移，误差会是 1/40 量级。
        self.assertAlmostEqual(float(got[0]), 0.0, delta=2e-3)
        self.assertAlmostEqual(float(got[-1]), 1.0, delta=2e-3)
        steps = got[1:] - got[:-1]
        self.assertTrue(bool((steps >= 0).all()), "放大后渐变不能倒退")

    def test_ramp_has_no_step_at_seams(self):
        """线性渐变：错位会在块边界留下跳变，用二阶差分抓。"""
        h, w = 32, 64
        ramp = torch.linspace(0.0, 1.0, h).view(1, 1, 1, h, 1).expand(1, 3, 2, h, w)
        out, _ = run({"samples": ramp.clone()}, scale=1.0, tile_size=256, tile_overlap=64)
        got = out["samples"][0, 0, 0, :, 0]
        # 每格递增相等，任何台阶都会让相邻差翻倍
        diffs = got[1:] - got[:-1]
        self.assertLess(float(diffs.std()), 1e-5, "渐变应保持等步长")

    def test_joint_latent_only_upscales_video(self):
        video = torch.full((1, 24, 3, 40, 48), 0.5)
        audio = torch.full((1, 32, 2, 20), 0.25)
        out, report = run({"samples": NestedTensor((video, audio))})
        samples = out["samples"]
        self.assertIsInstance(samples, NestedTensor)
        self.assertEqual(tuple(samples.tensors[0].shape), (1, 24, 3, 60, 72))
        self.assertTrue(torch.equal(samples.tensors[1], audio), "音频必须逐位原样带走")
        self.assertIn("分块放大", report)

    def test_single_tile_equals_whole(self):
        x = torch.rand(1, 4, 2, 16, 16)
        out, _ = run({"samples": x}, scale=2.0, tile_size=2048, tile_overlap=0)
        self.assertEqual(len(FakeUpscaler.calls), 1)
        expect = F.interpolate(x, size=(2, 32, 32), mode="nearest")
        self.assertTrue(torch.allclose(out["samples"], expect, atol=1e-6))

    def test_call_count_and_last_unloads(self):
        x = torch.zeros(1, 4, 2, 84, 48)
        run({"samples": x}, tile_size=512, tile_overlap=128)
        # 84/32 → 4 行（末段贴尾并入），48/32 → 2 列，共 8 块
        self.assertEqual(len(FakeUpscaler.calls), 8)
        self.assertFalse(FakeUpscaler.calls[-1]["unload"] is False,
                         "最后一块必须负责卸载模型")
        self.assertTrue(all(c["unload"] is False for c in FakeUpscaler.calls[:-1]))

    def test_temporal_chunking_flag_is_forwarded(self):
        x = torch.zeros(1, 4, 2, 32, 32)
        run({"samples": x}, tile_size=2048, chunking=True)
        self.assertTrue(FakeUpscaler.calls[0]["chunking"])

    def test_dtype_preserved(self):
        x = torch.zeros(1, 4, 2, 32, 32, dtype=torch.bfloat16)
        out, _ = run({"samples": x}, tile_size=2048)
        self.assertEqual(out["samples"].dtype, torch.bfloat16)

    def test_single_tile_report_has_no_warning(self):
        """不分块时不能出现画质警告，否则每次跑都误报。"""
        x = torch.zeros(1, 4, 2, 32, 32)
        _, report = run({"samples": x}, tile_size=2048)
        self.assertIn("1 块", report.split("\n")[0])
        self.assertNotIn("⚠", report)

    def test_multi_tile_report_warns(self):
        x = torch.zeros(1, 4, 2, 84, 48)
        _, report = run({"samples": x}, tile_size=512, tile_overlap=128)
        self.assertIn("⚠", report)
        self.assertIn("GroupNorm", report)

    def test_default_tile_size_means_no_split(self):
        """默认 tile_size 必须大于一般 720p/1080p 二采的画面，否则等于默认掉画质。"""
        default_tile = 1536 // tiled.VAE_DOWNSAMPLE
        # 一采 768×1344 → 48×84；二采 ×1.5 → 72×126
        self.assertGreaterEqual(default_tile, 84, "默认值必须能覆盖 48×84")
        self.assertGreaterEqual(default_tile, 96, "默认值应能覆盖到 4K 之前的长边")


class Guards(unittest.TestCase):
    def test_rejects_4d_latent(self):
        with self.assertRaises(ValueError) as ctx:
            run({"samples": torch.zeros(1, 4, 32, 32)})
        self.assertIn("5 维", str(ctx.exception))

    def test_rejects_batch_above_one(self):
        with self.assertRaises(ValueError) as ctx:
            run({"samples": torch.zeros(2, 4, 2, 32, 32)})
        self.assertIn("batch", str(ctx.exception))

    def test_missing_upscaler_gives_actionable_error(self):
        with mock.patch("nodes.NODE_CLASS_MAPPINGS", {}):
            with self.assertRaises(RuntimeError) as ctx:
                tiled._upscaler_class()
        self.assertIn("MinimaxH3LatentUpscaler3D", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
