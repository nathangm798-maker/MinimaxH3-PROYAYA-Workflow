"""把 H3 视频 latent 的空间放大切成 tile，逐块调用第三方 H3 latent upscaler。

**为什么需要这一层**：``MinimaxH3LatentUpscaler3D`` 只有时间分块。它在
``H3Upscaler3D.forward`` 里切的是

    out_full = torch.zeros(B, C, T, size[-2], size[-1])   # 时间维 = T
    seg      = x_padded[:, :, lo:hi]                      # 只截时间
    seg_size = (hi - lo, size[-2], size[-1])              # H/W 恒为全尺寸

也就是说 ``chunk=32`` 只解决「帧多」，空间面积一大，激活值照样顶满显存，
没有任何退路。这个节点在它外面套一层空间 tile，块内仍调它自己的 forward。

**但切块不是免费的吗（实测结论，别删）**：模型里 ``normalization`` 是
``nn.GroupNorm(32, C)``，统计范围是整张 ``T×H×W``。切块后每块在**自己的小块上**
重新统计均值方差，于是
- 块与块之间的平均亮度/对比度不一致（84×48 输入切 8 块时，块间增益极差 0.387，
  即相邻块能差三成以上）；整体 std 也会漂移。
- 重叠融合只能平滑过渡带，改变不了「块中心就是按局部统计算出来的」。

实测（真实放大器，84×48 → 126×72，平滑内容探针）：
    整张 1 块      残差 0        块间增益极差 0
    tile 768 2 块  残差 23%      极差 6.8%
    tile 512 8 块  残差 46%      极差 38%
（纯白噪声反而好些：20% / 3.8%——大面积同质区域让局部统计偏离全局更远。）

所以 **默认 tile_size=1536 不切块**，只在整张放大 OOM 时才调小。

融合方式：相邻 tile 的重叠区用线性 ramp 加权平均。tile 的输出尺寸先
``F.interpolate`` 校正到「按全局缩放比算出的理论尺寸」，再贴到画布上，
因此块与块之间不会因为放大器的 32 像素对齐而错位。

输入可以是单视频 latent，也可以是 H3 的 joint AV latent —— 后者只放大视频分支，
音频分支原样带走。
"""

import logging

import torch
import torch.nn.functional as F

from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

# H3 视频 VAE 的空间压缩率，与 comfy_extras/nodes_minimax_h3.py 的 height // 16 一致
VAE_DOWNSAMPLE = 16
# 单块小于这个 latent 尺寸就没意义了（放大器的卷积核都铺不开）
MIN_TILE = 2


def _upscaler_class():
    """从 ComfyUI 全局注册表取第三方放大器。

    不用 ``import Comfyui_Minimax_h3_latent_Upscaler...``：那个包把类注册进
    NODE_CLASS_MAPPINGS 时用的是 io.ComfyNode，模块路径不一定在 sys.path 上；
    走全局表是唯一稳定的拿法，而且没装的时候能给出人话报错。
    """
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
    if cls is None:
        raise RuntimeError(
            "PROYY Tiled Latent Upscale 依赖 Comfyui_Minimax_h3_latent_Upscaler 提供的 "
            "MinimaxH3LatentUpscaler3D，当前环境里没有注册这个节点。"
        )
    return cls


def _slices(length, tile, overlap):
    """把 [0, length) 切成覆盖全程的区间，相邻区间共享 ``overlap`` 宽。"""
    if tile >= length:
        return [(0, length)]
    step = max(1, tile - overlap)
    out = []
    start = 0
    while True:
        end = min(length, start + tile)
        # 尾巴太窄就并进当前块，避免出现 1~2 格宽的碎块
        if out and length - end < MIN_TILE:
            out.append((start, length))
            break
        out.append((start, end))
        if end >= length:
            break
        start = end - overlap
    return out


def _ramp(count, rising):
    """长度 count 的线性权重，rising=True 从低到高。两端都不到 1，避免硬边。"""
    if count <= 0:
        return None
    values = torch.arange(1, count + 1, dtype=torch.float32) / (count + 1)
    return values if rising else values.flip(0)


def _blend_weight(h, w, ov_h, ov_w, has_top, has_bottom, has_left, has_right):
    """给一块 h×w 的图算融合权重，只在「确实有邻居」的那几侧做渐变。"""
    wy = torch.ones((h, 1), dtype=torch.float32)
    if ov_h > 0:
        k = min(ov_h, h)
        if has_top:
            wy[:k, 0] *= _ramp(k, rising=True)
        if has_bottom:
            wy[h - k:, 0] *= _ramp(k, rising=False)

    wx = torch.ones((1, w), dtype=torch.float32)
    if ov_w > 0:
        k = min(ov_w, w)
        if has_left:
            wx[0, :k] *= _ramp(k, rising=True)
        if has_right:
            wx[0, w - k:] *= _ramp(k, rising=False)

    return wy * wx


def _out_side(latent_side, scale, align):
    """按第三方放大器的同一套规则算放大后的 latent 边长。

    它内部是 ``round(px / align) * align`` 再 ``// downsample``，这里必须完全一致，
    否则融合时会整体错位。
    """
    alignment = max(1, int(align))
    px = round(latent_side * VAE_DOWNSAMPLE * scale / alignment) * alignment
    px = round(px / VAE_DOWNSAMPLE) * VAE_DOWNSAMPLE
    return max(1, int(px // VAE_DOWNSAMPLE))


class PROYYTiledLatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = _upscaler_models()
        return io.Schema(
            node_id="PROYYTiledLatentUpscale",
            display_name="۞▷Latent Upscale◁۞",
            category="PROYAYA/H3",
            description=(
                "把视频 latent 的空间放大切成 tile 再逐块放大，用于大分辨率二采省显存。"
                "块内仍调用 Minimax H3 Latent Upscaler (3D) 的 forward，重叠区线性融合，"
                "块与块之间不会因 32 像素对齐错位。"
                "输入 joint AV latent 时只放大视频分支，音频原样带走。\n"
                "**默认不切块**（tile_size 1536 大于一般 720p/1080p 二采的画面）。"
                "切块会改变 GroupNorm 的统计范围，块间产生可测的明暗差，"
                "只有在整张放大爆显存时才把 tile_size 调小。"
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="视频 latent，或 H3 的 joint AV latent（音频分支不放大）。"),
                io.Combo.Input("model_name", options=models, default=_default_model(models),
                               tooltip="放大模型。列表来自 latent_upscale_models 目录。"),
                io.Float.Input("scale", default=1.5, min=1.0, max=4.0, step=0.05, round=0.01,
                               tooltip="放大倍率。输出边长 = 输入边长 × 本值，按 32 像素对齐（与第三方节点一致）。"),
                io.Int.Input("tile_size", display_name="分块尺寸(px)", default=1536, min=64, max=8192, step=32,
                             tooltip="单块的像素边长，会换算成 latent 格数（÷16）。\n"
                                     "**默认 1536 等于不分块**——只要它大于画面任一边就不再切。\n"
                                     "切块有画质代价：模型用 GroupNorm(32) 在整张 T×H×W 上统计，\n"
                                     "切块后每块独立重算均值方差，块间会出现明暗差。\n"
                                     "实测 84×48 输入、8 块（512）时块间增益差 38%、与整张残差 46%；\n"
                                     "2 块（768）降到 6.8% / 23%。只在整张放大 OOM 时才调小。"),
                io.Int.Input("tile_overlap", display_name="分块重叠(px)", default=256, min=0, max=4096, step=32,
                             tooltip="相邻块的重叠像素，会换算成 latent 格数（÷16）。\n"
                                     "重叠只能平滑过渡带，不能消除块中心的统计偏差；\n"
                                     "不分块时这个值不起作用。"),
                io.Int.Input("align", default=32, min=1, max=512, step=1,
                             tooltip="像素对齐栅格，交给第三方放大器的 align 参数。32 是它推荐的硬约束。"),
                io.Boolean.Input("enable_temporal_chunking", display_name="时间分块", default=False,
                                 tooltip="转交给放大器的 enable_temporal_chunking。\n"
                                         "空间已经切块后，显存不再由帧数主导，通常可以关掉——\n"
                                         "开着会让每块再切一次时间，forward 次数翻倍。"),
                io.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="分块报告"),
                # 放大后的像素宽高，给下游编码节点用：倍率改了，下游尺寸要跟着改，
                # 手填容易和倍率脱节（改 scale 忘了改宽高，二采编码就按旧尺寸规划窗口）。
                io.Int.Output(display_name="宽",
                              tooltip="放大后画面宽度（像素）= 放大后 latent 宽 × 16。"),
                io.Int.Output(display_name="高",
                              tooltip="放大后画面高度（像素）= 放大后 latent 高 × 16。"),
            ],
        )

    @classmethod
    def execute(cls, latent, model_name, scale, tile_size, tile_overlap, align,
                enable_temporal_chunking, precision, device) -> io.NodeOutput:
        samples = latent["samples"]
        is_nested = isinstance(samples, NestedTensor)
        if is_nested:
            video, side = samples.tensors[0], tuple(samples.tensors[1:])
        else:
            video, side = samples, ()

        if video.dim() != 5:
            raise ValueError(
                f"需要 5 维视频 latent [B,C,T,H,W]，当前是 {video.dim()} 维 {tuple(video.shape)}。")
        if int(video.shape[0]) != 1:
            raise ValueError(f"只支持 batch=1，当前 batch={int(video.shape[0])}。")

        _b, channels, frames, height, width = video.shape
        orig_dtype = video.dtype

        scale = float(scale)
        out_h = _out_side(height, scale, align)
        out_w = _out_side(width, scale, align)
        tile = max(MIN_TILE, int(tile_size) // VAE_DOWNSAMPLE)
        overlap = max(0, min(tile - 1, int(tile_overlap) // VAE_DOWNSAMPLE))

        rows = _slices(height, tile, overlap)
        cols = _slices(width, tile, overlap)

        # 有邻居的那一侧才做渐变；单块覆盖全程时两侧都不衰减
        ov_h_out = int(round(overlap * out_h / max(1, height)))
        ov_w_out = int(round(overlap * out_w / max(1, width)))

        canvas = torch.zeros((channels, frames, out_h, out_w), dtype=torch.float32)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

        upscaler = _upscaler_class()
        mode = {"mode": "scale by multiplier", "scale": scale}
        total = len(rows) * len(cols)
        index = 0

        logging.info(
            "PROYY Tiled Upscale: latent %d×%d → %d×%d | tile %d×%d latent (overlap %d) | %d 块",
            height, width, out_h, out_w, tile, tile, overlap, total)

        for ri, (r0, r1) in enumerate(rows):
            oh0 = int(round(r0 * out_h / height))
            oh1 = int(round(r1 * out_h / height))
            oh1 = max(oh0 + 1, min(out_h, oh1))
            for ci, (c0, c1) in enumerate(cols):
                ow0 = int(round(c0 * out_w / width))
                ow1 = int(round(c1 * out_w / width))
                ow1 = max(ow0 + 1, min(out_w, ow1))

                index += 1
                piece_in = video[:, :, :, r0:r1, c0:c1].contiguous()
                # 最后一块负责把模型踢回 CPU，前面各块都留在显存里复用
                last = index == total
                result = upscaler.execute(
                    {"samples": piece_in}, model_name, mode, int(align),
                    bool(enable_temporal_chunking), last, str(device), str(precision))
                piece = result.result[0]["samples"] if hasattr(result, "result") else result[0]["samples"]

                target = (oh1 - oh0, ow1 - ow0)
                if tuple(piece.shape[-2:]) != target:
                    # 放大器按 32 像素对齐，块边长不同会带来 1~2 格差异；
                    # 拉回理论尺寸，保证贴到画布时严格对齐。
                    piece = F.interpolate(
                        piece.to(torch.float32),
                        size=(piece.shape[-3],) + target,
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    piece = piece.to(torch.float32)

                weight = _blend_weight(
                    target[0], target[1], ov_h_out, ov_w_out,
                    has_top=ri > 0, has_bottom=ri < len(rows) - 1,
                    has_left=ci > 0, has_right=ci < len(cols) - 1,
                )
                canvas[:, :, oh0:oh1, ow0:ow1] += piece[0] * weight
                weights[:, :, oh0:oh1, ow0:ow1] += weight

                logging.info("  tile %d/%d  latent[%d:%d, %d:%d] → [%d:%d, %d:%d]",
                             index, total, r0, r1, c0, c1, oh0, oh1, ow0, ow1)

        canvas = canvas / weights.clamp(min=1e-8)
        out = canvas.unsqueeze(0).to(dtype=orig_dtype)

        if is_nested:
            payload = NestedTensor((out,) + side)
        else:
            payload = out

        report = (
            f"分块放大：{len(rows)}行 × {len(cols)}列 = {total} 块"
            f"（块 {tile}×{tile} latent，重叠 {overlap}）\n"
            f"latent {height}×{width} → {out_h}×{out_w}"
            f"（像素 {width * VAE_DOWNSAMPLE}×{height * VAE_DOWNSAMPLE} → "
            f"{out_w * VAE_DOWNSAMPLE}×{out_h * VAE_DOWNSAMPLE}）\n"
            f"时间分块：{'开' if enable_temporal_chunking else '关'}；"
            f"精度 {precision}；设备 {device}"
        )
        if total > 1:
            report += (
                f"\n⚠ {total} 块：GroupNorm 的统计范围被切开，块间会有可测的明暗差，"
                "与整张放大不等价。显存够时把 tile_size 调大（大于画面长边）即可完全避开。"
            )
        return io.NodeOutput(
            {"samples": payload},
            report,
            out_w * VAE_DOWNSAMPLE,
            out_h * VAE_DOWNSAMPLE,
        )


def _upscaler_models():
    """放大模型列表；第三方包没装时给空列表，节点仍能创建（执行时才报错）。"""
    import folder_paths

    try:
        return folder_paths.get_filename_list("latent_upscale_models") or []
    except Exception as exc:  # noqa: BLE001 - 目录不存在时不该让整包加载失败
        logging.warning("PROYY Tiled Upscale: 读取 latent_upscale_models 失败(%s)", exc)
        return []


# 新建节点默认选中的放大模型。本机没有这个文件时退回列表第一项，
# 否则一拖出节点就是「模型不在列表里」的校验错误。
PREFERRED_MODEL = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"


def _default_model(options):
    if PREFERRED_MODEL in options:
        return PREFERRED_MODEL
    return options[0] if options else ""
