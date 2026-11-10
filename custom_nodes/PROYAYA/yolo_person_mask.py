"""Per-frame person mask from YOLO instance segmentation.

Why this exists
---------------
SAM3 video tracking is the accurate option, but its weights (1.7 GB) plus
activations (4~6 GB) cannot coexist with H3's qwen3vl-32b text encoder
(14956 MB staged on a 16 GB card). A YOLOv8 segmentation model weighs 52 MB and
peaks at ~234 MB of VRAM while processing 720x1280 frames at 22 ms/frame, so the
mask can be produced inside the same graph as the sampler.

Trade-off: YOLO runs per frame with no cross-frame identity, so two people in
frame are both masked. ``largest_only`` keeps just the biggest instance, which is
enough for single-subject replacement work.

Edge control
------------
``expand`` is a Euclidean distance-transform grow/shrink in source pixels, not a
morphological kernel, so the offset is isotropic and exact for any radius.
``feather`` is a Gaussian sigma applied after the expand.

Note on feather: ``_video_mask_to_latent`` reduces the mask with
``adaptive_max_pool2d``, so a soft band narrower than one latent cell collapses
onto the cell's max value. At 720p the latent cell spans roughly 20 px, so a
feather below ~10 px barely survives the downscale. Keep it as a seam softener,
not as a way to reshape the mask.
"""

from __future__ import annotations

import json
import os

import folder_paths
import numpy as np
import torch

__all__ = ["PROYAYA_YoloPersonMask"]

_COMFY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_SEGM_DIR = os.path.join(_COMFY_ROOT, "models", "ultralytics", "segm")
_MODEL_CACHE: dict = {}


def _segm_models():
    """List segm/*.pt the way Impact's UltralyticsDetectorProvider does."""
    names = []
    try:
        names = [f"segm/{n}" for n in folder_paths.get_filename_list("ultralytics_segm")]
    except Exception:
        names = []
    if not names:
        try:
            names = [n for n in folder_paths.get_filename_list("ultralytics")
                     if os.path.basename(n).endswith("-seg.pt")]
        except Exception:
            names = []
    if not names and os.path.isdir(_DEFAULT_SEGM_DIR):
        names = [f"segm/{n}" for n in sorted(os.listdir(_DEFAULT_SEGM_DIR)) if n.endswith(".pt")]
    return names or ["<未找到 segm 模型：请放入 models/ultralytics/segm>"]


# 新建节点默认选中的分割模型。本机没有这个文件时退回列表第一项，
# 否则一拖出节点就是「模型不在列表里」的校验错误。
PREFERRED_MODEL = "segm/person_yolov8m-seg.pt"


def _default_model(options):
    if PREFERRED_MODEL in options:
        return PREFERRED_MODEL
    return options[0] if options else ""


def _resolve_model(name: str) -> str:
    base = name.split("/", 1)[-1]
    for folder, key in (("ultralytics", name), ("ultralytics_segm", base)):
        try:
            path = folder_paths.get_full_path(folder, key)
        except Exception:
            path = None
        if path and os.path.isfile(path):
            return path
    path = os.path.join(_DEFAULT_SEGM_DIR, base)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"找不到分割模型 {name!r}；已查找 models/ultralytics/segm/")


def _load_model(model_path: str, device: str):
    key = (model_path, device)
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO

        model = YOLO(model_path)
        try:
            model.to(device)
        except Exception:
            pass
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _expand(mask_u8: np.ndarray, px: int) -> np.ndarray:
    """Grow (px>0) or shrink (px<0) by a Euclidean radius, in pixels."""
    import cv2

    if px == 0:
        return mask_u8
    if px > 0:
        # distance to the nearest background pixel
        dist = cv2.distanceTransform((mask_u8 < 128).astype(np.uint8), cv2.DIST_L2, 5)
        return np.where(dist <= px, 255, 0).astype(np.uint8)
    dist = cv2.distanceTransform((mask_u8 >= 128).astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(dist > -px, 255, 0).astype(np.uint8)


def _feather(mask_u8: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    if sigma <= 0.0:
        return mask_u8.astype(np.float32) / 255.0
    return cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (0, 0), sigma,
                            borderType=cv2.BORDER_REPLICATE)


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill background components that do not touch the frame border.

    Person masks often have enclosed background gaps (the triangle between two
    raised arms, between legs). Left unfilled, a composite would paste the
    ORIGINAL person back through those gaps. A component counts as a hole when
    its tight bounding box does not reach the frame border.
    """
    import cv2

    bg = (mask_u8 < 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    h, w = mask_u8.shape
    holes = np.zeros((h, w), np.uint8)
    for lab in range(1, n):
        x, y, bw, bh, _ = stats[lab]
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            holes[labels == lab] = 255
    if not holes.any():
        return mask_u8
    return np.maximum(mask_u8, holes)


class PROYAYA_YoloPersonMask:
    @classmethod
    def INPUT_TYPES(cls):
        models = _segm_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (models, {"default": _default_model(models),
                                        "tooltip": "segm/*.pt。person_yolov8m-seg.pt 就够用。"}),
                "conf": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                                   "tooltip": "检测置信度阈值。调低会多检出、掩码松动，不适合用来微调边缘。"}),
                "expand": ("INT", {"default": 48, "min": -128, "max": 512, "step": 1,
                                   "tooltip": "边缘外扩像素（负值向内收缩），欧氏距离变换，各向同性。\n"
                                              "720p 下 +48 ≈ 2 个 latent 格。欠覆盖会留原人物残影，"
                                              "过覆盖只是多重绘一点背景。"}),
                "feather": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.5,
                                      "tooltip": "边缘高斯羽化 sigma（源像素）。\n"
                                                 "注意：遮罩进 latent 时走 max-pool，"
                                                 "窄于 1 个 latent 格（720p 约 20px）的柔边会被吃掉。"}),
                "largest_only": ("BOOLEAN", {"default": True,
                                             "tooltip": "每帧只保留面积最大的那个人。多人画面里锁定主体。"}),
                "max_det": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1,
                                    "tooltip": "每帧最多检测几个实例。"}),
                "fill_missing": ("BOOLEAN", {"default": True,
                                             "tooltip": "某帧没检出时沿用上一帧掩码，避免中途出现黑帧打洞。"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto",
                                                     "tooltip": "cpu 完全不占显存，124 帧约 1~2 分钟。"}),
                "emit_preview": ("BOOLEAN", {"default": False,
                                             "tooltip": "输出红色叠加的预览图。开启会额外占用与输入同尺寸的内存"
                                                        "（355 帧 720x1280 约 3.9 GB），只在需要肉眼看遮罩时打开。"}),
                "fill_holes": ("BOOLEAN", {"default": True,
                                           "tooltip": "填掉人物内部的封闭背景洞（举臂时两臂之间的三角区等）。\n"
                                                      "用于合成时必开，否则洞里会露出原片人物。",
                                           }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "preview", "run_report")
    FUNCTION = "run"
    CATEGORY = "PROYAYA/H3"
    DESCRIPTION = ("逐帧 YOLO 人物分割 → 遮罩。权重 52 MB、峰值显存约 234 MB，"
                   "可与 H3 主流程同图共存。expand 控制外扩像素，feather 控制边缘柔化。")

    def run(self, image, model_name, conf, expand, feather, largest_only, max_det,
            fill_missing, device, emit_preview=False, fill_holes=True):
        import cv2

        if image.ndim != 4:
            raise ValueError(f"期望 IMAGE 形如 [N,H,W,3]，得到 {tuple(image.shape)}")
        n, h, w, _ = image.shape
        dev = device
        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = _load_model(_resolve_model(model_name), dev)

        frames = (image.detach().cpu().numpy()[..., :3].clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        out = np.zeros((n, h, w), np.float32)
        detected = 0
        filled = 0
        prev = None

        for i in range(n):
            bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            result = model.predict(bgr, classes=[0], conf=float(conf), max_det=int(max_det),
                                   verbose=False, device=dev, retina_masks=False)[0]

            mask = np.zeros((h, w), np.uint8)
            if result.masks is not None and len(result.masks.data):
                inst = result.masks.data.cpu().numpy()
                if largest_only and len(inst) > 1:
                    areas = [float((m > 0.5).sum()) for m in inst]
                    inst = inst[[int(np.argmax(areas))]]
                for m in inst:
                    up = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask = np.maximum(mask, (up > 0.5).astype(np.uint8) * 255)

            if mask.any():
                detected += 1
                prev = mask.copy()
            elif fill_missing and prev is not None:
                mask = prev.copy()
                filled += 1

            if fill_holes:
                mask = _fill_holes(mask)
            out[i] = _feather(_expand(mask, int(expand)), float(feather))

        tensor = torch.from_numpy(out)
        if emit_preview:
            red = torch.tensor([0.9, 0.15, 0.1], dtype=torch.float32)
            alpha = tensor.unsqueeze(-1)
            preview = (image.detach().cpu()[..., :3] * (1.0 - 0.55 * alpha) + red * 0.55 * alpha).clamp(0, 1)
        else:
            # 直接透传输入，避免复制一份与输入同尺寸的张量
            preview = image.detach().cpu()[..., :3]

        report = {
            "frames": int(n),
            "size": [int(w), int(h)],
            "model": model_name,
            "device": dev,
            "conf": float(conf),
            "expand_px": int(expand),
            "feather_sigma": float(feather),
            "largest_only": bool(largest_only),
            "emit_preview": bool(emit_preview),
            "fill_holes": bool(fill_holes),
            "frames_with_detection": detected,
            "frames_filled_from_previous": filled,
            "frames_empty": int(n - detected - filled),
            "mask_mean": round(float(out.mean()), 5),
            "mask_max": round(float(out.max()), 5),
        }
        if feather > 0 and feather < 10:
            report["warning"] = ("feather 小于 1 个 latent 格（720p 约 20px），"
                                 "进 latent 时会被 max-pool 吃掉，柔边基本不可见。")

        if dev == "cuda":
            torch.cuda.empty_cache()
        text = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"text": [text]}, "result": (tensor, preview, text)}
