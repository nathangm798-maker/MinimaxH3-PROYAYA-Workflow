"""PROYAYA RefMod inject — append RefMod visual latents to H3 joint windows.

Sits between PROYY MiniMax H3 Reference to Window and the PROYAYA sampler.
The windows payload is a plain list of dicts (PROYAYA/common.py), so the
ref blocks can be appended without touching PROYAYA's source.

Block shape follows the native REF2VA / RefMod Apply convention:
    {"kind": "video", "latent": [1,24,T,h,w], "latent_t": T,
     "latent_h": h, "latent_w": w, "ref_audio_t": 0, "audio_latent": None}
Only the positive conditioning receives refs; the negative stays clean so
CFG behaves exactly like the native reference node.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import folder_paths
from safetensors import safe_open

META_KEY = "refmod_meta"
SKIP_DIRS = {"graph_presets", "__pycache__", ".git"}
MAX_FRAMES_CAP = 64
PLACEHOLDER = "<no refmod found>"
# 下拉框里的「不选择」项，固定放第一项 = 新建节点的默认值。
NO_MOD = ""

# path -> ((mtime_ns, size), has_meta)。只读 metadata header，命中后连 open 都省了。
_META_CACHE: Dict[str, Tuple[Tuple[int, int], bool]] = {}


def _has_refmod_meta(path: str) -> bool:
    """按 (mtime, size) 缓存判定结果；文件一变缓存自动失效。"""
    try:
        stat = os.stat(path)
        key = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return False
    cached = _META_CACHE.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        ok = META_KEY in meta
    except Exception:
        ok = False
    _META_CACHE[path] = (key, ok)
    return ok


def _registered_files(folder: str) -> List[str]:
    """用 folder_paths 注册表列文件，这样 extra_model_paths.yaml 里的额外根也覆盖到。"""
    try:
        names = folder_paths.get_filename_list(folder)
    except Exception:
        return []
    out = []
    for name in names:
        if not str(name).lower().endswith(".safetensors"):
            continue
        out.append(name.replace("\\", "/"))
    return out


def _mod_candidates() -> List[Tuple[str, str]]:
    """(界面显示名, 绝对路径)。

    显示名 = 相对所在根的路径，正斜杠。云端把 RefMod 当普通 LoRA 传上来时
    落在 models/loras 任意子目录，这里都能列出来。
    """
    seen_files = set()
    out: List[Tuple[str, str]] = []

    def add(display: str, path: Optional[str]) -> None:
        if not path:
            return
        real = os.path.abspath(path)
        if real in seen_files:
            return
        seen_files.add(real)
        out.append((display, real))

    for folder in ("loras", "refmods"):
        for name in _registered_files(folder):
            add(name, folder_paths.get_full_path(folder, name))

    # refmods 未必注册进 folder_paths，保留硬扫作为兜底
    refmod_dir = os.path.join(folder_paths.models_dir, "refmods")
    if os.path.isdir(refmod_dir):
        for base, dirs, files in os.walk(refmod_dir):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                if name.lower().endswith(".safetensors"):
                    full = os.path.join(base, name)
                    add(os.path.relpath(full, refmod_dir).replace("\\", "/"), full)

    return out


def _display_names(pairs: List[Tuple[str, str]]) -> List[str]:
    """文件名全局唯一时只显示文件名，重名才带上子目录。

    这样既有工作流里存的裸文件名不用改，出现同名 RefMod 时又能区分开。
    """
    grouped: Dict[str, List[str]] = {}
    for display, _path in pairs:
        grouped.setdefault(display.rsplit("/", 1)[-1], []).append(display)
    names: List[str] = []
    for base, displays in grouped.items():
        names.extend([base] if len(displays) == 1 else sorted(displays))
    return names


def _list_mods() -> List[str]:
    """列全部 LoRA（通用加载器行为），真正的 RefMod 置顶。

    不做元数据过滤：云端上传/转存可能丢掉 refmod_meta，过滤会让人以为文件没上传成功。
    选错文件时由 _build_block 报明确错误，而不是让它在列表里凭空消失。

    第一项固定是空字符串，代表「不选择」——组合框默认取第一项，所以新建节点时
    不会自动挑一个文件。
    """
    candidates = _mod_candidates()
    refmods = [(d, p) for d, p in candidates if _has_refmod_meta(p)]
    others = [(d, p) for d, p in candidates if not _has_refmod_meta(p)]
    names = _display_names(refmods) + _display_names(others)
    return [NO_MOD] + (names or [PLACEHOLDER])


def _require_mod(mod: str) -> str:
    """空选择时给一句能看懂的话，而不是「文件未找到：」。"""
    if not str(mod or "").strip():
        raise ValueError(
            "未选择 RefMod 文件：请在下方下拉框里选一个。\n"
            "  可选项来自 models/refmods 与 models/loras，带 refmod_meta 的 RefMod 排在最前。"
        )
    return mod


def _match(rel: str) -> List[Tuple[str, str]]:
    """三级匹配：精确 → 路径后缀 → 文件名。返回所有命中，交给调用方判歧义。"""
    candidates = _mod_candidates()
    exact = [(d, p) for d, p in candidates if d == rel]
    if exact:
        return exact
    tail = "/" + rel
    suffix = [(d, p) for d, p in candidates if d.endswith(tail)]
    if suffix:
        return suffix
    base = rel.rsplit("/", 1)[-1]
    return [(d, p) for d, p in candidates if d.rsplit("/", 1)[-1] == base]


def _resolve(rel: str) -> Optional[str]:
    rel = (rel or "").replace("\\", "/").strip().strip("/")
    if not rel or rel == PLACEHOLDER:
        return None
    hits = _match(rel)
    if len(hits) == 1:
        return hits[0][1]
    if len(hits) > 1:
        names = ", ".join(sorted(d for d, _ in hits)[:6])
        raise ValueError(f"RefMod 名称有歧义：{rel} 命中了多个文件（{names}…）。"
                         f"请在下拉里选带子目录的完整相对路径。")
    return None


def _latent_keys(keys: List[str]) -> List[str]:
    def index(key: str) -> int:
        tail = key.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 999

    return sorted((k for k in keys if k.startswith("ref_")), key=index)


def _visual_member_index(members: List[Dict]) -> Optional[int]:
    for i, member in enumerate(members):
        if str(member.get("kind", "")).lower() != "audio":
            return i
    return None


def _blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Same heavy spatial low-pass the RefMod Apply node uses for strength < 1."""
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    down = F.adaptive_avg_pool3d(z.float(), (t, max(1, h // factor), max(1, w // factor)))
    up = F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False)
    return up.to(z.dtype)


def _token_estimate(latent_t: int, latent_h: int, latent_w: int) -> int:
    # DiT patch is 2x2, so one latent frame contributes (h/2)*(w/2) rows
    return int(latent_t) * ((int(latent_h) + 1) // 2) * ((int(latent_w) + 1) // 2)


def _build_block(path: str, strength: float, max_frames: int) -> Tuple[Dict, Dict]:
    with safe_open(path, framework="pt") as f:
        raw_meta = f.metadata() or {}
        if META_KEY not in raw_meta:
            raise ValueError(
                f"{os.path.basename(path)} 不是 RefMod：文件里没有 refmod_meta 元数据。\n"
                f"  它看起来是个普通 LoRA。请换一个 RefMod 文件；"
                f"如果确定它就是 RefMod，说明上传/转存时元数据被重写了。"
            )
        meta = json.loads(raw_meta[META_KEY])
        keys = _latent_keys(list(f.keys()))
        members = meta.get("members") or []
        index = _visual_member_index(members)
        if index is None or index >= len(keys):
            raise ValueError("This RefMod has no visual member that can be injected.")
        info = members[index]
        latent = f.get_tensor(keys[index])

    if latent.ndim != 5 or latent.shape[1] != 24:
        raise ValueError(f"Unexpected RefMod latent shape {tuple(latent.shape)}; expected [1,24,T,h,w].")

    frames = latent.shape[2]
    if max_frames > 0 and frames > max_frames:
        latent = latent[:, :, :max_frames]
        frames = max_frames
    latent = latent.float().contiguous()

    if strength < 1.0:
        latent = strength * latent + (1.0 - strength) * _blur_latent(latent)

    latent_h, latent_w = int(latent.shape[3]), int(latent.shape[4])
    if frames == 1:
        block: Dict = {"kind": "image", "latent_h": latent_h,
                       "latent_w": latent_w, "latent": latent}
    else:
        block = {"kind": "video", "latent_h": latent_h, "latent_w": latent_w,
                 "latent_t": int(frames), "latent": latent,
                 "ref_audio_t": 0, "audio_latent": None}
    detail = {
        "name": meta.get("name") or os.path.basename(path),
        "description": info.get("description") or "",
        "latent_t": int(frames),
        "latent_h": latent_h,
        "latent_w": latent_w,
        "tokens": _token_estimate(frames, latent_h, latent_w),
        "strength": float(strength),
    }
    return block, detail


class PROYYRefModInject:
    """Append one RefMod's visual latent to every window's positive conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "windows": ("H3_JOINT_WINDOWS", {"tooltip": "PROYY MiniMax H3 Reference to Window 的 windows 输出。"}),
                "mod": (_list_mods(), {"default": NO_MOD,
                                       "tooltip": "列出 models/loras 与 models/refmods 下全部 .safetensors；"
                                                  "带 refmod_meta 的 RefMod 排在最前。默认留空不选；"
                                                  "空着时本节点会报错，普通 LoRA 选进来也会报错。"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "1.0 原样注入；<1.0 向模糊副本插值（与 RefMod Apply 同一配方）；0 等于跳过。"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": MAX_FRAMES_CAP, "step": 1,
                                       "tooltip": "只注入前 N 个潜帧，0 = 全部。token 成本与帧数线性相关。"}),
                "enabled": ("BOOLEAN", {"default": True,
                                        "tooltip": "关闭时原样透传 windows，不加载任何文件。"}),
            }
        }

    RETURN_TYPES = ("H3_JOINT_WINDOWS", "STRING")
    RETURN_NAMES = ("windows", "report")
    FUNCTION = "inject"
    CATEGORY = "PROYAYA/H3"
    DESCRIPTION = "把 RefMod 的视觉 latent 作为额外参考块注入每个窗口，不改动 PROYAYA 源码。"

    def inject(self, windows, mod, strength=1.0, max_frames=0, enabled=True):
        if not enabled:
            return (windows, "RefMod inject: disabled — windows passed through.")
        if strength <= 0.0:
            return (windows, "RefMod inject: strength=0 — windows passed through.")
        path = _resolve(_require_mod(mod))
        if path is None:
            candidates = _mod_candidates()
            total = len(candidates)
            usable = sum(1 for _d, p in candidates if _has_refmod_meta(p))
            if total and not usable:
                hint = "文件在，但一个都没有 refmod_meta —— 可能被重新保存过，元数据丢了。"
            else:
                hint = "下拉里选一个存在的名字，或确认文件确实已上传。"
            raise ValueError(
                f"RefMod 文件未找到：{mod}\n"
                f"  已扫描 models/loras（含全部子目录与 extra_model_paths 的额外根）"
                f"和 models/refmods：共 {total} 个 .safetensors，其中 {usable} 个带 refmod_meta。\n"
                f"  {hint}"
            )

        block, detail = _build_block(path, float(strength), int(max_frames))
        entries = list(windows or [])
        if not entries:
            raise ValueError("windows 为空，无法注入。")

        injected = 0
        for entry in entries:
            positive = entry.get("positive") or []
            if not positive:
                continue
            patched = []
            for cond in positive:
                tensor, meta = cond[0], dict(cond[1] or {})
                meta["minimax_refs"] = list(meta.get("minimax_refs") or []) + [block]
                patched.append([tensor, meta])
            entry["positive"] = patched
            injected += 1

        lines = [
            "RefMod inject: ok",
            f"  mod: {detail['name']} ({detail['description']})".rstrip(),
            f"  latent: t={detail['latent_t']} h={detail['latent_h']} w={detail['latent_w']}",
            f"  strength: {detail['strength']:.2f}",
            f"  windows patched: {injected}/{len(entries)}",
            f"  tokens added per window: {detail['tokens']}",
            f"  note: positive only; negative conditioning left untouched.",
        ]
        report = "\n".join(lines)
        print("[PROYYRefModInject] " + report.replace("\n", " | "))
        return (entries, report)


def _load_visual_latent(path: str) -> Tuple[Dict, torch.Tensor, Dict]:
    """读 RefMod 的视觉成员 latent，返回 (meta, latent[1,24,T,h,w], 成员信息)。"""
    with safe_open(path, framework="pt") as f:
        raw = f.metadata() or {}
        if META_KEY not in raw:
            raise ValueError(
                f"{os.path.basename(path)} 不是 RefMod：文件里没有 refmod_meta 元数据。\n"
                f"  它看起来是个普通 LoRA。请换一个 RefMod 文件；"
                f"如果确定它就是 RefMod，说明上传/转存时元数据被重写了。"
            )
        meta = json.loads(raw[META_KEY])
        keys = _latent_keys(list(f.keys()))
        members = meta.get("members") or []
        index = _visual_member_index(members)
        if index is None or index >= len(keys):
            raise ValueError("This RefMod has no visual member that can be injected.")
        latent = f.get_tensor(keys[index])
    info = members[index] if index < len(members) else {}
    return meta, latent, info


def _as_image_batch(frame: torch.Tensor) -> torch.Tensor:
    """把 VAE 解码结果统一成 ComfyUI 的 [N,H,W,3]。"""
    if frame.ndim == 4:
        if frame.shape[-1] == 3:
            return frame
        if frame.shape[1] == 3:
            return frame.permute(0, 2, 3, 1)
    if frame.ndim == 5:
        if frame.shape[-1] == 3:
            return frame.reshape(-1, frame.shape[2], frame.shape[3], 3)
        if frame.shape[1] == 3:
            return frame.permute(0, 2, 3, 4, 1).reshape(
                -1, frame.shape[3], frame.shape[4], 3)
    raise ValueError(f"无法把解码结果归一化成 IMAGE 批次：shape={tuple(frame.shape)}")


class PROYYRefModFrames:
    """把 RefMod 的视觉 latent 逐帧解码成图片序列，供 ref_images 使用。

    逐帧解码是刻意的：整段一起 decode 会被视频 VAE 的时序层把相邻帧混合，
    出来是双重曝光的糊图。一帧一次 decode 才拿到干净的静帧。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mod": (_list_mods(), {"default": NO_MOD,
                                       "tooltip": "models/loras 与 models/refmods 下的 RefMod 文件。默认留空不选。"}),
                "vae": ("VAE", {"tooltip": "视频 VAE（与主流程同一个）。逐帧解码，避免时序层混合相邻帧。"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": MAX_FRAMES_CAP, "step": 1,
                                       "tooltip": "只解前 N 帧，0 = 全部。每帧都会占一个 ref_image 槽。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "frames"
    CATEGORY = "PROYAYA/H3"
    DESCRIPTION = ("把 RefMod 的视觉 latent 逐帧解码成图片序列，填进 ref_images 让文本编码器"
                   "看到真实像素。注意 ref_images 每个槽只取 batch 第一张，所以要按帧拆开接。")

    def frames(self, mod, vae, max_frames=0):
        path = _resolve(_require_mod(mod))
        if path is None:
            candidates = _mod_candidates()
            raise ValueError(
                f"RefMod 文件未找到：{mod}\n"
                f"  已扫描 models/loras（含全部子目录与 extra_model_paths 的额外根）"
                f"和 models/refmods：共 {len(candidates)} 个 .safetensors。"
            )
        meta, latent, info = _load_visual_latent(path)
        if latent.ndim != 5 or latent.shape[1] != 24:
            raise ValueError(f"Unexpected RefMod latent shape {tuple(latent.shape)}; "
                             f"expected [1,24,T,h,w].")
        total = int(latent.shape[2])
        count = total if int(max_frames) <= 0 else min(total, int(max_frames))
        latent = latent.float()
        batches = []
        with torch.inference_mode():
            for index in range(count):
                out = vae.decode(latent[:, :, index:index + 1])
                if isinstance(out, dict):
                    out = out.get("samples", out)
                batches.append(_as_image_batch(out).clamp(0.0, 1.0))
        images = torch.cat(batches, dim=0)

        lines = [
            "RefMod frames: ok",
            f"  mod: {meta.get('name') or os.path.basename(path)}"
            + (f" ({info.get('description')})" if info.get("description") else ""),
            f"  latent: t={total} h={int(latent.shape[3])} w={int(latent.shape[4])}",
            f"  decoded frames: {count}",
            f"  image: {int(images.shape[2])}x{int(images.shape[1])} (WxH)",
            "  mode: per-frame decode, no temporal blending",
            "  note: ref_images 每个槽只取 batch 第一张，需要按帧拆开后分别连接。",
        ]
        report = "\n".join(lines)
        print("[PROYYRefModFrames] " + report.replace("\n", " | "))
        return (images, report)


NODE_CLASS_MAPPINGS = {"PROYYRefModInject": PROYYRefModInject,
                       "PROYYRefModFrames": PROYYRefModFrames}
NODE_DISPLAY_NAME_MAPPINGS = {"PROYYRefModInject": "۞▷Refmod Inject◁۞",
                              "PROYYRefModFrames": "۞▷Refmod Frames◁۞"}
