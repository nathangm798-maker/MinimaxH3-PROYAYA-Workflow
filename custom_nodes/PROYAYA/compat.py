"""启动自检：缺什么、需要哪个版本，一次说清。

为什么要有它：节点是在 ``__init__.py`` 里**立即导入**的，导入失败 ComfyUI 只留
一句 ``IMPORT FAILED``，用户分不清是「ComfyUI 太旧」「装漏了包」还是「文件没拷全」。

``loader.collect()`` 已经把导入拆成互不牵连的块，一块坏掉只丢它自己的节点；但那
只是「不让损失扩散」，损失本身还需要被解释清楚 —— 这里在导入之前先探一遍环境，
把问题连同最低版本要求列出来，再由 ``summary()`` 把「哪几个节点因此没注册」接上。

这里在导入节点之前先跑一遍，把问题连同最低版本要求列出来。它是纯探测：
不装东西、不改配置；自身出错也只写 DEBUG 日志，绝不影响节点加载。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import List, NamedTuple, Optional, Tuple

LOGGER = logging.getLogger("PROYAYA")

# MiniMax-H3 支持在 ComfyUI v0.30.0（PR #15224）落地，这是本包绕不开的下限：
# PROYYMiniMaxH3ReferenceToVideo / PROYYMaskedSampler 都直接吃 H3 的 NestedTensor。
MIN_COMFYUI = "0.30.0"

# (模块名, 用途, 需要的最低 ComfyUI 版本或 None)
MODULES: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("comfy_extras.nodes_minimax_h3", "MiniMax-H3 参考生视频节点", "0.30.0"),
    ("comfy.text_encoders.minimax", "MiniMax-H3 文本编码器", "0.30.0"),
    ("comfy.nested_tensor", "视频/音频联合 latent", None),
    ("comfy_api.latest", "V3 节点 schema（io.*）", None),
    ("comfy_execution.graph_utils", "ExecutionBlocker（preview_only 拦截）", None),
    ("server", "PromptServer 进度回报", None),
    ("node_helpers", "条件里注入 minimax_refs", None),
    ("safetensors", "读取 RefMod 元数据", None),
    ("av", "按窗口解码视频", None),
)

# (模块名, 属性路径, 用途) —— 属性被改名同样是硬伤，但报错信息比缺模块更难懂
ATTRS: Tuple[Tuple[str, str, str], ...] = (
    ("comfy_api.latest", "io.Schema", "V3 schema"),
    ("comfy_api.latest", "io.Autogrow", "动态参数组"),
    ("comfy_api.latest", "io.UploadType.video", "组合框内上传视频"),
    ("comfy_api.latest", "io.Video", "VIDEO 插座类型"),
    ("comfy_api.latest", "VideoInput", "VIDEO 输入的抽象基类"),
    ("comfy_api.latest", "InputImpl.VideoFromFile", "从文件读视频"),
    ("comfy_api.latest", "InputImpl.VideoFromComponents", "从帧+音频组装视频"),
    ("comfy_api.latest", "Types.VideoCodec", "视频编码枚举"),
    ("comfy.model_management", "unload_model_and_clones", "采样后精准卸载主模型"),
    ("folder_paths", "filter_files_content_types", "筛选 input 目录里的视频"),
    ("node_helpers", "conditioning_set_values", "注入 minimax_refs"),
    ("nodes", "ImageScale.upscale_methods", "复用原生放大算法表"),
    ("nodes", "ImageScale.crop_methods", "复用原生裁剪模式表"),
)

# 缺了只影响单个节点，不影响其它 7 个
OPTIONAL: Tuple[Tuple[str, str], ...] = (
    ("ultralytics", "YOLO Mask 的模型运行时"),
)


class Report(NamedTuple):
    version: str
    missing: List[str]
    missing_optional: List[str]
    too_old: bool


def _parse(text) -> Tuple[int, ...]:
    """把 ``v0.30.1`` 变成 ``(0, 30, 1)``。

    必须按数字比，不能按字符串比 —— ``"0.9" < "0.30"`` 在字符串意义上是成立的。
    解析不出来的部分直接截断（``0.30.0rc1`` -> ``(0, 30, 0)``）。
    """
    if not text:
        return ()
    parts: List[int] = []
    for chunk in str(text).strip().lstrip("vV").split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def comfyui_version() -> str:
    """读 ComfyUI 版本。拿不到就返回空串，绝不抛异常。"""
    try:
        import comfyui_version

        value = getattr(comfyui_version, "__version__", "")
        if value:
            return str(value)
    except Exception:
        pass
    return ""


def _module_problem(name: str) -> Optional[str]:
    """返回 ``None`` 表示可用，否则返回一句人话说明哪里不对。

    只用 ``find_spec`` 是不够的：它回答的是「文件在不在」，不是「能不能导入」。
    模块文件在、但它自己的依赖坏了（ComfyUI 升级后很常见），find_spec 照样返回
    spec，真导入时才炸 —— 那时用户看到的又只会是一句 IMPORT FAILED。所以这里
    先 find_spec 快速排除「根本没有」，再真导入一次把「有但坏的」也抓出来。
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is None:
        return "模块不存在"
    try:
        importlib.import_module(name)
    except Exception as error:
        return "导入时报错：" + type(error).__name__ + ": " + str(error)
    return None


def _attr_missing(module_name: str, dotted: str) -> bool:
    try:
        target = importlib.import_module(module_name)
    except Exception:
        return True
    for part in dotted.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            return True
    return False


def check() -> Report:
    """探测当前环境。任何意外都降级成「没有问题」，不干扰节点加载。"""
    try:
        version = comfyui_version()
        missing: List[str] = []
        for name, why, need in MODULES:
            problem = _module_problem(name)
            if problem is not None:
                hint = "，需要 ComfyUI >= " + need if need else ""
                missing.append(
                    "缺少模块 " + name + "（" + why + hint + "）：" + problem)
        for module_name, dotted, why in ATTRS:
            if _attr_missing(module_name, dotted):
                missing.append(
                    "缺少 " + module_name + "." + dotted + "（" + why + "）")
        optional = [name + "（" + why + "）" for name, why in OPTIONAL
                    if _module_problem(name) is not None]
        need = _parse(MIN_COMFYUI)
        too_old = bool(version) and bool(need) and _parse(version) < need
        return Report(version, missing, optional, too_old)
    except Exception as error:                       # pragma: no cover - 兜底
        LOGGER.debug("PROYAYA 兼容性自检本身出错：%s", error, exc_info=True)
        return Report("", [], [], False)


def report(state: Optional[Report]) -> None:
    """只说致命的。可选依赖不影响加载，交给 summary()，避免同一件事报两遍。"""
    if state is None:
        return
    version = state.version or "未知"
    if state.too_old:
        LOGGER.error(
            "[PROYAYA] ComfyUI %s 版本过低，本包需要 >= %s"
            "（MiniMax-H3 支持是 %s 才加入的）。请升级 ComfyUI 后重启。",
            version, MIN_COMFYUI, MIN_COMFYUI)
    if state.missing:
        LOGGER.error(
            "[PROYAYA] 依赖缺失，相关节点将无法注册：\n  · %s\n"
            "  修法：ComfyUI 版本不够就升级本体；模块本身是 pip 包的就补装。",
            "\n  · ".join(state.missing))


def summary(state: Optional[Report], version: str, node_count: int,
            failures=None) -> None:
    """加载成功后的一行摘要 —— 排障时第一时间要看的就是它。

    ``failures`` 来自 ``loader.collect()``，是「真的导不进来」的现场。它和
    ``state.missing`` 不重复：missing 说的是**原因**（环境里缺哪个模块），
    failures 说的是**后果**（于是哪个节点没注册上）。排障时先看 missing，
    再看这里，就知道这次损失有多大。
    """
    if state is None:
        return
    comfy = state.version or "未知"
    LOGGER.info("[PROYAYA] v%s 已加载 %d 个节点（ComfyUI %s）。",
                version, node_count, comfy)
    if failures:
        LOGGER.warning(
            "[PROYAYA] 以下 %d 个节点导入失败，未注册：\n  · %s\n"
            "  其余 %d 个节点照常加载。",
            len(failures),
            "\n  · ".join(item.label + "：" + item.detail for item in failures),
            node_count)
    if state.missing_optional:
        LOGGER.warning("[PROYAYA] YOLO Mask 不可用：缺 %s",
                       "、".join(name.split("（")[0] for name in state.missing_optional))
