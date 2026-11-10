"""PROYAYA: H3 long-video window nodes.

一个包，一个注册表。

历史：这里本来拆成三个独立包 —— PROYAYA / ComfyUI-PROYAYA-MaskedSampler /
ComfyUI-PROYAYA-RefModInject。但拆分本身是漏的：MaskedSampler 需要用
sys.path 兜底才能 `import PROYAYA` 拿到 runtime 和 rolling。合并之后包内
一律走相对导入，不再有跨包路径补丁。

已移除 PROYYBasicScheduler：调度/步数用官方 BasicScheduler，精确起始 sigma
用官方 ManualSigmas（直接吃 sigma 列表），不需要第三个做同样事的节点。

已移除 PROYYSamplerCustomAdvanced：它的功能被 PROYYMaskedSampler 完全覆盖 ——
后者本来就是「同一个 CPU 窗口 guider + 可选遮罩」，不接遮罩时走 PROYAYA 原生
sample 路径，等于前者的等价实现。两个节点同时存在只是给同一个采样行为提供两套
接口，容易出现「改了 A 忘了 B」。

已移除 PROYAYA_LumaMatch（亮度匹配）：它做的是「按背景区四分位做仿射增益/偏置，
再双向 EMA 平滑」的后处理，属于单帧就能算完的独立环节，不必留在这个包里。

导入按块进行（见 loader.py）：每块只碰自己的文件，一块坏掉只丢它自己的节点。
"""

__version__ = "3.3.0"
WEB_DIRECTORY = "./web"

from . import compat as _compat
from .loader import collect as _collect

# 先自检、再导节点。整包是「立即导入」的：任何一环缺失都会让节点注册不上，
# 而 ComfyUI 只留一句 IMPORT FAILED。提前把「缺什么、需要哪个版本」打到控制台，
# 比让用户去翻 traceback 强。自检自身出错只会静默降级，不影响加载。
_COMPAT = _compat.check()
_compat.report(_COMPAT)

# 预览取帧路由（GET /proyaya/frame）：浏览器解不了 HEVC 时前端退回静帧。
# 注册失败只丢这个兜底，不影响任何节点，所以单独 try。
try:
    from . import preview_frame as _preview_frame  # noqa: F401
except Exception as _exc:  # noqa: BLE001
    print("[PROYAYA] 预览取帧路由未注册：%s" % _exc)


# ---- 导入块 ----------------------------------------------------------------
# 每个函数就是「一个节点（或一组同文件的节点）」，返回
# (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)。
#
# 唯一硬依赖 MiniMax-H3 的是 _ref_to_video：它经 conditioning_weights /
# sized_references 直接 import comfy_extras.nodes_minimax_h3。其余 8 个节点在
# 老版本 ComfyUI 上本来都能跑，不该陪它一起消失。
#
# 注意 native_video 同时被 _ref_to_video 和 _resize_video 引用 —— 它坏了两个
# 都会报失败，这是对的：它们确实都缺同一个东西。


def _ref_to_video():
    """H3 参考生视频。链上带 comfy_extras.nodes_minimax_h3（>= 0.30.0）。"""
    from .encode_nodes import PROYYMiniMaxH3ReferenceToVideo

    return ({"PROYYMiniMaxH3ReferenceToVideo": PROYYMiniMaxH3ReferenceToVideo},
            {"PROYYMiniMaxH3ReferenceToVideo": "۞▷Reference to Window◁۞"})


def _resize_video():
    """视频缩放裁切 + 自绘预览控件条。只依赖 native_video / comfy_api。"""
    from .video_resize import PROYYImageScale

    return ({"PROYYImageScale": PROYYImageScale},
            {"PROYYImageScale": "۞▷Resize Video◁۞"})


def _control_video_check():
    """Validate ControlNet frame rate and minimum frame coverage."""
    from .control_video_check import PROYAYA_ControlVideoCheck

    return ({"PROYAYA_ControlVideoCheck": PROYAYA_ControlVideoCheck},
            {"PROYAYA_ControlVideoCheck": "۞▷Control Video Check◁۞"})


def _preview_any():
    """文本预览。不依赖任何一个 comfy 包。"""
    from .preview_nodes import PROYYPreviewAny

    return ({"PROYYPreviewAny": PROYYPreviewAny},
            {"PROYYPreviewAny": "۞▷Preview Any◁۞"})


def _latent_upscale():
    """latent 分块放大。依赖 comfy.nested_tensor 与 V3 schema。"""
    from .tiled_upscale import PROYYTiledLatentUpscale

    return ({"PROYYTiledLatentUpscale": PROYYTiledLatentUpscale},
            {"PROYYTiledLatentUpscale": "۞▷Latent Upscale◁۞"})


def _stream_decode():
    """流式 VAE 解码：不预分配全量像素，长视频不 OOM。"""
    from .stream_decode import PROYYStreamVAEDecode

    return ({"PROYYStreamVAEDecode": PROYYStreamVAEDecode},
            {"PROYYStreamVAEDecode": "۞▷Stream VAE Decode◁۞"})


def _masked_sampler():
    """原 ComfyUI-PROYAYA-MaskedSampler 包。走 runtime / rolling，不碰 H3。"""
    from .masked_sampler import NODE_CLASS_MAPPINGS as classes
    from .masked_sampler import NODE_DISPLAY_NAME_MAPPINGS as names

    return (dict(classes), dict(names))


def _yolo_mask():
    """人像遮罩。ultralytics 是延迟导入的，装不装只影响这一个节点。"""
    from .yolo_person_mask import PROYAYA_YoloPersonMask

    return ({"PROYAYA_YoloPersonMask": PROYAYA_YoloPersonMask},
            {"PROYAYA_YoloPersonMask": "۞▷YOLO Mask◁۞"})


def _refmod():
    """原 ComfyUI-PROYAYA-RefModInject 包。只依赖 safetensors。"""
    from .refmod_inject import NODE_CLASS_MAPPINGS as classes
    from .refmod_inject import NODE_DISPLAY_NAME_MAPPINGS as names

    return (dict(classes), dict(names))


# 标签写「人话 + 节点类名」：失败时会原样进控制台日志，两种都能被搜到。
_BLOCKS = (
    ("Reference to Window (PROYYMiniMaxH3ReferenceToVideo)", _ref_to_video),
    ("Resize Video (PROYYImageScale)", _resize_video),
    ("Control Video Check (PROYAYA_ControlVideoCheck)", _control_video_check),
    ("Preview Any (PROYYPreviewAny)", _preview_any),
    ("Latent Upscale (PROYYTiledLatentUpscale)", _latent_upscale),
    ("Stream VAE Decode (PROYYStreamVAEDecode)", _stream_decode),
    ("Window Sampler (PROYYMaskedSampler)", _masked_sampler),
    ("YOLO Mask (PROYAYA_YoloPersonMask)", _yolo_mask),
    ("Refmod Inject / Refmod Frames (PROYYRefModInject, PROYYRefModFrames)", _refmod),
)

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, _FAILURES = _collect(_BLOCKS)

# 最后再打一行摘要：正常启动时控制台应该能看到这句，排障时第一时间看它。
# 有块导入失败时，summary() 会紧跟一条 WARNING 说明丢了哪些节点、为什么。
_compat.summary(_COMPAT, __version__, len(NODE_CLASS_MAPPINGS), _FAILURES)
