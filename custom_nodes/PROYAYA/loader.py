"""按块导入：一块坏掉，只丢它自己的节点，不牵连整包。

为什么要有它：``__init__.py`` 里的导入块彼此独立 —— 每块只碰自己的文件，
块与块之间没有 import 关系。但原来的写法是几条平铺的 ``from .x import Y``：
第 3 条抛异常，后面所有节点跟着一起消失，而 ComfyUI 只留一句 ``IMPORT FAILED``。

实测的代价：在 ComfyUI 0.24 上缺 ``comfy_extras.nodes_minimax_h3``，原有 H3 节点
注册不上，而这 8 个里真正依赖 H3 的只有 1 个。

这里只做一件事：把「导入一组节点」变成「失败也不许扩散」的操作，并把失败原因
收集起来交给 ``compat`` 去打日志。不装东西、不改配置、不碰 sys.path。

注意 ``except Exception`` 而不是 ``BaseException``：``SystemExit`` / ``KeyboardInterrupt``
必须照旧穿出去，否则 Ctrl+C 会被一个坏的节点文件吞掉。
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Tuple


class Failure(NamedTuple):
    """一个导入块失败的现场。

    ``label`` 是「丢的是哪个节点」（人话 + 节点类名，方便在控制台里 grep），
    ``detail`` 是「为什么没进来」（异常类型 + 说明）。
    """

    label: str
    detail: str


def describe(error: BaseException) -> str:
    """把异常压成一行，直接能打进日志。

    ``str(error)`` 有时候是空的（自定义异常不传 message），所以类型名不能省。
    """
    text = str(error).strip()
    name = type(error).__name__
    return name + ": " + text if text else name


# 每个导入函数返回 (class_mappings, display_name_mappings)。
Block = Callable[[], Tuple[Dict[str, object], Dict[str, str]]]


def collect(groups) -> Tuple[Dict[str, object], Dict[str, str], List[Failure]]:
    """依次执行导入块，返回 ``(classes, display_names, failures)``。

    任何一块抛异常都只记进 ``failures``，其余块照常合并 —— 这是本模块存在的
    全部理由。块返回 ``None`` 会被当成「这一块没有节点」，不算失败。
    """
    classes: Dict[str, object] = {}
    names: Dict[str, str] = {}
    failures: List[Failure] = []

    for label, opener in groups:
        try:
            produced = opener()
        except Exception as error:                   # noqa: BLE001 - 见模块注释
            failures.append(Failure(label, describe(error)))
            continue
        if not produced:
            continue
        block_classes, block_names = produced
        classes.update(block_classes or {})
        names.update(block_names or {})

    return classes, names, failures
