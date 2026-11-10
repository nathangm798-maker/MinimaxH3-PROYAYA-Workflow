"""按块导入（loader.py）的契约 —— 外加「缺 H3 到底丢几个节点」的实测。

这个模块的存在理由只有一个：缩小爆炸半径。9 个节点里真正硬依赖 MiniMax-H3 的
只有 1 个，所以缺 ``comfy_extras.nodes_minimax_h3`` 时应该是 8 → 7，不是 8 → 0。

测试分两层：

- **单元层**：``collect()`` 在块抛异常时的行为 —— 不扩散、不吞 BaseException、
  失败原因能读。这一层不需要 ComfyUI。
- **集成层**：真的把 H3 模块屏蔽掉，导入整个包，数一数还剩几个节点。这是唯一能
  证明「执行了真实的 __init__.py 之后仍然剩 7 个」的办法，所以放在子进程里跑，
  既不受当前进程 sys.modules 影响，也不污染它。
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import loader  # noqa: E402


PACKAGE_DIR = Path(__file__).resolve().parent.parent
CUSTOM_NODES = PACKAGE_DIR.parent
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", r"D:\ComfyUI_windows_portable\ComfyUI"))

# 子进程里先正常导入一次（基线），清掉包模块后把这两个屏蔽成「导入即失败」，
# 再导入一次。两次共用同一个解释器：第二次不再重付 torch 的启动开销。
# 这里敢在同一进程里重复导入，是因为包内没有任何模块在导入期注册路由或写全局
# 状态（唯一的 PromptServer 调用在方法体内）。
#
# 路径靠 sys.path 注入而不是 PYTHONPATH：便携版 python 带了 python*._pth，
# 它会让解释器完全忽略 PYTHONPATH —— 设了也没用，还查不出原因。
_PROBE = r'''
import json
import sys

sys.path[:0] = __PATHS__


def load():
    import PROYAYA
    return PROYAYA


def keys_of(module):
    return sorted(module.NODE_CLASS_MAPPINGS)


baseline = load()
probe = {"base": keys_of(baseline),
         "base_names": len(baseline.NODE_DISPLAY_NAME_MAPPINGS)}

for name in [key for key in sys.modules
             if key == "PROYAYA" or key.startswith("PROYAYA.")]:
    del sys.modules[name]
for name in ("comfy_extras.nodes_minimax_h3", "comfy.text_encoders.minimax"):
    sys.modules[name] = None

degraded = load()
probe["degraded"] = keys_of(degraded)
probe["degraded_names"] = len(degraded.NODE_DISPLAY_NAME_MAPPINGS)

print("RESULT" + json.dumps(probe))
'''


class Collecting(unittest.TestCase):
    """collect() 的行为契约。"""

    def test_all_blocks_merge_in_order(self):
        groups = (("A", lambda: ({"a": 1}, {"a": "A"})),
                  ("B", lambda: ({"b": 2}, {"b": "B"})))
        classes, names, failures = loader.collect(groups)
        self.assertEqual(list(classes), ["a", "b"])
        self.assertEqual(list(names), ["a", "b"])
        self.assertEqual(failures, [])

    def test_broken_block_does_not_spread(self):
        """一个块炸了，后面的块必须照常合并 —— 这就是整个模块的意义。"""

        def boom():
            raise ImportError("No module named 'comfy_extras.nodes_minimax_h3'")

        marker = object()
        groups = (("H3", boom), ("Other", lambda: ({"ok": marker}, {"ok": "OK"})))
        classes, names, failures = loader.collect(groups)
        self.assertEqual(classes, {"ok": marker})
        self.assertEqual(names, {"ok": "OK"})
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].label, "H3")

    def test_failure_reason_names_the_exception_and_target(self):
        def boom():
            raise ImportError("No module named 'comfy_extras.nodes_minimax_h3'")

        _classes, _names, failures = loader.collect((("H3", boom),))
        detail = failures[0].detail
        self.assertIn("ImportError", detail)
        self.assertIn("nodes_minimax_h3", detail)

    def test_earlier_blocks_survive_a_later_failure(self):
        """失败在后面也一样：已经合并进去的不能被回滚。"""

        def boom():
            raise RuntimeError("嗯")

        groups = (("Good", lambda: ({"good": 1}, {})), ("Bad", boom))
        classes, _names, failures = loader.collect(groups)
        self.assertEqual(classes, {"good": 1})
        self.assertEqual([item.label for item in failures], ["Bad"])

    def test_every_failure_is_recorded(self):
        def boom():
            raise ImportError("x")

        classes, _names, failures = loader.collect(
            (("A", boom), ("B", boom), ("C", lambda: ({"c": 1}, {}))))
        self.assertEqual(classes, {"c": 1})
        self.assertEqual([item.label for item in failures], ["A", "B"])

    def test_block_returning_none_is_not_a_failure(self):
        classes, names, failures = loader.collect((("Empty", lambda: None),))
        self.assertEqual((classes, names, failures), ({}, {}, []))

    def test_empty_mappings_are_tolerated(self):
        classes, names, failures = loader.collect((("E", lambda: ({}, {})),))
        self.assertEqual((classes, names, failures), ({}, {}, []))

    def test_later_block_wins_on_duplicate_key(self):
        groups = (("A", lambda: ({"x": 1}, {"x": "first"})),
                  ("B", lambda: ({"x": 2}, {"x": "second"})))
        classes, names, _failures = loader.collect(groups)
        self.assertEqual(classes, {"x": 2})
        self.assertEqual(names, {"x": "second"})

    def test_describe_keeps_type_when_message_is_empty(self):
        """自定义异常常常不带 message，只打 str(error) 会得到一行空白。"""
        self.assertEqual(loader.describe(RuntimeError()), "RuntimeError")
        self.assertEqual(loader.describe(RuntimeError(" 坏了 ")), "RuntimeError: 坏了")

    def test_keyboard_interrupt_is_not_swallowed(self):
        """except Exception 而非 BaseException：Ctrl+C 不能被一个坏节点吞掉。"""

        def interrupt():
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            loader.collect((("X", interrupt),))


@unittest.skipUnless(COMFYUI_ROOT.is_dir(), "需要 ComfyUI 环境")
class DegradesToOneNode(unittest.TestCase):
    """真实执行 __init__.py：屏蔽 H3 之后应该还剩 8 个节点，不是 0 个。"""

    @classmethod
    def setUpClass(cls):
        env = dict(os.environ)
        # 子进程强制 UTF-8：日志里有中文，而这里要断言中文措辞。只影响这个探针
        # 子进程，测试主机本身的编码行为不被掩盖。
        env["PYTHONUTF8"] = "1"
        script = _PROBE.replace(
            "__PATHS__", repr([str(COMFYUI_ROOT), str(CUSTOM_NODES)]))
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=str(COMFYUI_ROOT), env=env, timeout=300,
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        cls.stderr = completed.stderr
        cls.returncode = completed.returncode
        lines = [line for line in completed.stdout.splitlines()
                 if line.startswith("RESULT")]
        if not lines:
            raise AssertionError(
                "探针没有输出结果（exit {}）：\nstdout:\n{}\nstderr:\n{}".format(
                    completed.returncode, completed.stdout[-3000:],
                    completed.stderr[-3000:]))
        cls.result = json.loads(lines[-1][len("RESULT"):])

    def test_probe_exits_cleanly(self):
        self.assertEqual(self.returncode, 0)

    def test_baseline_registers_nine_nodes(self):
        self.assertEqual(len(self.result["base"]), 9, self.result["base"])
        self.assertEqual(self.result["base_names"], 9)

    def test_only_the_h3_node_disappears(self):
        base = set(self.result["base"])
        degraded = set(self.result["degraded"])
        self.assertEqual(base - degraded, {"PROYYMiniMaxH3ReferenceToVideo"})
        self.assertEqual(degraded - base, set())
        self.assertEqual(len(degraded), 8, sorted(degraded))

    def test_display_names_do_not_keep_orphans(self):
        self.assertEqual(self.result["degraded_names"],
                         len(self.result["degraded"]))

    def test_console_explains_what_was_lost(self):
        text = self.stderr
        self.assertIn("导入失败", text)
        self.assertIn("PROYYMiniMaxH3ReferenceToVideo", text)
        self.assertIn("nodes_minimax_h3", text)
        self.assertIn("8 个节点", text)


if __name__ == "__main__":
    unittest.main()
