"""PROYAYA 启动自检（compat.py）的契约。

这个模块的价值全在「报错信息对不对」上：它本身不改任何东西，唯一的产出就是
几行日志。所以测试的重点不是「有没有检查到」，而是

- 版本比较必须按数字比（``"0.9" < "0.30"`` 在字符串意义上是成立的，那是错的）；
- 缺东西时要说出**是什么**和**需要哪个版本**，否则用户还是看不懂；
- 自检自己出错不能影响节点加载（一律降级成「没问题」）。

不 import 节点包、不加载 torch/CUDA；只有 check() 的集成用例需要 ComfyUI，
环境不具备时自动跳过。
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compat  # noqa: E402
import loader  # noqa: E402


class VersionParsing(unittest.TestCase):
    def test_plain_version(self):
        self.assertEqual(compat._parse("0.30.0"), (0, 30, 0))

    def test_leading_v(self):
        self.assertEqual(compat._parse("v1.2.3"), (1, 2, 3))

    def test_empty(self):
        self.assertEqual(compat._parse(""), ())
        self.assertEqual(compat._parse(None), ())

    def test_suffix_is_truncated(self):
        self.assertEqual(compat._parse("0.30.0rc1"), (0, 30, 0))

    def test_compares_numerically_not_lexically(self):
        """0.9 是旧版本，0.30 是新版本 —— 字符串比较会得到相反结论。"""
        self.assertLess(compat._parse("0.9"), compat._parse("0.30"))
        self.assertLess(compat._parse("0.29.9"), compat._parse(compat.MIN_COMFYUI))
        self.assertGreater(compat._parse("0.39.0"), compat._parse(compat.MIN_COMFYUI))

    def test_floor_is_h3_release(self):
        self.assertEqual(compat.MIN_COMFYUI, "0.30.0")


class ManifestShape(unittest.TestCase):
    def test_h3_modules_carry_the_floor(self):
        floors = {name: need for name, _why, need in compat.MODULES}
        for name in ("comfy_extras.nodes_minimax_h3", "comfy.text_encoders.minimax"):
            self.assertIn(name, floors)
            self.assertEqual(floors[name], compat.MIN_COMFYUI)

    def test_attr_entries_are_well_formed(self):
        self.assertTrue(compat.ATTRS)
        for entry in compat.ATTRS:
            self.assertEqual(len(entry), 3)
            self.assertTrue(all(isinstance(part, str) and part for part in entry))

    def test_no_duplicate_declarations(self):
        names = [name for name, _why, _need in compat.MODULES]
        self.assertEqual(len(names), len(set(names)))
        attrs = [(module, dotted) for module, dotted, _why in compat.ATTRS]
        self.assertEqual(len(attrs), len(set(attrs)))


class MissingDetection(unittest.TestCase):
    def test_absent_module_is_reported(self):
        problem = compat._module_problem("proyaya_definitely_not_a_module")
        self.assertIsNotNone(problem)
        self.assertIn("不存在", problem)

    def test_present_module_is_clean(self):
        self.assertIsNone(compat._module_problem("json"))

    def test_broken_module_is_reported_as_import_error(self):
        """文件在、但导入失败 —— find_spec 看不出来，必须真导入一次才发现。"""
        sys.modules["proyaya_fake_broken"] = None
        self.addCleanup(sys.modules.pop, "proyaya_fake_broken", None)
        problem = compat._module_problem("proyaya_fake_broken")
        self.assertIsNotNone(problem)


class Reporting(unittest.TestCase):
    def _report(self, **kwargs):
        base = dict(version="0.24.3", missing=[], missing_optional=[], too_old=False)
        base.update(kwargs)
        return compat.Report(**base)

    def test_silent_when_healthy(self):
        """一切正常时 report() 不该说话 —— 正常启动只留 summary 那一行。"""
        logger = compat.LOGGER
        records = []
        handler = type("H", (object,), {"emit": lambda _self, r: records.append(r),
                                        "level": 0})()
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        compat.report(self._report(version="0.39.0"))
        self.assertEqual(records, [])

    def test_too_old_names_the_floor(self):
        with self.assertLogs("PROYAYA", level="ERROR") as captured:
            compat.report(self._report(too_old=True))
        text = "\n".join(captured.output)
        self.assertIn("版本过低", text)
        self.assertIn(compat.MIN_COMFYUI, text)
        self.assertIn("0.24.3", text)

    def test_missing_module_is_not_swallowed(self):
        with self.assertLogs("PROYAYA", level="ERROR") as captured:
            compat.report(self._report(missing=["缺少模块 xxx（演示）：模块不存在"]))
        text = "\n".join(captured.output)
        # 按块导入之后不能再写「所有节点都消失」：缺一个 H3 模块只掉 1 个节点。
        self.assertIn("将无法注册", text)
        self.assertNotIn("所有节点", text)
        self.assertIn("xxx", text)

    def test_summary_names_the_nodes_that_failed(self):
        """missing 说原因，summary 说后果 —— 两条要接得上。"""
        failure = loader.Failure(
            "Reference to Window (PROYYMiniMaxH3ReferenceToVideo)",
            "ImportError: No module named 'comfy_extras.nodes_minimax_h3'")
        with self.assertLogs("PROYAYA", level="WARNING") as captured:
            compat.summary(self._report(version="0.39.0"), "3.2.0", 8, [failure])
        text = "\n".join(captured.output)
        self.assertIn("1 个节点导入失败", text)
        self.assertIn("PROYYMiniMaxH3ReferenceToVideo", text)
        self.assertIn("其余 8 个节点照常加载", text)

    def test_summary_is_quiet_about_failures_when_there_are_none(self):
        with self.assertLogs("PROYAYA", level="INFO") as captured:
            compat.summary(self._report(version="0.39.0"), "3.2.0", 9, [])
        self.assertNotIn("导入失败", "\n".join(captured.output))

    def test_summary_counts_nodes(self):
        with self.assertLogs("PROYAYA", level="INFO") as captured:
            compat.summary(self._report(version="0.39.0"), "3.2.0", 9)
        text = "\n".join(captured.output)
        self.assertIn("v3.2.0", text)
        self.assertIn("9 个节点", text)
        self.assertIn("0.39.0", text)

    def test_summary_flags_yolo_only(self):
        with self.assertLogs("PROYAYA", level="WARNING") as captured:
            compat.summary(self._report(missing_optional=["ultralytics（模型运行时）"]),
                           "3.2.0", 9)
        text = "\n".join(captured.output)
        self.assertIn("YOLO Mask 不可用", text)
        self.assertIn("ultralytics", text)

    def test_none_state_never_raises(self):
        compat.report(None)
        compat.summary(None, "3.1.0", 0)


@unittest.skipUnless(importlib.util.find_spec("comfyui_version") is not None,
                     "需要 ComfyUI 环境")
class LiveEnvironment(unittest.TestCase):
    def test_current_install_is_clean(self):
        state = compat.check()
        self.assertEqual(state.missing, [], "\n".join(state.missing))
        self.assertFalse(state.too_old, f"ComfyUI {state.version} 低于 {compat.MIN_COMFYUI}")


if __name__ == "__main__":
    unittest.main()
