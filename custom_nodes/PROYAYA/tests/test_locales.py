"""PROYAYA 语言包（locales/*/nodeDefs.json）的结构契约。

ComfyUI 前端查表规则是 ``nodeDefs.<节点类型>.inputs.<参数名>.name``,键写错了
不会报错,只会静默回落到 Python 里写死的显示名 —— 表现就是「切了语言没反应」。
所以这里把最容易写错的三类问题钉住:

- 参数名拼错(键对不上真实输入名);
- 两种语言漏翻某一条(en 有 zh 没有,切过去出现中英混排);
- 英文档直接复制中文档(看起来有翻译,实际没生效)。

不 import 节点包,不加载 torch/CUDA,只读 JSON。
"""

import json
import unittest
from pathlib import Path

LOCALES = Path(__file__).resolve().parent.parent / "locales"
REQUIRED_LANGS = ("en", "zh")


def load(lang):
    path = LOCALES / lang / "nodeDefs.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class LocaleFilesExist(unittest.TestCase):
    def test_required_languages_shipped(self):
        for lang in REQUIRED_LANGS:
            self.assertTrue(
                (LOCALES / lang / "nodeDefs.json").is_file(),
                f"缺少 locales/{lang}/nodeDefs.json")

    def test_every_language_folder_is_parseable(self):
        for folder in sorted(LOCALES.iterdir()):
            if not folder.is_dir():
                continue
            path = folder / "nodeDefs.json"
            if not path.is_file():
                continue
            with self.subTest(lang=folder.name):
                with path.open("r", encoding="utf-8") as f:
                    self.assertIsInstance(json.load(f), dict)


class KeySymmetry(unittest.TestCase):
    def setUp(self):
        self.zh = load("zh")
        self.en = load("en")

    def test_same_node_set(self):
        self.assertEqual(sorted(self.zh), sorted(self.en))

    def test_same_input_set_per_node(self):
        for node in self.zh:
            with self.subTest(node=node):
                self.assertEqual(
                    sorted(self.zh[node].get("inputs", {})),
                    sorted(self.en[node].get("inputs", {})))

    def test_same_output_set_per_node(self):
        for node in self.zh:
            with self.subTest(node=node):
                self.assertEqual(
                    sorted(self.zh[node].get("outputs", {})),
                    sorted(self.en[node].get("outputs", {})))


class EntriesAreComplete(unittest.TestCase):
    def test_display_name_present(self):
        for lang in REQUIRED_LANGS:
            for node, spec in load(lang).items():
                with self.subTest(lang=lang, node=node):
                    self.assertTrue(str(spec.get("display_name", "")).strip())

    def test_every_input_has_name(self):
        for lang in REQUIRED_LANGS:
            for node, spec in load(lang).items():
                for key, entry in spec.get("inputs", {}).items():
                    with self.subTest(lang=lang, node=node, input=key):
                        self.assertTrue(str(entry.get("name", "")).strip())

    def test_output_indices_are_numeric(self):
        for lang in REQUIRED_LANGS:
            for node, spec in load(lang).items():
                for key in spec.get("outputs", {}):
                    with self.subTest(lang=lang, node=node, output=key):
                        self.assertTrue(key.isdigit())

    def test_node_types_use_pack_prefix(self):
        for lang in REQUIRED_LANGS:
            for node in load(lang):
                with self.subTest(lang=lang, node=node):
                    self.assertTrue(
                        node.startswith("PROYY") or node.startswith("PROYAYA_"))


class TranslationsActuallyDiffer(unittest.TestCase):
    def test_chinese_and_english_are_not_copies(self):
        zh, en = load("zh"), load("en")
        identical = [n for n in zh if zh[n] == en[n]]
        self.assertEqual(identical, [], f"这些节点两种语言完全一样: {identical}")

    def test_chinese_file_has_cjk(self):
        """显示名现在是中英共用的品牌名（۞▷X◁۞），中文覆盖率由参数名承担。"""
        def has_cjk(text):
            return any("\u4e00" <= ch <= "\u9fff" for ch in text)

        for node, spec in load("zh").items():
            with self.subTest(node=node):
                names = [e.get("name", "") for e in spec.get("inputs", {}).values()]
                names += [e.get("name", "") for e in spec.get("outputs", {}).values()]
                self.assertTrue(any(has_cjk(name) for name in names),
                                f"{node} 的中文档里没有任何中文字段名")

    def test_display_name_is_language_neutral(self):
        """品牌名不随语言切换：zh 与 en 的 display_name 必须逐字相同。"""
        zh, en = load("zh"), load("en")
        for node in zh:
            with self.subTest(node=node):
                self.assertEqual(zh[node]["display_name"], en[node]["display_name"])
                self.assertIn("\u06de", zh[node]["display_name"],
                              f"{node} 的显示名没套品牌格式")

    def test_english_file_has_no_cjk(self):
        def has_cjk(text):
            return any("\u4e00" <= ch <= "\u9fff" for ch in text)

        for node, spec in load("en").items():
            texts = [spec["display_name"]]
            texts += [e["name"] for e in spec.get("inputs", {}).values()]
            texts += [e.get("name", "") for e in spec.get("outputs", {}).values()]
            for text in texts:
                with self.subTest(node=node, text=text):
                    self.assertFalse(has_cjk(text), f"英文档里残留中文: {text!r}")


if __name__ == "__main__":
    unittest.main()
