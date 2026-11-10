"""Reference-encoding cache contracts.

The cache exists so that re-running the node with a new ``length`` or a new
prompt does not redo VAE work whose inputs did not change. These tests pin the
three properties that makes it safe: identity keys (not content hashes), byte
bounded LRU eviction, and one encode per unchanged material.
"""

import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime as base_tests

torch = base_tests.torch
sized = importlib.import_module(base_tests.package_name + ".sized_references")
ref_cache = importlib.import_module(base_tests.package_name + ".ref_cache")


class IdentityTests(unittest.TestCase):
    def test_same_object_hashes_equal(self):
        marker = torch.zeros(2, 2)
        self.assertEqual(ref_cache.Identity(marker), ref_cache.Identity(marker))
        self.assertEqual(hash(ref_cache.Identity(marker)), hash(ref_cache.Identity(marker)))

    def test_different_object_is_not_equal(self):
        first, second = torch.zeros(2, 2), torch.zeros(2, 2)
        self.assertNotEqual(ref_cache.Identity(first), ref_cache.Identity(second))

    def test_identity_keeps_source_alive(self):
        # A cached entry must not outlive its source into a recycled id.
        marker = torch.zeros(2, 2)
        key = ref_cache.Identity(marker)
        reference = __import__("weakref").ref(marker)
        del marker
        self.assertIsNotNone(reference())
        self.assertIs(reference(), key.value)
        del key
        self.assertIsNone(reference())

    def test_model_identity_does_not_pin_the_model(self):
        import weakref

        class Model:
            pass

        model = Model()
        key = ref_cache.model_identity(model)
        reference = weakref.ref(model)
        del model
        self.assertIsNone(reference())
        self.assertEqual(key[0], "model")

    def test_model_identity_is_stable_per_object(self):
        class Model:
            pass

        model = Model()
        self.assertEqual(ref_cache.model_identity(model), ref_cache.model_identity(model))
        self.assertNotEqual(ref_cache.model_identity(model),
                            ref_cache.model_identity(Model()))


class PayloadBytesTests(unittest.TestCase):
    def test_counts_nested_tensors(self):
        value = [torch.zeros(4, dtype=torch.float32), {"a": torch.zeros(2, dtype=torch.float32)}]
        self.assertEqual(ref_cache.payload_bytes(value), 16 + 8)

    def test_ignores_plain_values(self):
        self.assertEqual(ref_cache.payload_bytes({"n": 3, "s": "x"}), 0)


class RefCacheTests(unittest.TestCase):
    def test_hit_and_miss_are_counted_per_level(self):
        cache = ref_cache.RefCache(1 << 20)
        self.assertIs(cache.get(("a",), "l1"), ref_cache.MISSING)
        cache.put(("a",), torch.zeros(4))
        self.assertEqual(cache.stats["l1_miss"], 1)
        self.assertEqual(torch.count_nonzero(cache.get(("a",), "l1")), 0)
        self.assertEqual(cache.stats["l1_hit"], 1)

    def test_eviction_is_least_recently_used(self):
        cache = ref_cache.RefCache(64)
        cache.put(("a",), torch.zeros(8, dtype=torch.float32))   # 32 bytes
        cache.put(("b",), torch.zeros(8, dtype=torch.float32))   # 32 bytes
        cache.get(("a",), "l1")                                  # a becomes newest
        cache.put(("c",), torch.zeros(8, dtype=torch.float32))   # evicts b
        self.assertIsNot(cache.get(("a",), "l1"), ref_cache.MISSING)
        self.assertIs(cache.get(("b",), "l1"), ref_cache.MISSING)
        self.assertIsNot(cache.get(("c",), "l1"), ref_cache.MISSING)

    def test_oversized_value_is_refused_not_stored(self):
        cache = ref_cache.RefCache(8)
        self.assertFalse(cache.put(("big",), torch.zeros(8, dtype=torch.float32)))
        self.assertEqual(len(cache), 0)

    def test_limit_shrink_evicts(self):
        cache = ref_cache.RefCache(1 << 20)
        for index in range(4):
            cache.put((index,), torch.zeros(8, dtype=torch.float32))
        cache.set_limit(40)
        self.assertLessEqual(cache.bytes_used, 40)
        self.assertEqual(cache.evictions, 3)

    def test_summary_is_empty_until_asked(self):
        self.assertEqual(ref_cache.RefCache(1024).summary(), "")
        cache = ref_cache.RefCache(1024)
        cache.get(("a",), "l2")
        self.assertIn("L2 0 命中 / 1 未命中", cache.summary())

    def test_configure_zero_disables(self):
        self.assertIsNone(ref_cache.configure(0))
        self.assertIs(ref_cache.configure(512), ref_cache.shared())
        self.assertEqual(ref_cache.shared().limit, 512 * 1024 * 1024)


class FakeVae:
    def __init__(self):
        self.calls = 0

    def encode(self, frames):
        self.calls += 1
        return torch.zeros(1, 24, 1, 4, 4)


class FakeClip:
    def __init__(self):
        self.batches = []

    def tokenize(self, prompt, **kwargs):
        self.batches.append((prompt, kwargs.get("minimax_ref_items")))
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {}]]


def encode(image, cache, provenance=None, width=64, height=64, length=124):
    return sized.encode_sized_references(
        clip=FakeClip(), prompt="a test prompt", width=width, height=height, length=length,
        vae=FakeVae(), ref_images={"ref_image_0": image}, cache=cache, provenance=provenance)


class EncoderCacheTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

    def test_disabled_cache_encodes_every_call(self):
        first = encode(self.image, None)
        second = encode(self.image, None)
        self.assertEqual(first.result[1]["samples"].unbind()[0].shape[2],
                         second.result[1]["samples"].unbind()[0].shape[2])

    def test_same_image_is_encoded_once(self):
        cache = ref_cache.RefCache(1 << 20)
        provenance = {"images": {"ref_image_0": ("image", ref_cache.Identity(self.image))}}
        probe = FakeVae()
        sized.encode_sized_references(
            clip=FakeClip(), prompt="p", width=64, height=64, length=124, vae=probe,
            ref_images={"ref_image_0": self.image}, cache=cache, provenance=provenance)
        sized.encode_sized_references(
            clip=FakeClip(), prompt="p", width=64, height=64, length=124, vae=probe,
            ref_images={"ref_image_0": self.image}, cache=cache, provenance=provenance)
        self.assertEqual(probe.calls, 1)
        self.assertEqual(cache.stats["l1_hit"], 1)

    def test_prompt_change_reuses_the_material_encode(self):
        cache = ref_cache.RefCache(1 << 20)
        provenance = {"images": {"ref_image_0": ("image", ref_cache.Identity(self.image))}}
        probe = FakeVae()
        for prompt in ("first prompt", "second prompt"):
            sized.encode_sized_references(
                clip=FakeClip(), prompt=prompt, width=64, height=64, length=124, vae=probe,
                ref_images={"ref_image_0": self.image}, cache=cache, provenance=provenance)
        self.assertEqual(probe.calls, 1)

    def test_different_source_does_not_hit(self):
        cache = ref_cache.RefCache(1 << 20)
        probe = FakeVae()
        other = torch.ones(1, 64, 64, 3, dtype=torch.float32)
        for image in (self.image, other):
            sized.encode_sized_references(
                clip=FakeClip(), prompt="p", width=64, height=64, length=124, vae=probe,
                ref_images={"ref_image_0": image}, cache=cache,
                provenance={"images": {"ref_image_0": ("image", ref_cache.Identity(image))}})
        self.assertEqual(probe.calls, 2)

    def test_output_size_change_still_reuses_the_material(self):
        # Reference images keep their own aligned size, so a new output canvas
        # changes the window latent but not the reference encode.
        cache = ref_cache.RefCache(1 << 20)
        provenance = {"images": {"ref_image_0": ("image", ref_cache.Identity(self.image))}}
        probe = FakeVae()
        for width in (64, 96):
            sized.encode_sized_references(
                clip=FakeClip(), prompt="p", width=width, height=64, length=124, vae=probe,
                ref_images={"ref_image_0": self.image}, cache=cache, provenance=provenance)
        self.assertEqual(probe.calls, 1)

    def test_larger_image_does_not_hit(self):
        cache = ref_cache.RefCache(1 << 20)
        probe = FakeVae()
        for image in (self.image, torch.ones(1, 128, 128, 3, dtype=torch.float32)):
            sized.encode_sized_references(
                clip=FakeClip(), prompt="p", width=64, height=64, length=124, vae=probe,
                ref_images={"ref_image_0": image}, cache=cache,
                provenance={"images": {"ref_image_0": ("image", ref_cache.Identity(image))}})
        self.assertEqual(probe.calls, 2)

    def test_cached_latent_is_on_cpu(self):
        cache = ref_cache.RefCache(1 << 20)
        result = encode(self.image, cache)
        refs = result.result[0][0][1]["minimax_refs"]
        self.assertEqual(refs[0]["latent"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
