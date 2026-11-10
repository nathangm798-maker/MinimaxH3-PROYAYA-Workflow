"""Bounded LRU caches for REF2VA encoding work.

Two levels:

``l1``
    One encoded latent per reference *material* — a reference image, an
    auxiliary video (always read from frame 0, so identical across windows), a
    standalone reference audio, or an encoded source-video window.

``l2``
    One encoded *window* — the positive conditioning plus the empty AV latent,
    reused verbatim when the same prompt meets the same slice again.

Keys are provenance tuples, never content hashes: hashing 124 frames of
1344x768 pixels costs more than the VAE encode it would replace. Provenance
relies on the identity of the source object plus the parameters that select a
slice, and ``Identity`` keeps the source alive so its ``id`` cannot be recycled
while an entry still refers to it.

Model objects (CLIP, VAE) are identified by a tag stamped on the object rather
than by a strong reference, so an entry never pins a multi-gigabyte model in
memory. A reloaded model is a different object and therefore gets a new tag,
which invalidates its entries instead of returning a stale latent.
"""

import collections
import hashlib
import uuid

import torch

DEFAULT_MB = 1024


def tensor_fingerprint(t):
    """Content hash of a tensor — stable across runs, unlike ``Identity``.

    Used for small/medium materials whose identity would otherwise invalidate
    the whole-window L2 entry on every run (LoadImage/LoadAudio/Latent rebuild
    their tensors each time). Full blake2b over the bytes: ~0.5 GB/s, so even a
    1.5 GiB latent costs ~2-3 s once per run — trivial next to a 20 s/window
    VAE encode it unlocks.
    """
    if not isinstance(t, torch.Tensor):
        return ("raw", type(t).__name__, repr(t)[:64])
    t = t.detach().contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    view = t.view(torch.uint8) if t.dtype != torch.uint8 else t
    data = view.cpu().numpy().tobytes()
    h.update(data)
    return ("tensor", tuple(t.shape), str(t.dtype), h.hexdigest())


class _Missing:
    __slots__ = ()

    def __bool__(self):
        return False

    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


class Identity:
    """Hashable wrapper that keeps ``value`` alive so ``id(value)`` stays unique.

    Holds a strong reference on purpose: an entry keyed by a recycled ``id``
    would hand back another video's latent, which is worse than the memory.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other):
        return isinstance(other, Identity) and other.value is self.value

    def __repr__(self):
        return "<Identity %s %x>" % (type(self.value).__name__, id(self.value))


def model_identity(obj):
    """Tag a CLIP/VAE object without holding a reference to it."""
    if obj is None:
        return None
    tag = getattr(obj, "_proyaya_ref_cache_tag", None)
    if tag is None:
        tag = uuid.uuid4().hex
        try:
            setattr(obj, "_proyaya_ref_cache_tag", tag)
        except Exception:  # slots-only or otherwise frozen object
            return ("unstamped", type(obj).__name__, id(obj))
    return ("model", type(obj).__name__, tag)


def payload_bytes(value):
    """Bytes retained by a cached value (tensors only; ints/strings are free)."""
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(payload_bytes(item) for item in value)
    return 0


class RefCache:
    """Byte-bounded LRU with per-level hit accounting."""

    def __init__(self, limit_bytes):
        self._data = collections.OrderedDict()
        self._bytes = 0
        self._limit = max(0, int(limit_bytes))
        self.stats = {"l1_hit": 0, "l1_miss": 0, "l2_hit": 0, "l2_miss": 0}
        self.stores = 0
        self.evictions = 0

    def __len__(self):
        return len(self._data)

    @property
    def bytes_used(self):
        return self._bytes

    @property
    def limit(self):
        return self._limit

    def get(self, key, level):
        entry = self._data.get(key)
        if entry is None:
            self.stats[level + "_miss"] = self.stats.get(level + "_miss", 0) + 1
            return MISSING
        self._data.move_to_end(key)
        self.stats[level + "_hit"] = self.stats.get(level + "_hit", 0) + 1
        return entry[0]

    def put(self, key, value):
        nbytes = payload_bytes(value)
        if nbytes <= 0 or nbytes > self._limit:
            return False
        previous = self._data.pop(key, None)
        if previous is not None:
            self._bytes -= previous[1]
        self._data[key] = (value, nbytes)
        self._bytes += nbytes
        self.stores += 1
        self._evict()
        return True

    def _evict(self):
        while self._bytes > self._limit and self._data:
            _, old = self._data.popitem(last=False)
            self._bytes -= old[1]
            self.evictions += 1

    def set_limit(self, limit_bytes):
        self._limit = max(0, int(limit_bytes))
        self._evict()

    def clear(self):
        self._data.clear()
        self._bytes = 0

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0
        self.stores = 0
        self.evictions = 0

    def summary(self):
        """One line for plan_report; empty when nothing was asked of the cache."""
        asked = sum(self.stats.values())
        if not asked:
            return ""
        return ("参考编码缓存：L1 {l1_hit} 命中 / {l1_miss} 未命中；"
                "L2 {l2_hit} 命中 / {l2_miss} 未命中；占用 {used:.1f}/{limit:.0f} MB，"
                "淘汰 {evictions} 项").format(
                    l1_hit=self.stats.get("l1_hit", 0), l1_miss=self.stats.get("l1_miss", 0),
                    l2_hit=self.stats.get("l2_hit", 0), l2_miss=self.stats.get("l2_miss", 0),
                    used=self._bytes / 1048576.0, limit=self._limit / 1048576.0,
                    evictions=self.evictions)


_SHARED = RefCache(DEFAULT_MB * 1024 * 1024)


def configure(limit_mb):
    """Reset per-run stats and apply the requested ceiling.

    ``limit_mb`` of zero (or less) disables caching and drops what is held.
    Returns the shared cache, or ``None`` when disabled.
    """
    if limit_mb is None or limit_mb <= 0:
        _SHARED.clear()
        return None
    _SHARED.set_limit(int(limit_mb) * 1024 * 1024)
    _SHARED.reset_stats()
    return _SHARED


def shared():
    return _SHARED
