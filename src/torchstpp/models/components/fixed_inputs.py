"""Bounded metadata cache for non-differentiable, versioned tensor inputs."""
from collections import OrderedDict


class FixedInputCache:
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.entries = OrderedDict()

    def get_or_create(self, tensors, settings, factory):
        # Inference tensors lack version counters; differentiable inputs and
        # inference-mode inputs retain the fully checked path.
        if any(t.requires_grad or t.is_inference() for t in tensors):
            return factory()
        key = (settings, tuple((t.untyped_storage().data_ptr(), t.storage_offset(),
                               tuple(t.shape), tuple(t.stride()), t.dtype,
                               t.device, t._version) for t in tensors))
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][1]
        result = factory()
        # Retain inputs so allocator reuse cannot produce a false cache hit.
        self.entries[key] = (tuple(tensors), result)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return result
