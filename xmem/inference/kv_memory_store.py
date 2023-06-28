import torch
from typing import List



class KeyValueMemoryStore:
    """
    Works for key/value pairs type storage
    e.g., working and long-term memory
    """

    """
    An object group is created when new objects enter the video
    Objects in the same group share the same temporal extent
    i.e., objects initialized in the same frame are in the same group
    For DAVIS/interactive, there is only one object group
    For YouTubeVOS, there can be multiple object groups
    """

    def __init__(self, count_usage: bool):

        # keys are stored in a single tensor and are shared between groups/objects
        # values are stored as a dict indexed by object ids
        self.k = None
        self.values = {}

        # shrinkage and selection are also single tensors
        self.shrinkage = self.selection = None

        # usage
        self.count_usage = count_usage
        # self.use_count = self.life_count = None
        self.use_counts = {}
        self.life_counts = {}

    def __str__(self) -> str:
        nk = tuple(self.k.shape) if self.k is not None else '-'
        return f'Memory(key: {nk}, {len(self.values)} values)'

    def add(self, key, value, shrinkage, selection, objects: List[int]):
        # key: [1, C, N]
        # value: [num_objects, C, N]
        # shrinkage: [?, ?, N]
        # selection: [?, ?, N]
        # new_count: [1, 1, N]
        # new_life: [1, 1, N]
        # objects - object ids - otherwise assume value is sorted by object ID
        new_count = torch.zeros((key.shape[0], 1, key.shape[2]), device=key.device, dtype=torch.float32)
        new_life = torch.zeros((key.shape[0], 1, key.shape[2]), device=key.device, dtype=torch.float32) + 1e-7

        # add the key
        self.k = maybe_cat(self.k, key)
        self.shrinkage = maybe_cat(self.shrinkage, shrinkage)
        self.selection = maybe_cat(self.selection, selection)

        # add one value per object
        if not isinstance(value, dict):
            value = dict(zip(objects, value[:, None]))
        for gi in set(self.values) | set(value):
            gv = value[gi]
            self.values[gi] = maybe_cat(self.values.get(gi), gv)
            if self.count_usage:
                self.use_counts[gi] = maybe_cat(self.use_counts.get(gi), new_count)
                self.life_counts[gi] = maybe_cat(self.life_counts.get(gi), new_life)

    def delete(self, i):
        if i in self.values:
            del self.values[i]
            if self.count_usage:
                del self.use_counts[i]
                del self.life_counts[i]
            # # TODO: prune keys
            n = max((x.shape[-1] for x in self.values.values()), default=0)
            self.k = self.k[:, :, -n:] if n and self.k is not None else None
            self.shrinkage = self.shrinkage[:, :, -n:] if n and self.shrinkage is not None else None
            self.selection = self.selection[:, :, -n:] if n and self.selection is not None else None

    def update_usage(self, usages):
        # increase all life count by 1
        # increase use of indexed elements
        if not self.count_usage:
            return 
        
        for gi, usage in usages.items():
            if not len(usage):
                continue
            self.use_counts[gi][:, :, -usage.shape[-1]:] += usage[-self.use_counts[gi].shape[-1]:]
            self.life_counts[gi][:, :, -usage.shape[-1]:] += 1

    def get_usage(self):
        # return normalized usage
        if not self.count_usage:
            raise RuntimeError('I did not count usage! ;o;')
        # return self.use_count / self.life_count
        u = max(self.use_counts.values(), key=lambda x: x.shape[-1])
        shape = u.shape
        usage = torch.zeros(shape, device=u.get_device())
        counts = torch.zeros_like(usage)
        for object_id in self.use_counts:
            u = self.use_counts[object_id] / self.life_counts[object_id]
            usage[:, :, -u.shape[-1]:] += u
            counts[:, :, -u.shape[-1]:] += 1
        return usage / counts

    def sieve_by_range(self, start: int, end: int, min_size: int):
        # keep only the elements *outside* of this range (with some boundary conditions)
        # i.e., concat (a[:start], a[end:])
        # min_size is only used for values, we do not sieve values under this size
        # (because they are not consolidated)

        self.k = splice(self.k, start, end)
        self.shrinkage = splice(self.shrinkage, start, end)
        self.selection = splice(self.selection, start, end)
        # self.use_count = slice_outside(self.use_count, start, end)
        # self.life_count = slice_outside(self.life_count, start, end)

        for gi, gv in self.values.items():
            self.values[gi] = splice(gv, start, end)
            if self.count_usage:
                self.use_counts[gi] = splice(self.use_counts[gi], start, end)
                self.life_counts[gi] = splice(self.life_counts[gi], start, end)

    def remove_obsolete_features(self, max_size: int):
        # get topk usage
        usage = self.get_usage().flatten()
        values, _ = torch.topk(usage, k=(self.size-max_size), largest=False, sorted=True)
        survived = (usage > values[-1])

        # filter all using topk usage
        self.k = self.k[:, :, survived]
        self.shrinkage = self.shrinkage[:, :, survived] if self.shrinkage is not None else None
        # Long-term memory does not store ek so this should not be needed
        self.selection = self.selection[:, :, survived] if self.selection is not None else None

        for gi in self.values:
            s = survived[-self.values[gi].shape[-1]:]
            self.values[gi] = self.values[gi][:, :, s]
            if self.count_usage:
                self.use_counts[gi] = self.use_counts[gi][:, :, s]
                self.life_counts[gi] = self.life_counts[gi][:, :, s]

    def get_all_sliced(self, start: int, end: int):
        # return k, sk, ek, usage in order, sliced by start and end
        end = end or None
        usage = self.get_usage()
        k = self.k[:,:,start:end]
        sk = self.shrinkage[:,:,start:end] if self.shrinkage is not None else None
        ek = self.selection[:,:,start:end] if self.selection is not None else None
        usage = usage[:,:,start:end] if usage is not None else None
        return k, sk, ek, usage

    def get_v_size(self, ni: int):
        return self.values[ni].shape[-1]

    def engaged(self):
        # return self.k is not None
        return bool(self.values)

    @property
    def size(self):
        return self.k.shape[-1] if self.k is not None else 0

    @property
    def key(self):
        return self.k


def maybe_cat(prev, new):
    return torch.cat([prev, new], -1) if prev is not None else new

def splice(x, start, end):
    if x is None:
        return None
    if end == 0:
        return x[:,:,:start]
    return torch.cat([x[:,:,:start], x[:,:,end:]], -1)