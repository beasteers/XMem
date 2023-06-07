import torch
import warnings

from .kv_memory_store import KeyValueMemoryStore, splice
from xmem.model.memory_util import *
from .config import DEFAULT_CONFIG

class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """
    min_work_elements = None
    max_work_elements = None
    def __init__(self, config=None):
        config = config or DEFAULT_CONFIG
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

        self.enable_long_term = config['enable_long_term']
        self.enable_long_term_usage = config['enable_long_term_count_usage']
        if self.enable_long_term:
            self.max_mt_frames = config['max_mid_term_frames']
            self.min_mt_frames = config['min_mid_term_frames']
            self.num_prototypes = config['num_prototypes']
            self.max_long_elements = config['max_long_term_elements']

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The hidden state will be stored in a single tensor for all objects
        # B x num_objects x CH x H x W
        self.hidden = None

        self.work_mem = KeyValueMemoryStore(count_usage=self.enable_long_term)
        if self.enable_long_term:
            self.long_mem = KeyValueMemoryStore(count_usage=self.enable_long_term_usage)

    def __str__(self) -> str:
        return (
            f'MemoryManager(CK={self.CK}, CV={self.CV}, H={self.H}, W={self.W}, '
            f'\n  work_mem_full={self.work_mem.size}/{self.max_work_elements},'
            f'\n  work_mem={self.work_mem},'
            f'\n  long_mem={self.long_mem})')

    @property
    def object_ids(self):
        return sorted(self.work_mem.values)

    @property
    def n_objects(self):
        return len(self.work_mem.values)

    def update_config(self, config):
        self.H = None
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

        assert self.enable_long_term == config['enable_long_term'], 'cannot update this'
        assert self.enable_long_term_usage == config['enable_long_term_count_usage'], 'cannot update this'

        self.enable_long_term_usage = config['enable_long_term_count_usage']
        if self.enable_long_term:
            self.max_mt_frames = config['max_mid_term_frames']
            self.min_mt_frames = config['min_mid_term_frames']
            self.num_prototypes = config['num_prototypes']
            self.max_long_elements = config['max_long_term_elements']

    def _readout(self, affinity, v):
        # this function is for a single object group
        return v @ affinity

    def match_memory(self, query_key, selection):
        # query_key: B x C^k x H x W
        # selection:  B x C^k x H x W
        h, w = query_key.shape[-2:]

        query_key = query_key.flatten(start_dim=2)
        selection = selection.flatten(start_dim=2) if selection is not None else None

        """
        Memory readout using keys
        """

        # Use long-term memory?
        use_long_term = self.enable_long_term and self.long_mem.engaged()
        long_mem_size = self.long_mem.size if use_long_term else 0

        # get keys
        memory_key = self.work_mem.key
        shrinkage = self.work_mem.shrinkage

        # combine working memory with long term memory
        if use_long_term:
            memory_key = torch.cat([self.long_mem.key, memory_key], -1)
            shrinkage = torch.cat([self.long_mem.shrinkage, shrinkage], -1) 

        # get memory similarity
        similarity = get_similarity(memory_key, shrinkage, query_key, selection)
        work_mem_similarity = similarity[:, long_mem_size:]
        long_mem_similarity = similarity[:, :long_mem_size]

        # # compute affinity per object
        affinity = {}
        usages = {}
        for gi in self.work_mem.values:
            # get similarity for a specific object
            sim = work_mem_similarity[:, -self.work_mem.get_v_size(gi):]
            
            # maybe add long term
            if use_long_term and gi in self.long_mem.values:
                long_sim = long_mem_similarity[:, -self.long_mem.get_v_size(gi):]
                sim = torch.cat([long_sim, sim], 1)

            # get affinity
            aff, usage = do_softmax(sim, top_k=self.top_k, inplace=False, return_usage=True)
            affinity[gi] = aff
            usages[gi] = usage

        # merge the working and lt values before readout
        all_memory_value = {}
        for gi, gv in self.work_mem.values.items():
            if use_long_term and gi in self.long_mem.values:
                gv = torch.cat([self.long_mem.values[gi], self.work_mem.values[gi]], -1)
            all_memory_value[gi] = gv

        """
        Record memory usage for working and long-term memory
        """
        if usages:
            # ignore the index return for long-term memory
            self.work_mem.update_usage({
                gi: usage[:, long_mem_size:].flatten()
                for gi, usage in usages.items()
            })

            if self.enable_long_term_usage:
                # ignore the index return for working memory
                self.long_mem.update_usage({
                    gi: usage[:, :long_mem_size].flatten()
                    for gi, usage in usages.items()
                    if gi in self.long_mem.values
                })

        # Shared affinity within each group
        all_readout_mem = torch.cat([
            self._readout(affinity[gi], gv)
            for gi, gv in sorted(all_memory_value.items())
        ], 0) if len(all_memory_value) else torch.zeros((0, self.CV, h, w))

        return all_readout_mem.view(all_readout_mem.shape[0], self.CV, h, w)

    def add_memory(self, key, shrinkage, value, objects, selection=None):
        # key: 1*C*H*W
        # value: 1*num_objects*C*H*W
        # objects contain a list of object indices
        if self.H is None:
            self.H, self.W = key.shape[-2:]
            self.HW = self.H*self.W
            if self.enable_long_term:
                # convert from num. frames to num. nodes
                self.min_work_elements = self.min_mt_frames*self.HW
                self.max_work_elements = self.max_mt_frames*self.HW

        # key:   1*C*N
        # value: num_objects*C*N
        key = key.flatten(start_dim=2)
        value = value[0].flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        if shrinkage is not None:
            shrinkage = shrinkage.flatten(start_dim=2) 

        if selection is not None:
            if not self.enable_long_term:
                warnings.warn('the selection factor is only needed in long-term mode', UserWarning)
            selection = selection.flatten(start_dim=2)

        self.work_mem.add(key, value, shrinkage, selection, objects)

        # long-term memory cleanup
        if self.enable_long_term:
            # Do memory compressed if needed
            if self.work_mem.size >= self.max_work_elements:
                # Remove obsolete features if needed
                if self.long_mem.size >= (self.max_long_elements-self.num_prototypes):
                    self.long_mem.remove_obsolete_features(self.max_long_elements-self.num_prototypes)
                    
                self.compress_features()

    def delete_object_id(self, track_id, i):
        # print(self.hidden.shape, track_id, i)
        self.work_mem.delete(track_id)
        self.long_mem.delete(track_id)
        self.hidden = (
            torch.cat([self.hidden[:,:i], self.hidden[:,i+1:]], 1)
            # if i < self.hidden.shape[1]-1 else
            # self.hidden[:,:i] if i else 
            # self.hidden[:,i+1:]
        )
        # print(self.hidden.shape)

    def create_hidden_state(self, n, sample_key):
        # n is the TOTAL number of objects
        h, w = sample_key.shape[-2:]
        if self.hidden is None:
            self.hidden = torch.zeros((1, n, self.hidden_dim, h, w), device=sample_key.device)
        elif self.hidden.shape[1] != n:
            self.hidden = torch.cat([
                self.hidden, 
                torch.zeros((1, n-self.hidden.shape[1], self.hidden_dim, h, w), device=sample_key.device)
            ], 1)

        assert self.hidden.shape[1] == n

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden

    def compress_features(self):
        HW = self.HW
        candidate_value = {}
        total_work_mem_size = self.work_mem.size
        for gi, gv in self.work_mem.values.items():
            # Some object groups might be added later in the video
            # So not all keys have values associated with all objects
            # We need to keep track of the key->value validity
            mem_size = gv.shape[-1]
            # mem_size is smaller than total_work_mem_size, but at least HW
            assert HW <= mem_size <= total_work_mem_size
            if self.min_work_elements + HW < mem_size <= total_work_mem_size:
                candidate_value[gi] = gv[:, :, HW:-self.min_work_elements + HW]

        # perform memory consolidation
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.work_mem.get_all_sliced(HW, -self.min_work_elements+HW), candidate_value)

        # remove consolidated working memory
        self.work_mem.sieve_by_range(HW, -self.min_work_elements+HW, min_size=self.min_work_elements+HW)

        # add to long-term memory
        self.long_mem.add(prototype_key, prototype_value, prototype_shrinkage, selection=None, objects=None)

    def consolidation(self, candidate_key, candidate_shrinkage, candidate_selection, usage, candidate_value):
        # keys: 1*C*N
        # values: num_objects*C*N
        N = candidate_key.shape[-1]

        # find the indices with max usage
        _, max_usage_indices = torch.topk(usage, k=min(self.num_prototypes, usage.shape[-1]), dim=-1, sorted=True)
        prototype_indices = max_usage_indices.flatten()

        prototype_key = candidate_key[:, :, prototype_indices]
        prototype_selection = candidate_selection[:, :, prototype_indices] if candidate_selection is not None else None

        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key, candidate_shrinkage, prototype_key, prototype_selection)

        prototype_value = {}
        prototype_shrinkage = torch.zeros(1, 1, prototype_key.shape[-1], device=prototype_indices.get_device())
        prototype_shrinkage_count = torch.zeros_like(prototype_shrinkage) + 1e-7
        for gi, gv in candidate_value.items():
            # Prototypes are invalid for out-of-bound groups
            valid = prototype_indices >= (N-candidate_value[gi].shape[2]) 
            if not valid.any():
                continue

            # convert similarity to affinity
            aff = do_softmax(similarity[:, -gv.shape[2]:, valid])
            prototype_value[gi] = self._readout(aff[0], gv)

            prototype_shrinkage[:, :, valid] += self._readout(aff, candidate_shrinkage[:, :, -gv.shape[2]:]) if candidate_shrinkage is not None else None
            prototype_shrinkage_count[:, :, valid] += 1

        # readout the shrinkage term
        prototype_shrinkage = prototype_shrinkage / prototype_shrinkage_count if candidate_shrinkage is not None else None
        return prototype_key, prototype_value, prototype_shrinkage
