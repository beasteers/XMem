import time
import torch
import logging
from collections import Counter

log = logging.getLogger('XMem')

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = "Tentative"
    Confirmed = "Confirmed"
    Deleted = "Deleted"



class Track:
    def __init__(self, track_id, t_obs, n_init=3, max_age=None, tentative_age=1, ema_alpha=0.2):
        # params
        self._n_init = n_init
        self._max_age = max_age
        self._tentative_age = tentative_age or 1

        # initial state
        self.state = TrackState.Tentative if n_init > 0 else TrackState.Confirmed
        self.track_id = track_id
        # initial time
        self.first_seen = self.last_seen = t_obs
        self.last_predict_time = t_obs
        self.steps_since_update = 0
        self.hits = 0

        # class distribution
        self.label_count = Counter()
        self.class_distribution = 0.001
        self.ema_alpha = ema_alpha

    def __str__(self):
        return f'Track({self.first_seen}-{self.last_seen}, {self.label_count})'

    @property
    def pred_label(self):
        xs = self.label_count.most_common(1)
        return xs[0][0] if xs else None

    # --------------------------- Collection Management -------------------------- #

    @classmethod
    def update_tracks(cls, tracks, track_ids, curr_time, delete_dict=True, **kw):
        track_ids = [t for t in track_ids if t >= 0]
        new = [t for t in track_ids if t not in tracks]
        # update track counters
        for track in tracks.values():
            track.step(curr_time)
        # create new tracks
        for ti in track_ids:
            if ti not in tracks:
                tracks[ti] = cls(ti, curr_time, **kw)
            tracks[ti].mark_hit()
        # check for missed tracks
        for t in tracks.values():
            t.mark_missed()

        # delete objects that didn't have consistent initial detections
        deleted = {ti for ti in tracks if tracks[ti].is_deleted()}
        if delete_dict:
            for ti in deleted:
                tracks.pop(ti)
        return new, deleted
    
    @staticmethod
    def potentially_delete(tracks):
        return {t for t in tracks if tracks[t].leiway() < 2}
    
    # ------------------------------------ Age ----------------------------------- #

    @property
    def age(self):
        return self.last_predict_time - self.first_seen

    @property
    def time_since_update(self):
        return self.last_predict_time - self.last_seen
    
    def leiway(self):
        if self.state == TrackState.Tentative and self.steps_since_update > 1:
            return self._tentative_age - self.steps_since_update # self.steps_since_update > 1
        elif self._max_age:
            return self._max_age - self.steps_since_update
        return 1000

    # -------------------------------- Step Update ------------------------------- #
    
    def step(self, t_obs):
        self.steps_since_update += 1
        self.last_predict_time = t_obs or time.time()

    def mark_hit(self):
        self.hits += 1
        self.steps_since_update = 0
        self.last_seen = self.last_predict_time
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            if self.hits > 2:
                log.info(f'Track {self.track_id} confirmed')
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.leiway() < 1:
            self.mark_deleted()

    def mark_deleted(self):
        log.debug(f'{self.state} Track {self.track_id} deleted.')
        self.state = TrackState.Deleted

    # ------------------------------- State Checks ------------------------------- #

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    # ---------------------------- Class Distribution ---------------------------- #

    def compare_class_distribution(self, class_distribution):
        if not torch.is_tensor(self.class_distribution):
            return torch.ones_like(class_distribution)
        return 1 - torch.norm(self.class_distribution - class_distribution, dim=1)

    def update_class_distribution(self, class_distribution):
        if torch.is_tensor(class_distribution):
            top = torch.topk(class_distribution, k=10)
            class_distribution = torch.zeros_like(class_distribution)
            class_distribution[top.indices] = top.values
        # print('aa', self.track_id, self.label_count)
        if not torch.is_tensor(self.class_distribution):
            self.class_distribution = class_distribution
            return
        # print('a1', self.track_id, top)
        self.class_distribution = (
            self.class_distribution * self.ema_alpha + 
            class_distribution * (1 - self.ema_alpha)
        )

        # print('a2', self.track_id, torch.topk(self.class_distribution, 5))
        # input()

    def check_class_distribution(self, tracked_labels, tracked_conf_threshold):
        if not torch.is_tensor(self.class_distribution):
            return True  # idk wait for class distribution to get assigned ig
        # print('b', self)
        # print('b,', self.class_distribution[tracked_labels] > tracked_conf_threshold, self.class_distribution[tracked_labels])
        return (self.class_distribution[tracked_labels] > tracked_conf_threshold).any(0)
    
    def missed_class_distribution(self, factor=0.15):
        self.class_distribution = self.class_distribution ** (1 + factor)

    # def update_distribution(self, topk_classes, topk_scores):
    #     self.class_distribution *= (1 - self.ema_alpha)
    #     self.class_distribution[topk_classes] += self.ema_alpha * topk_scores

    # def compare_distribution(self, dists):
    #     pass