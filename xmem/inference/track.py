import time

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3



class Track:
    def __init__(self, track_id, t_obs, n_init=3, max_age=None):
        # params
        self._n_init = n_init
        self._max_age = max_age

        # initial state
        self.state = TrackState.Tentative
        self.track_id = track_id
        # initial time
        self.first_seen = self.last_seen = t_obs
        self.last_predict_time = t_obs
        self.steps_since_update = 0
        self.hits = 1

    @staticmethod
    def update_tracks(tracks, track_ids, curr_time, **kw):
        new = [t for t in track_ids if t not in tracks]
        # update track counters
        for track in tracks.values():
            track.step(curr_time)
        for ti in track_ids:
            if ti not in tracks:
                tracks[ti] = Track(ti, curr_time, **kw)
            tracks[ti].mark_hit()
        for t in tracks.values():
            t.check_missed()

        # delete objects that didn't have consistent initial detections
        deleted = {ti for ti in tracks if tracks[ti].is_deleted()}
        for ti in deleted:
            tracks.pop(ti)
        return new, deleted
    
    @staticmethod
    def potentially_delete(tracks):
        return {t for t in tracks if tracks[t].leiway() < 2}

    @property
    def age(self):
        return self.last_predict_time - self.first_seen

    @property
    def time_since_update(self):
        return self.last_predict_time - self.last_seen
    
    def step(self, t_obs):
        self.steps_since_update += 1
        self.last_predict_time = t_obs or time.time()

    def mark_hit(self):
        self.hits += 1
        self.steps_since_update = 0
        self.last_seen = self.last_predict_time
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def leiway(self):
        if self.state == TrackState.Tentative and self.steps_since_update > 1:
            return 1 - self.steps_since_update # self.steps_since_update > 1
        elif self._max_age and self.steps_since_update > self._max_age:
            return self._max_age - self.steps_since_update
        return 1000

    def check_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.leiway() < 1:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
