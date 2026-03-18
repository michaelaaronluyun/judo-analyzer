"""
Kalman-filter-based athlete tracker.
Extracted from DLSU Judo Analyzer notebook (cells 14-15).
"""
import numpy as np
from scipy.spatial.distance import cdist


class KalmanTrack:
    """2-D Kalman filter tracking an athlete centre-of-mass (x, y, vx, vy)."""

    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        dt = 1.0
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * 0.001
        self.R = np.eye(2) * 0.01
        self.missed = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missed += 1
        return self.x[:2]

    def update(self, measurement):
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0
        return self.x[:2]

    @property
    def position(self):
        return self.x[:2]

    @property
    def velocity(self):
        return self.x[2:]


class MultiPersonTracker:
    MAX_MISSED = 15
    MAX_DISTANCE = 0.35

    def __init__(self):
        self.tracks = {}
        self._next_id = 0

    def _com(self, kps):
        pts = [[kps[i]["x"], kps[i]["y"]] for i in [5, 6, 11, 12] if i in kps]
        return np.mean(pts, axis=0) if pts else None

    def update(self, poses):
        for t in self.tracks.values():
            t.predict()

        measurements, valid_poses = [], []
        for p in poses:
            com = self._com(p)
            if com is not None:
                measurements.append(com)
                valid_poses.append(p)

        assigned_ids = {}

        if self.tracks and measurements:
            track_ids = list(self.tracks.keys())
            track_preds = np.array([self.tracks[i].position for i in track_ids])
            meas_arr = np.array(measurements)
            cost = cdist(meas_arr, track_preds)
            used_tracks, used_meas = set(), set()
            flat_order = np.argsort(cost.ravel())
            for idx in flat_order:
                m_i, t_i = divmod(int(idx), len(track_ids))
                if m_i in used_meas or t_i in used_tracks:
                    continue
                if cost[m_i, t_i] > self.MAX_DISTANCE:
                    break
                assigned_ids[track_ids[t_i]] = valid_poses[m_i]
                self.tracks[track_ids[t_i]].update(measurements[m_i])
                used_tracks.add(t_i)
                used_meas.add(m_i)
            for m_i, pose in enumerate(valid_poses):
                if m_i not in used_meas:
                    self.tracks[self._next_id] = KalmanTrack(measurements[m_i])
                    assigned_ids[self._next_id] = pose
                    self._next_id += 1
        elif measurements:
            for com, pose in zip(measurements, valid_poses):
                self.tracks[self._next_id] = KalmanTrack(com)
                assigned_ids[self._next_id] = pose
                self._next_id += 1

        stale = [i for i, t in self.tracks.items() if t.missed > self.MAX_MISSED]
        for i in stale:
            del self.tracks[i]

        if len(self.tracks) > 2:
            keep = {k for k, _ in sorted(self.tracks.items(), key=lambda kv: kv[1].missed)[:2]}
            for k in list(self.tracks.keys()):
                if k not in keep:
                    del self.tracks[k]
            assigned_ids = {k: v for k, v in assigned_ids.items() if k in keep}

        return assigned_ids
