#!/usr/bin/env python3
# ===============================================================
# Scenario 2 – Vision-Driven Swarm Stability (Collision Suppression)
#
# Neuromorphic Perception + Multi-Object Tracking (Fast Version)
# + velocity estimation
# + predictive future grid cell
# + enhanced ZMQ messages to SpiNNaker
# ===============================================================

import numpy as np
import cv2
import zmq
import os
import time
import csv

from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_ui import MTWindow, BaseWindow, EventLoop, UIKeyEvent

import fast_blobs


# -------------------------
# Configuration
# -------------------------
ACCUM_MS = 5
GRID_W, GRID_H = 32, 24
THRESHOLD = 5
MIN_AREA = 30
DECAY = 0.7
DISTANCE_THRESHOLD = 45.0
MAX_MISS = 10
MAX_OBJECTS = 5

ZMQ_PORT = 5555
PROCESS_INTERVAL = 0.03

WARMUP_SECONDS = 1.0
PREDICT_HORIZON_SEC = 0.15    # predictive forward time

LOG_DIR = "perception_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ===============================================================
# Tracker with velocity
# ===============================================================
class Tracker:
    def __init__(self, tracker_id, bbox, centroid):
        self.id = tracker_id
        self.bbox = bbox
        self.centroid = centroid.astype(np.float32)
        self.prev_centroid = centroid.astype(np.float32)
        self.vx = 0.0
        self.vy = 0.0
        self.last_ts = time.time()
        self.miss = 0

    def update(self, bbox, centroid, current_ts):
        dt = current_ts - self.last_ts
        self.last_ts = current_ts

        centroid = centroid.astype(np.float32)

        if dt > 1e-5:
            self.vx = (centroid[0] - self.prev_centroid[0]) / dt
            self.vy = (centroid[1] - self.prev_centroid[1]) / dt

        self.prev_centroid = centroid.copy()
        self.centroid = centroid
        self.bbox = bbox
        self.miss = 0


# ===============================================================
# Utility
# ===============================================================
def get_new_id(trackers):
    used = {t.id for t in trackers}
    for i in range(1, MAX_OBJECTS + 1):
        if i not in used:
            return i
    return -1


def send_to_zmq(socket, objects):
    """
    objects = [{
        id, bx, by, vx, vy, pbx, pby
    }]
    """
    for o in objects:
        msg = (
            f"ID:{o['id']} "
            f"Block:({o['bx']},{o['by']}) "
            f"Vel:({o['vx']:.2f},{o['vy']:.2f}) "
            f"Pred:({o['pbx']},{o['pby']});"
        )
        try:
            socket.send_string(msg)
        except zmq.ZMQError:
            pass


# ===============================================================
# Perception Class
# ===============================================================
class NeuromorphicPerceptionFast:
    def __init__(self, input_file=None):

        # Camera
        if input_file:
            self.camera = Camera.from_file(input_file)
        else:
            self.camera = Camera.from_first_available()

        self.width = self.camera.width()
        self.height = self.camera.height()

        self.slicer = CameraStreamSlicer(
            self.camera.move(),
            SliceCondition.make_n_us(ACCUM_MS * 1000)
        )

        self.event_map = np.zeros((self.height, self.width), np.float32)
        self.memory_map = np.zeros((self.height, self.width), np.float32)
        self.frame = np.zeros((self.height, self.width, 3), np.uint8)

        self.trackers = []

        self.block_w = self.width // GRID_W
        self.block_h = self.height // GRID_H

        # ZMQ
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{ZMQ_PORT}")
        print(f"[ZMQ] PUB bound on tcp://*:{ZMQ_PORT}")

        self.last_process_wall = time.time()
        self.start_time = time.time()

        # CSV LOGGING
        self.log_path = os.path.join(LOG_DIR, "trackers_log.csv")
        self.log_file = open(self.log_path, "w", newline="")
        self.logger = csv.writer(self.log_file)

        self.logger.writerow([
            "unix_time",
            "frame_ts_us",
            "id",
            "cx", "cy",
            "vx", "vy",
            "bx", "by",
            "pbx", "pby",
            "w", "h",
            "processing_time"
        ])

    def key_cb(self, key, scancode, action, mods, window):
        if key in [UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q]:
            window.set_close_flag()

    def run(self):

        with MTWindow("Neuromorphic Perception (Scenario 2)",
                      self.width, self.height,
                      BaseWindow.RenderMode.BGR) as window:

            window.set_keyboard_callback(lambda *a: self.key_cb(*a, window))

            for sl in self.slicer:

                EventLoop.poll_and_dispatch()
                if window.should_close():
                    break

                ev = sl.events
                if ev.size == 0:
                    continue

                # Flip X
                ev["x"][:] = (self.width - 1) - ev["x"]

                xs = ev["x"]
                ys = ev["y"]
                np.add.at(self.event_map, (ys, xs), 1)

                now = time.time()
                if now - self.last_process_wall < PROCESS_INTERVAL:
                    continue

                processing_start = now
                self.last_process_wall = now

                BaseFrameGenerationAlgorithm.generate_frame(ev, self.frame)

                # Memory decay
                max_val = self.event_map.max()
                scaled = self.event_map / max_val if max_val > 0 else 0.0
                self.memory_map = DECAY * self.memory_map + (1 - DECAY) * scaled
                mem8 = (self.memory_map * 255.0).astype(np.uint8)

                # Blob detection
                bboxes, cents = fast_blobs.detect_blobs(mem8, THRESHOLD, MIN_AREA)
                detections = []
                for (x, y, w, h), (cx, cy) in zip(bboxes, cents):
                    detections.append({
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "centroid": np.array([cx, cy], np.float32)
                    })

                # WARM-UP
                if time.time() - self.start_time < WARMUP_SECONDS:
                    self.event_map.fill(0)
                    window.show_async(self.frame)
                    continue

                # =====================================================
                # TRACKING
                # =====================================================
                updated = {tr: False for tr in self.trackers}
                num_tr = len(self.trackers)
                num_det = len(detections)

                if num_tr and num_det:
                    cost = np.full((num_tr, num_det), np.inf)
                    for ti, tr in enumerate(self.trackers):
                        for di, det in enumerate(detections):
                            cost[ti, di] = np.linalg.norm(det["centroid"] - tr.centroid)

                    assigned_dets = set()
                    while True:
                        ti, di = divmod(np.argmin(cost), num_det)
                        if not np.isfinite(cost[ti, di]) or cost[ti, di] > DISTANCE_THRESHOLD:
                            break

                        tr = self.trackers[ti]
                        det = detections[di]
                        tr.update(det["bbox"], det["centroid"], now)

                        updated[tr] = True
                        assigned_dets.add(di)

                        cost[ti, :] = np.inf
                        cost[:, di] = np.inf

                    # new trackers
                    for di, det in enumerate(detections):
                        if di not in assigned_dets:
                            nid = get_new_id(self.trackers)
                            if nid != -1:
                                tr = Tracker(nid, det["bbox"], det["centroid"])
                                self.trackers.append(tr)
                                updated[tr] = True

                elif num_det > 0:
                    # no existing trackers
                    for det in detections:
                        nid = get_new_id(self.trackers)
                        if nid == -1:
                            break
                        tr = Tracker(nid, det["bbox"], det["centroid"])
                        self.trackers.append(tr)
                        updated[tr] = True

                # miss counting
                for tr in self.trackers:
                    if not updated.get(tr, False):
                        tr.miss += 1
                self.trackers = [tr for tr in self.trackers if tr.miss <= MAX_MISS]

                # =====================================================
                # SENDING + VISUALISATION
                # =====================================================
                vis = self.frame.copy()
                zmq_objects = []

                unix_time = time.time()
                frame_ts_us = sl.t

                for tr in self.trackers:

                    x, y, w, h = tr.bbox
                    cx, cy = tr.centroid
                    vx, vy = tr.vx, tr.vy

                    # Grid location
                    bx = int(cx // self.block_w)
                    by = int(cy // self.block_h)
                    bx = max(0, min(GRID_W - 1, bx))
                    by = max(0, min(GRID_H - 1, by))

                    # Prediction
                    px = cx + vx * PREDICT_HORIZON_SEC
                    py = cy + vy * PREDICT_HORIZON_SEC

                    pbx = int(px // self.block_w)
                    pby = int(py // self.block_h)
                    pbx = max(0, min(GRID_W - 1, pbx))
                    pby = max(0, min(GRID_H - 1, pby))

                    zmq_objects.append({
                        "id": tr.id,
                        "bx": bx, "by": by,
                        "vx": vx, "vy": vy,
                        "pbx": pbx, "pby": pby
                    })

                    # logging
                    self.logger.writerow([
                        unix_time,
                        frame_ts_us,
                        tr.id,
                        float(cx), float(cy),
                        vx, vy,
                        bx, by,
                        pbx, pby,
                        w, h,
                        now - processing_start
                    ])

                    # drawing
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 3, (0,0,255), -1)

                # Send to SpiNNaker
                if zmq_objects:
                    send_to_zmq(self.socket, zmq_objects)

                # final image
                heat = cv2.applyColorMap(mem8, cv2.COLORMAP_JET)
                out = cv2.addWeighted(heat, 0.65, vis, 0.35, 0)
                window.show_async(out)

                self.event_map.fill(0)

        self.log_file.close()


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    app = NeuromorphicPerceptionFast()
    app.run()

