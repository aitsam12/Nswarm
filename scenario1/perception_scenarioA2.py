# ===============================================================
# Neuromorphic Perception + Multi-Object Tracking (Fast Version)
# Corrected to send ALL trackers + correct CSV logging
# FIXED: Slice timestamp now uses sl.ts
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
MAX_OBJECTS = 3

ZMQ_PORT = 5555
PROCESS_INTERVAL = 0.03

WARMUP_SECONDS = 1.0     # <--- NEW

LOG_DIR = "perception_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ===============================================================
# Tracker
# ===============================================================
class Tracker:
    def __init__(self, tracker_id, bbox, centroid):
        self.id = tracker_id
        self.bbox = bbox
        self.centroid = centroid.astype(np.float32)
        self.miss = 0

    def update(self, bbox, centroid):
        self.bbox = bbox
        self.centroid = centroid.astype(np.float32)
        self.miss = 0


# ===============================================================
# Utility Functions
# ===============================================================
def get_new_id(trackers):
    used = {t.id for t in trackers}
    for i in range(1, MAX_OBJECTS + 1):
        if i not in used:
            return i
    return -1


def send_to_zmq(socket, objects):
    """Send every tracker individually."""
    try:
        for o in objects:
            msg = f"ID:{o['id']} Block:({o['bx']},{o['by']});"
            socket.send_string(msg, zmq.NOBLOCK)
    except zmq.Again:
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

        # 5ms slicer
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

        self.last_process_wall = time.time()

        # NEW: Warm-up start time
        self.start_time = time.time()

        # CSV LOGGING
        self.log_path = os.path.join(LOG_DIR, "trackers_log.csv")
        self.log_file = open(self.log_path, "w", newline="")
        self.logger = csv.writer(self.log_file)

        self.logger.writerow([
            "unix_time",
            "frame_ts_us",
            "tracker_id",
            "cx_px", "cy_px",
            "bx", "by",
            "w_px", "h_px",
            "processing_time_s"
        ])

    def key_cb(self, key, scancode, action, mods, window):
        if key in [UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q]:
            window.set_close_flag()

    def run(self):

        with MTWindow("Neuromorphic Perception (Fast, Multi-Tracker)",
                      self.width, self.height,
                      BaseWindow.RenderMode.BGR) as window:

            window.set_keyboard_callback(
                lambda *args: self.key_cb(*args, window)
            )

            for sl in self.slicer:

                EventLoop.poll_and_dispatch()
                if window.should_close():
                    break

                ev = sl.events
                if ev.size == 0:
                    continue

                # Flip X
                ev["x"][:] = (self.width - 1) - ev["x"]

                # Accumulate
                xs = ev["x"]
                ys = ev["y"]
                np.add.at(self.event_map, (ys, xs), 1)

                now = time.time()
                if now - self.last_process_wall < PROCESS_INTERVAL:
                    continue

                processing_start = now
                self.last_process_wall = now

                # Frame update
                BaseFrameGenerationAlgorithm.generate_frame(ev, self.frame)

                # Decay memory
                max_val = self.event_map.max()
                if max_val > 0:
                    scaled = self.event_map / max_val
                else:
                    scaled = 0.0

                self.memory_map = DECAY * self.memory_map + (1 - DECAY) * scaled
                mem8 = (self.memory_map * 255.0).astype(np.uint8)

                # FAST BLOB DETECTION
                bboxes, cents = fast_blobs.detect_blobs(mem8, THRESHOLD, MIN_AREA)
                bboxes = np.asarray(bboxes)
                cents = np.asarray(cents)

                detections = []
                for i in range(bboxes.shape[0]):
                    x, y, w, h = bboxes[i]
                    cx, cy = cents[i]
                    detections.append({
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "centroid": np.array([cx, cy], np.float32)
                    })

                # ---------------------------------------------------------------
                # WARM-UP: skip tracking for the first WARMUP_SECONDS
                # ---------------------------------------------------------------
                if time.time() - self.start_time < WARMUP_SECONDS:
                    self.event_map.fill(0.0)
                    window.show_async(self.frame)
                    continue

                # TRACKING
                updated = {t: False for t in self.trackers}
                used_det = [False] * len(detections)

                # Match existing
                for di, det in enumerate(detections):
                    dc = det["centroid"]
                    best_tr = None
                    best_d = 1e9

                    for tr in self.trackers:
                        d = np.linalg.norm(dc - tr.centroid)
                        if d < best_d and d < DISTANCE_THRESHOLD:
                            best_d = d
                            best_tr = tr

                    if best_tr:
                        best_tr.update(det["bbox"], det["centroid"])
                        updated[best_tr] = True
                        used_det[di] = True

                # Create new
                for di, det in enumerate(detections):
                    if used_det[di]:
                        continue
                    new_id = get_new_id(self.trackers)
                    if new_id == -1:
                        continue
                    tr = Tracker(new_id, det["bbox"], det["centroid"])
                    self.trackers.append(tr)
                    updated[tr] = True

                # Remove stale
                self.trackers = [
                    tr for tr in self.trackers
                    if updated.get(tr, False) or tr.miss + 1 <= MAX_MISS
                ]

                # Drawing + Logging + ZMQ
                vis = self.frame.copy()
                zmq_objects = []

                unix_time = time.time()
                frame_ts_us = sl.t      # <-- FIXED

                for tr in self.trackers:
                    x, y, w, h = tr.bbox
                    cx, cy = tr.centroid

                    # Draw
                    cv2.rectangle(vis, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 3,
                               (0, 0, 255), -1)
                    cv2.putText(vis, f"ID {tr.id}",
                                (x, max(0, y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

                    # Grid cell
                    bx = int(cx // self.block_w)
                    by = int(cy // self.block_h)
                    bx = max(0, min(GRID_W - 1, bx))
                    by = max(0, min(GRID_H - 1, by))

                    zmq_objects.append({"id": tr.id, "bx": bx, "by": by})

                    # CSV log
                    self.logger.writerow([
                        unix_time,
                        frame_ts_us,
                        tr.id,
                        float(cx), float(cy),
                        bx, by,
                        w, h,
                        now - processing_start
                    ])

                send_to_zmq(self.socket, zmq_objects)

                # Visualisation
                heat = cv2.applyColorMap(mem8, cv2.COLORMAP_JET)
                out = cv2.addWeighted(heat, 0.65, vis, 0.35, 0)
                window.show_async(out)

                self.event_map.fill(0.0)

        self.log_file.close()


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    app = NeuromorphicPerceptionFast()
    app.run()

