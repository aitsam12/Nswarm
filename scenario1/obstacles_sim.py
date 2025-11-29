#!/usr/bin/env python3
"""
Scenario A – Speed-Stratified Circular Orbiting Obstacles (Corrected)
- Slow obstacle on inner orbit
- Medium obstacle on middle orbit
- Fast obstacle on outer orbit
- All orbits guaranteed safe inside 640×480
- Logs full GT data
"""

import pygame
import time
import csv
import os
import math

# ------------------------------------------------------
# SIMULATION CONFIG
# ------------------------------------------------------
WIDTH, HEIGHT = 640, 480
GRID_W, GRID_H = 32, 24
BG_COLOR = (0, 0, 0)

OBS_RADIUS = 14              # radius of circle (drawn on screen)
FPS = 60

# ------------------------------------------------------
# ORBIT CONFIG
# ------------------------------------------------------
CX = WIDTH // 2
CY = HEIGHT // 2

# Maximum radius such that nothing touches screen edge:
# radius < min(CX, CY) - OBS_RADIUS
MAX_SAFE_R = min(CX, CY) - OBS_RADIUS - 10   # 10px margin of safety

# Define three nested radii (slow → medium → fast)
INNER_R  = MAX_SAFE_R * 0.30     # slow
MID_R    = MAX_SAFE_R * 0.60     # medium
OUTER_R  = MAX_SAFE_R * 0.90     # fast  (still safe)

# Angular speeds (rad/s)
SLOW_W   = 1.0
MID_W    = 1.6
FAST_W   = 1.8

# Colours
COLOURS = [
    (255, 255, 255),     # white  – slow (inner)
    (255, 180, 180),     # light red – medium (middle)
    (180, 255, 180),     # light green – fast (outer)
]

# ------------------------------------------------------
# LOGGING
# ------------------------------------------------------
LOG_DIR = "obstacle_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def main():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Speed-Stratified Orbital Obstacles – Scenario A")

    clock = pygame.time.Clock()
    running = True

    # ------------------------------------------------------
    # INITIALISE OBSTACLES (angles 0°, 120°, 240°)
    # ------------------------------------------------------
    obstacles = [
        {
            "id": "obs_1",
            "radius": INNER_R,
            "angle": 0.0,
            "omega": SLOW_W,
            "colour": COLOURS[0]
        },
        {
            "id": "obs_2",
            "radius": MID_R,
            "angle": 2 * math.pi / 3,     # 120°
            "omega": MID_W,
            "colour": COLOURS[1]
        },
        {
            "id": "obs_3",
            "radius": OUTER_R,
            "angle": 4 * math.pi / 3,     # 240°
            "omega": FAST_W,
            "colour": COLOURS[2]
        }
    ]

    # Grid block size
    block_w = WIDTH / GRID_W
    block_h = HEIGHT / GRID_H

    # ------------------------------------------------------
    # LOG FILE
    # ------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"orbit_speed_stratified_{ts}.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)

    writer.writerow([
        "unix_time",
        "frame_idx",
        "obs_id",
        "x_px", "y_px",
        "cx_px", "cy_px",
        "bx", "by",
        "radius_px",
        "angle_rad",
        "omega_rad_s"
    ])
    log_file.flush()

    frame_idx = 0
    last_time = time.time()

    print("Running speed-stratified orbital obstacle simulation.")
    print("Close the window or press ESC to quit.")

    # ------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------
    try:
        while running:

            now = time.time()
            dt = now - last_time
            dt = max(dt, 1.0 / FPS)
            last_time = now

            # --------------------------
            # Input
            # --------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

            screen.fill(BG_COLOR)

            # --------------------------
            # Update orbits
            # --------------------------
            for obs in obstacles:

                obs["angle"] += obs["omega"] * dt

                x = CX + obs["radius"] * math.cos(obs["angle"])
                y = CY + obs["radius"] * math.sin(obs["angle"])

                obs["x"] = x
                obs["y"] = y

            # --------------------------
            # Render + log
            # --------------------------
            for obs in obstacles:

                x = obs["x"]
                y = obs["y"]

                bx = int(x // block_w)
                by = int(y // block_h)
                bx = max(0, min(GRID_W - 1, bx))
                by = max(0, min(GRID_H - 1, by))

                writer.writerow([
                    now,
                    frame_idx,
                    obs["id"],
                    x,
                    y,
                    x,
                    y,
                    bx,
                    by,
                    obs["radius"],
                    obs["angle"],
                    obs["omega"]
                ])

                pygame.draw.circle(
                    screen,
                    obs["colour"],
                    (int(x), int(y)),
                    OBS_RADIUS
                )

            log_file.flush()
            pygame.display.flip()

            frame_idx += 1
            clock.tick(FPS)

    finally:
        log_file.close()
        pygame.quit()


if __name__ == "__main__":
    main()

