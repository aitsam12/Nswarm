#!/usr/bin/env python3
"""
Scenario 2 – Swarm Arena with Predictive Neuromorphic Repulsion
Improved version with:

✔ predictive repulsion
✔ strong direction reversal
✔ temporary speed boost
✔ smooth decay back to normal speed
✔ reliable parsing of multi-ID repel messages
"""

import pygame
import random
import math
import time
import csv
import os
import zmq
import re

# ---------------------------------------------------------
# Arena Parameters
# ---------------------------------------------------------
WIDTH, HEIGHT = 640, 480
FPS = 60

NUM_ROBOTS = 5

ROBOT_SIZE = 16
ROBOT_MIN_SPEED = 1.0
ROBOT_MAX_SPEED = 2.0

WALL_MARGIN = 40
WALL_GAIN = 120.0

GRID_W, GRID_H = 32, 24
CELL_W = WIDTH / GRID_W
CELL_H = HEIGHT / GRID_H

ZMQ_PORT = 6000    # from SpiNNaker side

LOG_DIR = "swarm_log"
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------
# ZMQ Receiver (SpiNNaker → Swarm)
# ---------------------------------------------------------
context = zmq.Context.instance()
receiver = context.socket(zmq.SUB)
receiver.connect(f"tcp://localhost:{ZMQ_PORT}")
receiver.setsockopt_string(zmq.SUBSCRIBE, "")

print(f"[ZMQ] Swarm SUB connected on tcp://localhost:{ZMQ_PORT}")

# Matches single or multiple robot IDs
repel_pattern = re.compile(
    r"repel:ID:([\d,]+);cell:\((\d+),(\d+)\);"
)


# ---------------------------------------------------------
# Star shape (visible)
# ---------------------------------------------------------
def star_points(cx, cy, radius, inner_scale=0.45, points=5):
    pts = []
    ang = 0
    step = math.pi / points
    for i in range(points * 2):
        r = radius if (i % 2 == 0) else (radius * inner_scale)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        ang += step
    return pts


# ---------------------------------------------------------
# Robot
# ---------------------------------------------------------
class Robot:
    def __init__(self, x, y, rid):
        self.x = float(x)
        self.y = float(y)
        self.id = rid
        self.speed = random.uniform(ROBOT_MIN_SPEED, ROBOT_MAX_SPEED)
        self.angle = random.uniform(0, 2 * math.pi)

        self.last_repel_time = 0
        self.repel_boost_timer = 0   # seconds remaining for speed boost

    def random_jitter(self):
        self.angle += random.uniform(-0.15, 0.15)

    def apply_repulsion(self, repel_gx, repel_gy):
        """
        Predictive repulsion:
        Use robot's FUTURE direction, not its current position.
        """

        # predicted future position (10 px ahead)
        future_x = self.x + math.cos(self.angle) * 10
        future_y = self.y + math.sin(self.angle) * 10

        repel_px = repel_gx * CELL_W + CELL_W / 2
        repel_py = repel_gy * CELL_H + CELL_H / 2

        dx = future_x - repel_px
        dy = future_y - repel_py

        # Strong turn away (180 degrees + randomness)
        turn = math.pi + random.uniform(-0.6, 0.6)
        self.angle = math.atan2(dy, dx) + turn

        # temporary speed boost
        self.speed = min(self.speed * 1.6, ROBOT_MAX_SPEED * 2.0)
        self.repel_boost_timer = 0.28   # boost for 0.28 seconds

        self.last_repel_time = time.time()

    def update(self, dt):

        # speed boost decay
        if self.repel_boost_timer > 0:
            self.repel_boost_timer -= dt
        else:
            self.speed = max(ROBOT_MIN_SPEED, self.speed * 0.92)

        fx, fy = 0.0, 0.0

        # Wall avoidance
        if self.x < WALL_MARGIN:
            fx += WALL_GAIN
        elif self.x > WIDTH - WALL_MARGIN:
            fx -= WALL_GAIN

        if self.y < WALL_MARGIN:
            fy += WALL_GAIN
        elif self.y > HEIGHT - WALL_MARGIN:
            fy -= WALL_GAIN

        if fx != 0 or fy != 0:
            self.angle = math.atan2(fy, fx)
        else:
            self.random_jitter()

        # Move forward
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Clamp
        self.x = max(ROBOT_SIZE, min(WIDTH - ROBOT_SIZE, self.x))
        self.y = max(ROBOT_SIZE, min(HEIGHT - ROBOT_SIZE, self.y))

    def draw(self, screen, blink):
        white = (255, 255, 255)

        pygame.draw.polygon(screen, white, star_points(self.x, self.y, ROBOT_SIZE), width=1)

        hx = self.x + math.cos(self.angle) * (ROBOT_SIZE + 6)
        hy = self.y + math.sin(self.angle) * (ROBOT_SIZE + 6)
        pygame.draw.line(screen, white, (self.x, self.y), (hx, hy), 2)

        font = pygame.font.SysFont(None, 16)
        label = font.render(str(self.id), True, white)
        screen.blit(label, (self.x - 6, self.y - 20))

        if blink:
            pygame.draw.circle(screen, white, (int(self.x), int(self.y - ROBOT_SIZE - 5)), 3)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Scenario 2 – Predictive Neuromorphic Swarm")
    clock = pygame.time.Clock()

    robots = []
    for rid in range(NUM_ROBOTS):
        rx = random.randint(ROBOT_SIZE + 10, WIDTH - ROBOT_SIZE - 10)
        ry = random.randint(ROBOT_SIZE + 10, HEIGHT - ROBOT_SIZE - 10)
        robots.append(Robot(rx, ry, rid))

    # Logging
    pos_log = csv.writer(open(os.path.join(LOG_DIR, "swarm_positions.csv"), "w", newline=""))
    pos_log.writerow(["t", "robot_id", "x", "y", "gx", "gy"])

    repel_log = csv.writer(open(os.path.join(LOG_DIR, "swarm_repulsion.csv"), "w", newline=""))
    repel_log.writerow(["t", "robot_id", "repel_gx", "repel_gy"])

    coll_log = csv.writer(open(os.path.join(LOG_DIR, "swarm_collisions.csv"), "w", newline=""))
    coll_log.writerow(["t", "robot_id_1", "robot_id_2"])

    blink = True
    last_blink = time.time()

    running = True
    try:
        while running:

            dt = clock.get_time() / 1000.0  # seconds since last frame

            # -------------------------------------------------
            # Receive repulsion signals
            # -------------------------------------------------
            try:
                msg = receiver.recv_string(flags=zmq.NOBLOCK)
                m = repel_pattern.search(msg)
                if m:
                    repel_ids = [int(r) for r in m.group(1).split(",")]
                    gx = int(m.group(2))
                    gy = int(m.group(3))

                    for r in robots:
                        if r.id in repel_ids:
                            r.apply_repulsion(gx, gy)
                            repel_log.writerow([time.time(), r.id, gx, gy])

            except zmq.Again:
                pass

            # -------------------------------------------------
            # Pygame events
            # -------------------------------------------------
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

            now = time.time()

            if now - last_blink > 1.0:
                blink = not blink
                last_blink = now

            screen.fill((15, 15, 15))

            # -------------------------------------------------
            # Update robot states
            # -------------------------------------------------
            for r in robots:
                r.update(dt)
                r.draw(screen, blink)

                gx = int(r.x // CELL_W)
                gy = int(r.y // CELL_H)
                pos_log.writerow([now, r.id, r.x, r.y, gx, gy])

            # -------------------------------------------------
            # Collision detection (baseline measure)
            # -------------------------------------------------
            for i in range(NUM_ROBOTS):
                for j in range(i + 1, NUM_ROBOTS):
                    dx = robots[i].x - robots[j].x
                    dy = robots[i].y - robots[j].y
                    if math.hypot(dx, dy) < ROBOT_SIZE:
                        coll_log.writerow([now, robots[i].id, robots[j].id])

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        pygame.quit()
        print("[Scenario 2] Swarm simulation ended.")


if __name__ == "__main__":
    main()

