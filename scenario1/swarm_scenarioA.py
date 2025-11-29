#!/usr/bin/env python3
"""
Scenario A – ULTRA-REALTIME Swarm Simulation with FULL LOGGING

Logs:
- robot positions (px,py)
- robot grid positions
- obstacle grid positions
- obstacle pixel positions
For every frame.

This is essential for ANTS2026 latency + accuracy analysis.
"""

import pygame
import random
import zmq
import re
import math
import time
import csv


# ----------------------------------------------------
# CONFIGURATION (REAL-TIME)
# ----------------------------------------------------
WIDTH, HEIGHT = 640, 480
FPS = 60
NUM_ROBOTS = 5

ROBOT_SIZE = 18
ROBOT_MIN_SPEED = 4.0
ROBOT_MAX_SPEED = 6.2
COMM_RADIUS = 110

SMOOTHING = 0.05     # near-instant, low-latency
# SMOOTHING = 0.0    # truly instant

ZMQ_PORT = 6000

GRID_W, GRID_H = 32, 24
CELL_W = WIDTH // GRID_W
CELL_H = HEIGHT // GRID_H


# ----------------------------------------------------
# ZMQ SETUP – ZERO LATENCY
# ----------------------------------------------------
context = zmq.Context()
socket = context.socket(zmq.SUB)

# Drop backlog → keep only latest
socket.setsockopt(zmq.CONFLATE, 1)

socket.connect(f"tcp://localhost:{ZMQ_PORT}")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Example msg: obstacles:[(10,5),(13,6),(7,8)]
pattern = re.compile(r"obstacles:\[(.*?)\]")


# ----------------------------------------------------
# SMOOTHING
# ----------------------------------------------------
def smooth(old, new):
    return old * (1 - SMOOTHING) + new * SMOOTHING


# ----------------------------------------------------
# ROBOT CLASS
# ----------------------------------------------------
class Robot:
    def __init__(self, x, y, robot_id):
        self.x = float(x)
        self.y = float(y)
        self.id = robot_id
        self.speed = random.uniform(ROBOT_MIN_SPEED, ROBOT_MAX_SPEED)
        self.angle = random.uniform(0, 2 * math.pi)

    def move(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        self.x = max(ROBOT_SIZE, min(WIDTH - ROBOT_SIZE, self.x))
        self.y = max(ROBOT_SIZE, min(HEIGHT - ROBOT_SIZE, self.y))

    def avoid(self, robots, obstacles):
        fx = fy = 0.0

        # Avoid robots
        for other in robots:
            if other is self:
                continue
            dx = self.x - other.x
            dy = self.y - other.y
            dist = math.hypot(dx, dy)
            if 0 < dist < COMM_RADIUS:
                rep = 9000 / (dist ** 2)
                fx += rep * (dx / dist)
                fy += rep * (dy / dist)

        # Avoid obstacles
        for oid, pos in obstacles.items():
            ox = pos["x"] * CELL_W + CELL_W / 2
            oy = pos["y"] * CELL_H + CELL_H / 2

            dx = self.x - ox
            dy = self.y - oy
            dist = math.hypot(dx, dy)
            if dist < COMM_RADIUS:
                rep = 20000 / (dist ** 2)
                fx += rep * (dx / dist)
                fy += rep * (dy / dist)

        # Wall avoidance
        margin = 40
        if self.x < margin: fx += 120
        elif self.x > WIDTH - margin: fx -= 120
        if self.y < margin: fy += 120
        elif self.y > HEIGHT - margin: fy -= 120

        if fx != 0 or fy != 0:
            self.angle = math.atan2(fy, fx)

        self.move()

    def draw(self, screen, blink_state):
        white = (255, 255, 255)

        pts = []
        ang = 0
        for i in range(10):
            r = ROBOT_SIZE if i % 2 == 0 else (ROBOT_SIZE * 0.4)
            x = self.x + r * math.cos(ang)
            y = self.y + r * math.sin(ang)
            pts.append((x, y))
            ang += math.pi / 5
        pygame.draw.polygon(screen, white, pts, width=1)

        hx = self.x + math.cos(self.angle) * (ROBOT_SIZE + 6)
        hy = self.y + math.sin(self.angle) * (ROBOT_SIZE + 6)
        pygame.draw.line(screen, white, (self.x, self.y), (hx, hy), 2)

        font = pygame.font.SysFont(None, 16)
        label = font.render(str(self.id), True, white)
        screen.blit(label, (self.x - 6, self.y - 20))

        if blink_state:
            pygame.draw.circle(
                screen, white,
                (int(self.x), int(self.y - ROBOT_SIZE - 5)),
                4
            )


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ultra-Realtime Swarm Simulation")
    clock = pygame.time.Clock()

    robots = [
        Robot(
            random.randint(60, WIDTH - 60),
            random.randint(60, HEIGHT - 60),
            i
        ) for i in range(NUM_ROBOTS)
    ]

    # Persistent, smoothed obstacle positions
    obstacles = {}

    # ------------------------------------------------
    # LOGGING (robots + obstacles)
    # ------------------------------------------------
    logfile = open("trytryswarm_full_log.csv", "w", newline="")
    logger = csv.writer(logfile)

    logger.writerow([
        "timestamp",
        "robot_id",
        "robot_x", "robot_y",
        "robot_gx", "robot_gy",
        "obstacle_id",
        "ob_gx", "ob_gy",
        "ob_px", "ob_py"
    ])

    blink_state = True
    last_blink = time.time()
    running = True

    while running:

        screen.fill((20, 20, 20))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        if time.time() - last_blink >= 1.0:
            blink_state = not blink_state
            last_blink = time.time()

        # ------------------------------------------------
        # ZMQ RECEIVE — zero-latency mode
        # ------------------------------------------------
        last_msg = None

        while True:
            try:
                last_msg = socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        if last_msg:
            m = pattern.search(last_msg)
            if m:
                pairs = re.findall(r"\((\d+),\s*(\d+)\)", m.group(1))
                obs_list = [(int(bx), int(by)) for bx, by in pairs]

                # Smooth update
                for oid, (bx, by) in enumerate(obs_list, start=1):
                    if oid not in obstacles:
                        obstacles[oid] = {"x": float(bx), "y": float(by)}
                    else:
                        obstacles[oid]["x"] = smooth(obstacles[oid]["x"], bx)
                        obstacles[oid]["y"] = smooth(obstacles[oid]["y"], by)

        # ------------------------------------------------
        # ROBOT UPDATES + LOGGING
        # ------------------------------------------------
        timestamp = time.time()

        for r in robots:

            gx = int(r.x // CELL_W)
            gy = int(r.y // CELL_H)

            # Log robot + ALL obstacles for analysis
            for oid, pos in obstacles.items():
                ob_gx = pos["x"]
                ob_gy = pos["y"]
                ob_px = ob_gx * CELL_W + CELL_W / 2
                ob_py = ob_gy * CELL_H + CELL_H / 2

                logger.writerow([
                    timestamp,
                    r.id,
                    r.x, r.y,
                    gx, gy,
                    oid,
                    ob_gx, ob_gy,
                    ob_px, ob_py
                ])

            r.avoid(robots, obstacles)
            r.draw(screen, blink_state)

        # ------------------------------------------------
        # DRAW OBSTACLES
        # ------------------------------------------------
        for oid, pos in obstacles.items():
            px = pos["x"] * CELL_W
            py = pos["y"] * CELL_H
            pygame.draw.rect(
                screen,
                (220, 60, 60),
                pygame.Rect(px, py, CELL_W, CELL_H)
            )

        pygame.display.flip()
        clock.tick(FPS)

    logfile.close()
    pygame.quit()


if __name__ == "__main__":
    main()

