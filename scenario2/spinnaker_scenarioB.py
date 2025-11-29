#!/usr/bin/env python3
"""
Scenario 2 – Vision-Driven Swarm Stability (Collision Suppression)
Updated for:
    - Velocity estimation
    - Predictive grid cell
    - Immediate repulsion (host-side)
    - Full support for messages:
          ID:<id> Block:(bx,by) Vel:(vx,vy) Pred:(pbx,pby);

This version:
    • Does NOT rely on live-output for repulsion (avoids latency)
    • Uses host-side real-time logic to detect collisions & near-collisions
    • Sends repulsion instantly back to the swarm
    • Injects spikes into SpiNNaker for neural logging / heatmaps

"""

import time
import zmq
import re
import csv
import os
import threading
import spynnaker8 as Frontend


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
GRID_W = 32
GRID_H = 24
N_NEURONS = GRID_W * GRID_H

RUN_TIME = 60000      # ms
WEIGHT_TO_SPIKE = 5.0

PERCEPTION_PORT = 5555
SWARM_PORT = 6000

MAX_ROBOTS = 5
COLLISION_GRID_RADIUS_CURR = 1
COLLISION_GRID_RADIUS_PRED = 2

SMOOTHING = 0.12

LOG_DIR = "spinnaker_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------------------------------------
# NEURON PARAMETERS
# -------------------------------------------------------
cell_params = {
    'cm': 0.25,
    'i_offset': 0.0,
    'tau_m': 20.0,
    'tau_refrac': 2.0,
    'tau_syn_E': 5.0,
    'tau_syn_I': 5.0,
    'v_reset': -70.0,
    'v_rest': -65.0,
    'v_thresh': -50.0
}


# -------------------------------------------------------
# ZMQ INPUT (PERCEPTION → SPI)
# -------------------------------------------------------
context = zmq.Context.instance()

receiver = context.socket(zmq.SUB)
receiver.connect(f"tcp://localhost:{PERCEPTION_PORT}")
receiver.setsockopt_string(zmq.SUBSCRIBE, "")

print(f"[ZMQ] SUB connected to tcp://localhost:{PERCEPTION_PORT}")

time.sleep(0.5)

# FULL PATTERN:
# ID:1 Block:(12,6) Vel:(20.1,-5.4) Pred:(13,5);
pattern = re.compile(
    r"ID:(\d+)\s+Block:\((\d+),(\d+)\)\s+Vel:\(([-\d.]+),([-\d.]+)\)\s+Pred:\((\d+),(\d+)\);"
)


# -------------------------------------------------------
# ZMQ OUTPUT (SPI → SWARM)
# -------------------------------------------------------
swarm_sender = context.socket(zmq.PUB)
swarm_sender.bind(f"tcp://*:{SWARM_PORT}")
print(f"[ZMQ] PUB bound to tcp://*:{SWARM_PORT}")


# -------------------------------------------------------
# ROBOT STATE
# -------------------------------------------------------
robots = {}  
"""
robots = {
    robot_id : {
        "x": current grid x,
        "y": current grid y,
        "vx": velocity x,
        "vy": velocity y,
        "px": predicted grid x,
        "py": predicted grid y
    }
}
"""

def smooth(a, b):
    return (1 - SMOOTHING) * a + SMOOTHING * b


# -------------------------------------------------------
# CSV LOGGING
# -------------------------------------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"spinnaker_scenario2_vel_pred_{timestamp}.csv")
log_file = open(log_path, "w", newline="")
log_csv = csv.writer(log_file)

log_csv.writerow([
    "unix_time_ms",
    "sim_time_ms",
    "robot_id",
    "gx",
    "gy",
    "vx",
    "vy",
    "pgx",
    "pgy",
    "neuron_id",
    "event"   # inject / repel
])


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def neuron_id_from_grid(gx, gy):
    return gy * GRID_W + gx


def robots_close_to_point(x, y, radius):
    """Return robots whose current grid location is within Manhattan radius."""
    close = []
    for rid, st in robots.items():
        if abs(int(st["x"]) - x) + abs(int(st["y"]) - y) <= radius:
            close.append(rid)
    return close


def robots_close_to_pred_point(px, py, radius):
    """Return robots whose predicted grid is close."""
    close = []
    for rid, st in robots.items():
        if abs(int(st["px"]) - px) + abs(int(st["py"]) - py) <= radius:
            close.append(rid)
    return close


def send_repulsion(robot_ids, gx, gy, unix_ms, sim_ms):
    if not robot_ids:
        return

    id_str = ",".join(str(r) for r in robot_ids)
    msg = f"repel:ID:{id_str};cell:({gx},{gy});"
    swarm_sender.send_string(msg)

    print(f"[REPEL] → robots {id_str} from cell ({gx},{gy})")

    for rid in robot_ids:
        log_csv.writerow([
            unix_ms, sim_ms, rid,
            gx, gy,
            robots[rid]["vx"], robots[rid]["vy"],
            robots[rid]["px"], robots[rid]["py"],
            neuron_id_from_grid(gx, gy),
            "repel"
        ])


# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------
def init_pop(label, n_neurons_cb, run_time_ms, timestep_ms):
    print(f"[Init] Population {label} ready.")


def send_spike_callback(label, live_sender):
    """
    When SpiNNaker simulation starts, a background thread will
    read ZMQ messages and inject spikes.
    """
    def loop():
        start = time.perf_counter()
        print("[Thread] Injection loop running...")

        while (time.perf_counter() - start) * 1000 < RUN_TIME:

            try:
                msg = receiver.recv_string(flags=zmq.NOBLOCK)

                m = pattern.search(msg)
                if not m:
                    continue

                robot_id = int(m.group(1))
                gx = int(m.group(2))
                gy = int(m.group(3))

                vx = float(m.group(4))
                vy = float(m.group(5))
                pgx = int(m.group(6))
                pgy = int(m.group(7))

                if robot_id > MAX_ROBOTS:
                    continue

                # Bounds
                gx = max(0, min(GRID_W - 1, gx))
                gy = max(0, min(GRID_H - 1, gy))
                pgx = max(0, min(GRID_W - 1, pgx))
                pgy = max(0, min(GRID_H - 1, pgy))

                nid = neuron_id_from_grid(gx, gy)

                unix_ms = int(time.time() * 1000)
                sim_ms = int((time.perf_counter() - start) * 1000)

                # Inject spike into SpiNNaker
                live_sender.send_spike(label, nid, send_full_keys=True)

                # Update robot state
                if robot_id not in robots:
                    robots[robot_id] = {
                        "x": gx, "y": gy,
                        "vx": vx, "vy": vy,
                        "px": pgx, "py": pgy
                    }
                else:
                    robots[robot_id]["x"] = smooth(robots[robot_id]["x"], gx)
                    robots[robot_id]["y"] = smooth(robots[robot_id]["y"], gy)
                    robots[robot_id]["vx"] = vx
                    robots[robot_id]["vy"] = vy
                    robots[robot_id]["px"] = pgx
                    robots[robot_id]["py"] = pgy

                # Log injection
                log_csv.writerow([
                    unix_ms, sim_ms,
                    robot_id,
                    gx, gy,
                    vx, vy,
                    pgx, pgy,
                    nid,
                    "inject"
                ])

                # ------------------------------------------------------
                # COLLISION AVOIDANCE USING:
                #   • CURRENT GRID POSITION
                #   • PREDICTED GRID POSITION
                # ------------------------------------------------------

                # 1. Immediate neighbourhood check
                close_curr = robots_close_to_point(gx, gy, COLLISION_GRID_RADIUS_CURR)

                if len(close_curr) >= 2:
                    send_repulsion(close_curr, gx, gy, unix_ms, sim_ms)

                # 2. Predictive neighbourhood check
                close_pred = robots_close_to_pred_point(pgx, pgy, COLLISION_GRID_RADIUS_PRED)

                if len(close_pred) >= 2:
                    send_repulsion(close_pred, pgx, pgy, unix_ms, sim_ms)

            except zmq.Again:
                pass

            time.sleep(0.005)   # fast loop

        print("[Thread] Injection loop ended.")

    th = threading.Thread(target=loop, daemon=True)
    th.start()


def receive_spike_callback(label, time_ms, neuron_ids):
    """
    Live spikes from SpiNNaker (not required for repulsion).
    """
    pass


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():

    Frontend.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

    # Main attractor sheet
    pop = Frontend.Population(
        N_NEURONS,
        Frontend.IF_curr_exp(**cell_params),
        label="grid_pop"
    )

    # simple lateral connectivity
    conn = []
    w = 0.4
    for y in range(GRID_H):
        for x in range(GRID_W):
            src = neuron_id_from_grid(x, y)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                        conn.append((src, neuron_id_from_grid(nx, ny), w, 1.0))

    Frontend.Projection(
        pop, pop,
        Frontend.FromListConnector(conn),
        Frontend.StaticSynapse(weight=w)
    )

    # Injector
    injector = Frontend.Population(
        N_NEURONS,
        Frontend.external_devices.SpikeInjector(),
        label="grid_spike_injector",
        additional_parameters={"port": 12345, "virtual_key": 0x80000}
    )

    Frontend.Projection(
        injector, pop,
        Frontend.OneToOneConnector(),
        Frontend.StaticSynapse(weight=WEIGHT_TO_SPIKE)
    )

    pop.record("spikes")

    live = Frontend.external_devices.SpynnakerLiveSpikesConnection(
        send_labels=["grid_spike_injector"],
        receive_labels=["grid_pop"]
    )

    live.add_init_callback("grid_spike_injector", init_pop)
    live.add_start_resume_callback("grid_spike_injector", send_spike_callback)
    live.add_receive_callback("grid_pop", receive_spike_callback)

    # Live output not strictly needed but kept for completeness
    Frontend.external_devices.activate_live_output_for(pop, database_notify_port_num=live.local_port)

    print("[System] Running...")
    Frontend.run(RUN_TIME)

    print("[System] Fetching spike data...")
    data = pop.get_data("spikes")

    Frontend.end()
    log_file.close()
    print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()

