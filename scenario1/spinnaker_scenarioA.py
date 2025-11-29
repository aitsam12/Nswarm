#!/usr/bin/env python3
"""
Scenario A – Multi-Obstacle Neuromorphic SpiNNaker Interface
WITH REAL-TIME GRID VISUALISATION

Every spike prints:
- Obstacle ID
- Grid position (bx,by)
- Neuron ID
- Live 32x24 ASCII grid showing the spiking neuron
"""

import time
import zmq
import re
import csv
import os
import spynnaker8 as Frontend


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
GRID_W = 32
GRID_H = 24
N_NEURONS = GRID_W * GRID_H

RUN_TIME = 60000
WEIGHT_TO_SPIKE = 2.0

PERCEPTION_PORT = 5555
SWARM_PORT = 6000

SMOOTHING = 0.15
MAX_OBSTACLES = 3

LOG_DIR = "spinnaker_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------------------------------------
# NEURON PARAMETERS (LIF)
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
# REAL-TIME ASCII GRID VISUALISATION
# -------------------------------------------------------
def print_grid(gx, gy, obst_id):
    """Print a 32x24 grid showing the neuron that spiked."""

    print("\n[GRID VISUALISATION]")
    for row in range(GRID_H):
        line = ""
        for col in range(GRID_W):

            if col == gx and row == gy:
                # Mark the firing neuron
                # Different colour per obstacle ID
                if obst_id == 1:
                    mark = "1"
                elif obst_id == 2:
                    mark = "2"
                elif obst_id == 3:
                    mark = "3"
                elif obst_id == 4:
                    mark = "4"
                else:
                    mark = "X"
                line += f"{mark} "
            else:
                line += ". "
        print(line)
    print("\n")   # extra spacing


# -------------------------------------------------------
# ZMQ INPUT
# -------------------------------------------------------
context = zmq.Context.instance()

receiver = context.socket(zmq.SUB)
receiver.connect(f"tcp://localhost:{PERCEPTION_PORT}")
receiver.setsockopt_string(zmq.SUBSCRIBE, "")

pattern = re.compile(r"ID:(\d+)\s+Block:\((\d+),(\d+)\);")


# -------------------------------------------------------
# ZMQ OUTPUT (SPINNAKER → SWARM)
# -------------------------------------------------------
swarm_sender = context.socket(zmq.PUB)
swarm_sender.bind(f"tcp://*:{SWARM_PORT}")


# -------------------------------------------------------
# OBSTACLE STATE
# -------------------------------------------------------
obstacles = {}

def smooth(old, new):
    return old * (1 - SMOOTHING) + new * SMOOTHING


# -------------------------------------------------------
# CSV LOGGING
# -------------------------------------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"spinnaker_multi_obstacles_{timestamp}.csv")

log_file = open(log_path, "w", newline="")
log_csv = csv.writer(log_file)

log_csv.writerow([
    "unix_time_ms",
    "sim_time_ms",
    "obstacle_id",
    "grid_x", "grid_y",
    "neuron_id"
])


# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------
def init_pop(label, n_neurons_cb, run_time_ms, timestep_ms):
    print(f"[Init] Population '{label}' initialised with {n_neurons_cb} neurons")


def send_spike_callback(label, live_sender):
    global obstacles

    start = time.perf_counter()

    print("[Callback] Starting multi-obstacle ZMQ→SpiNNaker loop")

    while (time.perf_counter() - start) * 1000 < RUN_TIME:

        try:
            msg = receiver.recv_string(flags=zmq.NOBLOCK)

            for m in pattern.finditer(msg):
                obst_id = int(m.group(1))
                gx = int(m.group(2))
                gy = int(m.group(3))

                if obst_id > MAX_OBSTACLES:
                    continue

                gx = max(0, min(GRID_W - 1, gx))
                gy = max(0, min(GRID_H - 1, gy))

                neuron_id = gy * GRID_W + gx

                unix_ms = int(time.time() * 1000)
                sim_ms = int((time.perf_counter() - start) * 1000)

                # Inject spike
                live_sender.send_spike(label, neuron_id, send_full_keys=True)

                # Log CSV
                log_csv.writerow([unix_ms, sim_ms, obst_id, gx, gy, neuron_id])

                # ----------------------------
                # REAL-TIME PRINTING
                # ----------------------------
                print(
                    f"\n[Spike] Obstacle {obst_id} → Grid=({gx},{gy}) → Neuron={neuron_id}"
                )
                print_grid(gx, gy, obst_id)

                # Update smoothed state
                if obst_id not in obstacles:
                    obstacles[obst_id] = {"x": float(gx), "y": float(gy)}
                else:
                    obstacles[obst_id]["x"] = smooth(obstacles[obst_id]["x"], gx)
                    obstacles[obst_id]["y"] = smooth(obstacles[obst_id]["y"], gy)

        except zmq.Again:
            pass

        # Send smoothed map to SWARM
        if len(obstacles):
            out_list = [(int(pos["x"]), int(pos["y"])) for pos in obstacles.values()]
            swarm_sender.send_string(f"obstacles:{out_list}")

        time.sleep(0.002)

    print("[Callback] Completed multi-obstacle loop")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():

    Frontend.setup(
        timestep=1.0,
        min_delay=1.0,
        max_delay=144.0
    )

    pop = Frontend.Population(
        N_NEURONS,
        Frontend.IF_curr_exp(**cell_params),
        label="grid_pop"
    )

    injector = Frontend.Population(
        N_NEURONS,
        Frontend.external_devices.SpikeInjector(),
        label="grid_spike_injector",
        additional_parameters={
            "port": 12345,
            "virtual_key": 0x80000
        }
    )

    Frontend.Projection(
        injector,
        pop,
        Frontend.OneToOneConnector(),
        Frontend.StaticSynapse(weight=WEIGHT_TO_SPIKE)
    )

    pop.record("spikes")

    live = Frontend.external_devices.SpynnakerLiveSpikesConnection(
        send_labels=["grid_spike_injector"],
        local_port=19999
    )

    live.add_init_callback("grid_spike_injector", init_pop)
    live.add_start_resume_callback("grid_spike_injector", send_spike_callback)

    print(f"[System] Starting SpiNNaker run for {RUN_TIME} ms")

    Frontend.run(RUN_TIME)

    print("[System] Fetching spike data...")
    data = pop.get_data("spikes")

    Frontend.end()
    log_file.close()

    print("[System] SpiNNaker shutdown complete.")


if __name__ == "__main__":
    main()

