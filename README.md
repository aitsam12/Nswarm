# Nswarm

> **Real-Time Neuromorphic Sensing and Spiking Attractor Dynamics for Coordinated Multi-Agent Behaviour**  
> Muhammad Aitsam, Alessandro Di Nuovo, *ANTS 2026 (under review)*

The repository implements two experimental scenarios:

- **Scenario A – Multi-Obstacle Neuromorphic Tracking**  
- **Scenario B – Vision-Driven Swarm Stability (Collision Suppression)**  

Both scenarios use a common architecture:

1. **Swarm / Obstacle Simulator** (pygame-based 2-D arena)  
2. **Event-Based Perception Pipeline** (event camera + fast blob detection + multi-object tracking)  
3. **SpiNNaker Interface** (grid encoding, spike injection, attractor dynamics, repulsion output)  

## System Overview

![System Overview](overview_pic2_cropped.png)

*Figure 1: End-to-end neuromorphic perception–SpiNNaker–swarm control pipeline used in the experiments.*


---

## Repository Layout

```text
.
├── scenario_A/
│   ├── perception_scenarioA2.py        # Event-based perception + tracking (Scenario A)
│   ├── obstacles_sim.py                # Obstacle/trajectory generator
│   ├── spinnaker_scenarioA.py          # SpiNNaker interface (multi-obstacle, Scenario A)
│   ├── swarm_scenarioA.py              # Swarm/obstacle visualisation
│   ├── fast_blobs*.so                  # Precompiled fast blob detector
│   ├── trackers_log.csv                # Perception-side tracking log (grid + centroids)
│   ├── spinnaker_multi_obstacles_*.csv # SpiNNaker spike log (Scenario A)
│   ├── swarm_full_log.csv              # Swarm log (Scenario A)
│   ├── orbit_speed_stratified_*.csv    # Ground-truth obstacle orbits
│   └── figures/
│       ├── block_scenarioA_cropped.pdf # Block diagram (paper)
│       └── good_heatmap.png            # GT vs perceived heatmap (paper)
│
├── scenario_B/
│   ├── perception_scenarioB2.py        # Event-based perception + tracking (Scenario B)
│   ├── spinnaker_scenarioB.py          # SpiNNaker interface (swarm repulsion)
│   ├── swarm_scenarioB.py              # Swarm simulation (baseline + controlled)
│   ├── fast_blobs*.so                  # Precompiled fast blob detector
│   ├── swarm_scenarioB_baseline.csv    # Baseline swarm positions/collisions
│   ├── swarm_positions_cropped.csv     # Controlled swarm positions, time-aligned
│   └── figures/
│       └── block_scenarioB_cropped.pdf # Block diagram (paper)
│
│
└── README.md


## Dependencies

The codebase assumes the following environment and libraries:

- **Python 3.12**
- **Metavision SDK** used in:
  - `perception_scenarioA2.py`
  - `perception_scenarioB2.py`
- **Standard scientific Python stack**:
  - NumPy  
  - OpenCV  
  - pandas  
  - matplotlib  
  - pyzmq  
  - pygame
- **spynnaker8** and a **configured SpiNNaker board** for live spike injection (Python 3.8)

Exact library versions and hardware configuration depend on your local SpiNNaker installation and event-camera setup. Please refer to the docstrings at the top of each script for scenario-specific details such as ports, grid dimensions and timing parameters.

---

## How to Use This Repository

At a high level, each experimental run follows the same execution pattern:

1. **Start the swarm/obstacle simulator**  
   - `swarm_scenarioA.py` (Scenario A)  
   - `swarm_scenarioB.py` (Scenario B)

2. **Start the perception script** so that ZMQ messages containing grid positions are published:
   - `perception_scenarioA2.py`
   - `perception_scenarioB2.py`

3. **Start the SpiNNaker interface** to receive grid positions, inject spikes, and (for Scenario B) send repulsion commands:
   - `spinnaker_scenarioA.py`
   - `spinnaker_scenarioB.py`

4. **After the experiment completes**, run the relevant analysis scripts in the `analysis/` directory to reproduce:
   - Spike–robot matching accuracy  
   - End-to-end latency measurements  
   - Collision-rate reduction statistics  
   - Nearest-neighbour distance metrics

The exact run order, timing parameters and configuration values should match those described in the paper’s experimental section.



