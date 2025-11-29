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

Figures in the paper (overview diagram, Scenario A and B block diagrams, and tracking heatmaps) are generated from the exact pipelines and log files contained here.

---

## Repository Layout

```text
.
├── scenario_A/
│   ├── perception_scenarioA2.py        # Event-based perception + tracking (Scenario A)
│   ├── obstacles_sim.py                # Obstacle / trajectory generator
│   ├── spinnaker_scenarioA.py          # SpiNNaker interface (multi-obstacle, Scenario A)
│   ├── swarm_scenarioA.py              # Swarm / obstacle visualisation
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
│   ├── swarm_scenarioB_baseline.csv    # Baseline swarm positions / collisions
│   ├── swarm_positions_cropped.csv     # Controlled swarm positions, time-aligned
│   └── figures/
│       └── block_scenarioB_cropped.pdf # Block diagram (paper)
│
├── analysis/
│   ├── results_ana.py                  # Main analysis script (matching, latency, collisions)
│   ├── results_ana2.py                 # Variant / additional analysis for Scenario B
│   └── (generated CSV/TXT outputs)     # e.g. spike_match_results.csv, latency_results.csv, …
│
├── overview/
│   └── overview_pic2_cropped.pdf       # System overview figure used in the paper
│
└── README.md

