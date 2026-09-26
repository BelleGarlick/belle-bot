# Isaac Sim Simulation

This directory contains simulation scripts for Isaac Sim to test Belle Bot's vision and mapping algorithms in a virtual environment.

## Quick Start: Hello World Simulation

The `hello_world.py` script generates a random room with walls and obstacles and sets up a depth camera to capture RGB and depth data.

### Prerequisites

- **Isaac Sim:** You must have NVIDIA Isaac Sim installed via the Omniverse Launcher.
- **NVIDIA GPU:** A compatible NVIDIA GPU with the latest drivers is required.

### How to Run

Isaac Sim uses a specialized Python environment. You must use the `python.sh` (Linux) or `python.bat` (Windows) wrapper located in your Isaac Sim installation folder.

1.  **Locate your Isaac Sim installation path.** 
    *   Typical Linux path: `~/.local/share/ov/pkg/isaac-sim-4.2.0/`
2.  **Run the script using the Isaac Sim Python wrapper:**

    ```bash
    # From the project root
    /path/to/isaac-sim/python.sh simulation/hello_world.py
    ```

### Running Headless

If you are running on a server without a monitor, you can run the simulation in headless mode. 

In `simulation/hello_world.py`, update the `SimulationApp` configuration:

```python
simulation_app = SimulationApp({"headless": True})
```

### Features
- **Random Room Generation:** Creates a floor, outer walls, and a random number of internal obstacles.
- **Depth Camera:** A camera prim is created at `/World/Camera` and captures:
    - **RGBA:** 4-channel color images.
    - **Depth:** 32-bit float distance maps (in meters).
- **Standalone Execution:** Uses `omni.isaac.kit.SimulationApp` for running outside of the full Isaac Sim GUI.
