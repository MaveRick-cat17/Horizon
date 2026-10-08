\# Horizon 🌌

\### Discovering Physical Laws from Pixels



Horizon is a computer-vision and sparse-regression system that attempts to discover governing physical equations directly from video.



Instead of manually programming the physics of a system, Horizon observes its motion, extracts the dynamics, and uses Sparse Identification of Nonlinear Dynamics (SINDy) to discover a mathematical model.



\---



\## 🚀 Core Idea



\*\*Video → Motion → Dynamics → Equation → Prediction\*\*



```text

🎥 Video

&#x20;  ↓

👁️ Computer Vision

&#x20;  ↓

📍 Object Tracking

&#x20;  ↓

📈 x(t), y(t)

&#x20;  ↓

📐 θ(t)

&#x20;  ↓

🧮 Numerical Differentiation

&#x20;  ↓

🔬 SINDy

&#x20;  ↓

📜 Governing Equation

&#x20;  ↓

🚀 Forward Simulation

&#x20;  ↓

📊 Prediction vs Reality


🧪 Current Demonstration: Simple Pendulum
The current prototype uses a video of a simple pendulum.
Horizon:
1. Detects the pendulum bob.
2. Detects the pivot.
3. Tracks the bob throughout the video.
4. Converts the trajectory into angular displacement.
5. Calculates angular velocity and acceleration.
6. Uses physics-constrained SINDy to identify the governing dynamics.
7. Uses the discovered equation for forward simulation.
🔬 Discovered Physics
From the video data, Horizon discovered:
\[
\ddot{\theta}
=
-28.334142\sin(\theta)
-
0.040713\dot{\theta}
\]
This corresponds to a damped pendulum model:
\[
\ddot{\theta}
=
-\frac{g}{L}\sin(\theta)
-c\dot{\theta}
\]
Experimental Validation
Quantity	Result
Video samples	733
Sampling frequency	30 Hz
Angle range	−19.05° to +19.23°
Measured period	1.1667 s
Theoretical g/L	29.0046 s⁻²
Horizon estimate	28.3341 s⁻²
Relative difference	2.31%


The discovered dynamics reproduce the observed oscillatory behaviour and include a damping term corresponding to the gradual reduction in amplitude observed in the video.
🚀 Forward Simulation
After discovering the equation, Horizon uses the discovered dynamics to simulate the pendulum forward in time.
The simulation is initialized using the observed angular position and velocity.
This allows Horizon to compare:
Observed motion
vs.
Motion predicted by the discovered equation
The current prototype achieves a long-horizon angular RMSE of approximately 5.42° over the available trajectory. The accumulated error is primarily associated with phase drift over time.
📊 Results
The project generates:
- Corrected pendulum angle θ(t)
- Angular velocity θ̇(t)
- Angular acceleration θ̈(t)
- SINDy-discovered dynamics
- Measured vs discovered acceleration
- Measured vs forward-simulated motion
- Phase-space representation
🛠️ Technology Stack
- Python
- OpenCV
- NumPy
- SciPy
- PySINDy
- Matplotlib
📁 Project Structure
Horizon/
│
├── data/
│   └── Videos/
│
├── src/
│   ├── pendulum_tracker_v12.py
│   ├── discover_dynamics.py
│   ├── simulate_dynamics.py
│   └── ...
│
├── horizon_pendulum_v12.npz
├── horizon_sindy_results.npz
├── horizon_forward_simulation.npz
│
├── horizon_sindy_acceleration.png
├── horizon_phase_space.png
├── horizon_forward_simulation.png
└── horizon_forward_phase_space.png

🧠 Why Horizon?
Traditional physics modelling usually starts with a known physical system and manually derives its governing equations.
Horizon explores the reverse direction:
Start with observations and discover the equations.

The pendulum is the first proof-of-concept.
The long-term goal is to extend Horizon to other dynamical systems such as:
- Bouncing objects
- Spring-mass systems
- Rotational systems
- Coupled oscillators
- More complex physical systems
🔭 Future Work
Future versions of Horizon aim to:
- Automatically identify different types of physical systems
- Reduce dependence on manual calibration
- Improve robust object tracking
- Discover more complex nonlinear dynamics
- Support multiple interacting objects
- Automatically select appropriate mathematical libraries
- Perform real-time physics discovery from video

👥 Project
Horizon
Built as a prototype for a student hackathon.
Core Concept
Discover physics from pixels.


### Then save it

In PowerShell:

```powershell
notepad README.md
