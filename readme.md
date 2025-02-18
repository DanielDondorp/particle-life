# Particle Life Simulation

## Overview
This project implements a GPU-accelerated particle simulation inspired by artificial life systems. Different types of particles interact with each other based on attraction and repulsion rules, creating emergent behaviors and patterns. The simulation leverages compute shaders for efficient parallel processing, allowing for real-time interaction with large numbers of particles.

## Features

### Core Mechanics
- GPU-accelerated particle simulation using OpenGL compute shaders
- Support for multiple particle types (up to 5 types with distinct colors)
- Toroidal world space (particles wrap around screen edges)
- Random interaction matrix defining attraction/repulsion between particle types
- Real-time parameter adjustment through an intuitive UI

### Interactive Controls
The simulation includes a comprehensive control panel that allows real-time adjustment of:

#### Radius Parameters
- **Interaction Radius** (50-400): Maximum distance at which particles can influence each other
- **Repulsion Radius** (10-100): Distance at which particles start repelling each other

#### Force Parameters
- **Repulsion Strength** (0-10): Intensity of repulsion force between particles
- **Attraction Strength** (0-5): Intensity of attraction force between particles
- **Max Force** (0.1-2.0): Maximum force that can be applied to a particle

#### Movement Parameters
- **Max Speed** (1-20): Maximum velocity a particle can achieve
- **Friction** (0.5-0.99): Velocity dampening factor

A "Reset to Defaults" button allows quick restoration of original parameters.

### Video Recording
- Press Ctrl+R to start/stop recording
- Recordings are automatically saved in MP4 format
- Filename includes timestamp and simulation parameters

## Requirements

### System Requirements
- OpenGL 4.3 or higher (for compute shader support)
- GPU with compute shader capabilities
- Windows/Linux/MacOS with appropriate graphics drivers

### Dependencies
```python
arcade>=3.0.0
numpy>=1.24.0
seaborn>=0.12.0
opencv-python>=4.8.0
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the simulation:
```bash
python run_simulation.py
```

2. Interact with the simulation:
   - Use the control panel on the right to adjust parameters in real-time
   - Observe how different parameters affect particle behavior
   - Experiment with various combinations to create interesting patterns

3. Record interesting behaviors:
   - Press Ctrl+R to start recording
   - Press Ctrl+R again to stop and save the recording
   - Find recordings in the `videos` directory

## Tips for Interesting Behaviors

- Try high attraction strength with low repulsion for clustering
- Increase max speed and reduce friction for more chaotic movement
- Use large interaction radius with low forces for subtle, long-range effects
- Experiment with different combinations to discover unique patterns

## Technical Details

### Particle Properties
- Position (2D)
- Velocity (2D)
- Color (determines particle type)
- Radius

### Performance
- Processes particles in batches of 256 (compute shader work group size)
- Efficient parallel computation of particle interactions
- Optimized for real-time interaction with large particle counts (30,000+ particles)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New features or improvements
- Bug fixes
- Documentation updates
- Performance optimizations

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details.

This means you are free to:
- Use this software for any purpose
- Change the software to suit your needs
- Share the software with your friends and neighbors
- Share the changes you make

Under the following conditions:
- If you distribute this software or any derivative works, you must:
  - Make the source code available
  - License it under GPL-3.0
  - State your changes
  - Include the original copyright and license notices 