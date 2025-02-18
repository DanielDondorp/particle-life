# Particle Interaction Simulation

## Overview
This project implements a GPU-accelerated particle simulation system where different types of particles interact with each other based on defined rules. The simulation uses compute shaders for efficient parallel processing of particle behaviors.

## Technical Requirements

### Core Features
- GPU-based particle simulation using OpenGL compute shaders
- Support for multiple particle types (up to 8 types based on RGB combinations)
- Toroidal world space (particles wrap around screen edges)
- Configurable interaction matrix defining attraction/repulsion between particle types

### Particle Properties
- Position (2D)
- Velocity
- Color (determines particle type)
- Radius

### Physics Parameters
- Interaction radius: 50.0 units
- Repulsion radius: 10.0 units
- Maximum force: 0.6 units
- Maximum speed: 10.0 units
- Friction coefficient: 0.9

### Performance
- Processes particles in batches of 256 (compute shader work group size)
- Efficient parallel computation of particle interactions

### Input Requirements
- Screen dimensions
- Number of particles
- Number of particle types
- Interaction matrix (256 elements defining type relationships)

## Dependencies
- OpenGL 4.3 or higher (for compute shader support)
- GPU with compute shader capabilities 