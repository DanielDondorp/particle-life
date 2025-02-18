#!/usr/bin/env python3
"""
Entry point for the Particle Life Simulation.
Run this script to start the simulation.
"""

import arcade
from src.main import ParticleLifeWindow

if __name__ == "__main__":
    app = ParticleLifeWindow()
    arcade.run() 