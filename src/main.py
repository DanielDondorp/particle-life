"""
Particle Life Simulation

A GPU-accelerated particle simulation that demonstrates emergent behavior through
particle interactions. The simulation uses compute shaders for efficient parallel
processing and provides real-time control over simulation parameters through an
interactive UI.

The simulation features multiple particle types that interact based on a randomly
generated interaction matrix, creating complex and interesting patterns. Particles
move in a toroidal space (wrapping around screen edges) and respond to various
forces including attraction, repulsion, and friction.

Key Features:
- GPU-accelerated particle simulation using OpenGL compute shaders
- Multiple particle types with distinct behaviors
- Real-time parameter adjustment through UI controls
- Video recording capabilities
- Toroidal world space
"""
import random
import numpy as np
from array import array
from typing import Generator, Tuple, Optional
import seaborn as sns
import arcade
from arcade.gl import BufferDescription
from pathlib import Path
import cv2
import time
from datetime import datetime
from arcade.gui import UIManager, UISlider, UILabel, UIBoxLayout, UIAnchorLayout


# Window dimensions
WINDOW_WIDTH = 1300  # Increased to accommodate controls
WINDOW_HEIGHT = 1000
CONTROLS_WIDTH = 300  # Width of the control panel

# Simulation canvas dimensions (actual particle area)
CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 1000

# Performance graph dimensions 
GRAPH_WIDTH = 200
GRAPH_HEIGHT = 120
GRAPH_MARGIN = 5

# Simulation parameters
N_PARTICLE_TYPES: int = 5 # DO NOT CHANGE THIS VALUE
NUM_PARTICLES: int = 35000

# Add simulation parameters that will be controllable
class SimulationParams:
    """
    Encapsulates all adjustable parameters for the particle simulation.
    
    This class serves as a central configuration point for the simulation,
    storing parameters that control particle behavior, interactions, and
    movement characteristics.
    
    Attributes:
        interaction_radius (float): Maximum distance for particle interactions (50-400)
        repulsion_radius (float): Distance where repulsion begins (10-100)
        repulsion_strength (float): Intensity of repulsion force (0-10)
        attraction_strength (float): Intensity of attraction force (0-5)
        max_force (float): Maximum force magnitude allowed (0.1-2.0)
        max_speed (float): Maximum velocity magnitude allowed (1-20)
        friction (float): Velocity dampening factor (0.5-0.99)
    """
    def __init__(self):
        # Interaction parameters
        self.interaction_radius: float = 200.0    # Maximum radius for interaction
        self.repulsion_radius: float = 50.0       # Distance where repulsion starts
        self.repulsion_strength: float = 4.0      # Base strength of repulsion force
        self.attraction_strength: float = 1.0     # Base strength of attraction force
        
        # Movement parameters
        self.max_force: float = 0.7              # Maximum force magnitude
        self.max_speed: float = 10.0             # Maximum velocity magnitude
        self.friction: float = 0.9               # Velocity dampening


def gen_initial_data(
        screen_size: Tuple[int, int],
        num_particles: int = NUM_PARTICLES,
        num_particle_types: int = N_PARTICLE_TYPES,
        palette: str = "husl"  # Good for distinguishable colors
) -> array:
    """
    Generate initial particle data for the simulation.
    
    Creates an array of particle data including positions, velocities, and colors.
    Particles are randomly positioned within the screen bounds with random initial
    velocities. Each particle is assigned one of the available particle types,
    represented by a distinct color. The type ID is stored in the alpha channel.
    
    Args:
        screen_size (Tuple[int, int]): Width and height of the simulation area
        num_particles (int): Total number of particles to generate
        num_particle_types (int): Number of different particle types
        palette (str): Color palette name (unused, using fixed colors)
    
    Returns:
        array: Flat array of particle data in the format:
              [pos.x, pos.y, pos.z, radius, vel.x, vel.y, vel.z, unused,
               color.r, color.g, color.b, type_id] * num_particles
    """
    width, height = screen_size
    
    # Use a colormap to generate colors for each particle type
    cmap = sns.color_palette(palette, num_particle_types)
    
    def _data_generator() -> Generator[float, None, None]:
        particles_per_type = num_particles // num_particle_types
        remaining = num_particles % num_particle_types
        
        for type_id in range(num_particle_types):
            # Calculate how many particles of this type to generate
            count = particles_per_type + (1 if type_id < remaining else 0)
            
            for _ in range(count):
                # Position/radius
                yield random.randrange(0, width)
                yield random.randrange(0, height)
                yield 0.0
                yield 3.0  # Smaller radius for particles

                # Velocity
                yield random.uniform(-1.0, 1.0)
                yield random.uniform(-1.0, 1.0)
                yield 0.0
                yield 0.0

                # Color (represents particle type)
                r, g, b = cmap[type_id]
                yield r
                yield g
                yield b
                yield float(type_id)  # Store type_id in alpha channel

    # Use the generator function to fill an array in RAM
    return array('f', _data_generator())

class ParticleLifeWindow(arcade.Window):
    """
    Main window class for the particle life simulation.
    
    This class handles the initialization of OpenGL resources, shader programs,
    UI elements, and manages the main simulation loop. It provides real-time
    control over simulation parameters through an interactive UI and supports
    video recording of the simulation.
    
    The simulation uses compute shaders for efficient parallel processing of
    particle interactions and updates. Particles move in a toroidal space,
    wrapping around screen edges, and interact based on their types and
    distances from each other.
    
    Attributes:
        params (SimulationParams): Container for all simulation parameters
        ui_manager (UIManager): Manages UI elements and interactions
        recording (bool): Whether video recording is active
        compute_shader (ComputeShader): Shader program for particle updates
        interaction_matrix (ndarray): Matrix defining particle type interactions
    """

    def __init__(self):
        """
        Initialize the simulation window and all required resources.
        
        Sets up the OpenGL context, initializes simulation parameters,
        creates UI elements, and prepares shader programs and buffers
        for particle simulation. Also generates a random interaction
        matrix for particle type relationships.
        """
        # Ask for OpenGL context supporting version 4.3 or greater when
        # calling the parent initializer to make sure we have compute shader
        # support.
        super().__init__(
            WINDOW_WIDTH, WINDOW_HEIGHT,
            "Particle Life",
            gl_version=(4, 3),
            resizable=True,
            update_rate=1/30,
        )
        
        # Initialize simulation parameters
        self.params = SimulationParams()
        
        # Initialize UI manager
        self.ui_manager = UIManager()
        self.ui_manager.enable()
        self.setup_ui()
        
        # Video recording properties
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.frame_count = 0
        
        # Center window
        self.center_window()

        # --- Create buffers

        # Create pairs of buffers for the compute & visualization shaders.
        # We will swap which buffer instance is the initial value and
        # which is used as the current value to write to.

        # ssbo = shader storage buffer object
        initial_data = gen_initial_data((CANVAS_WIDTH, CANVAS_HEIGHT), num_particle_types=N_PARTICLE_TYPES)
        self.ssbo_previous = self.ctx.buffer(data=initial_data)
        self.ssbo_current = self.ctx.buffer(data=initial_data)

        # vao = vertex array object
        # Format string describing how to interpret the SSBO buffer data.
        # 4f = position and size -> x, y, z, radius
        # 4x4 = Four floats used for calculating velocity. Not needed for visualization.
        # 4f = color -> rgba
        buffer_format = "4f 4x4 4f"

        # Attribute variable names for the vertex shader
        attributes = ["in_vertex", "in_color"]

        self.vao_previous = self.ctx.geometry(
            [BufferDescription(self.ssbo_previous, buffer_format, attributes)],
            mode=self.ctx.POINTS,
        )
        self.vao_current = self.ctx.geometry(
            [BufferDescription(self.ssbo_current, buffer_format, attributes)],
            mode=self.ctx.POINTS,
        )

        # --- Create the visualization shaders

        vertex_shader_source = Path("src/shaders/vertex_shader.glsl").read_text()
        fragment_shader_source = Path("src/shaders/fragment_shader.glsl").read_text()
        geometry_shader_source = Path("src/shaders/geometry_shader.glsl").read_text()

        # Create the complete shader program which will draw the stars
        self.program = self.ctx.program(
            vertex_shader=vertex_shader_source,
            geometry_shader=geometry_shader_source,
            fragment_shader=fragment_shader_source,
        )

        # --- Create our compute shader

        # Load in the raw source code safely & auto-close the file
        compute_shader_source = Path("src/shaders/compute_shader.glsl").read_text()

        # Compute shaders use groups to parallelize execution.
        # You don't need to understand how this works yet, but the
        # values below should serve as reasonable defaults. Later, we'll
        # preprocess the shader source by replacing the templating token
        # with its corresponding value.
        self.group_x = 256
        self.group_y = 1

        self.compute_shader_defines = {
            "COMPUTE_SIZE_X": self.group_x,
            "COMPUTE_SIZE_Y": self.group_y
        }

        # Preprocess the source by replacing each define with its value as a string
        for templating_token, value in self.compute_shader_defines.items():
            compute_shader_source = compute_shader_source.replace(templating_token, str(value))

        self.compute_shader = self.ctx.compute_shader(source=compute_shader_source)
        
        # Create random interaction matrix
        self.interaction_matrix = np.random.uniform(-2, 2, (N_PARTICLE_TYPES, N_PARTICLE_TYPES)).astype(np.float32)
        # Flatten the matrix for sending to shader
        matrix_data = self.interaction_matrix.flatten()

        # Pad the array to maximum size if needed
        max_size = 8*8  # Must match shader declaration
        if len(matrix_data) < max_size:
            matrix_data = np.pad(matrix_data, (0, max_size - len(matrix_data)))
        
        # After creating compute shader
        self.compute_shader["screen_size"] = (CANVAS_WIDTH, CANVAS_HEIGHT)  # Use canvas size instead of window size
        self.compute_shader["NUM_PARTICLES"] = NUM_PARTICLES
        self.compute_shader["N_PARTICLE_TYPES"] = N_PARTICLE_TYPES
        self.compute_shader["interaction_matrix"] = matrix_data.tolist()  # Convert to list for arcade
        
        # Set simulation parameters as uniforms
        self.compute_shader["interaction_radius"] = self.params.interaction_radius
        self.compute_shader["repulsion_radius"] = self.params.repulsion_radius
        self.compute_shader["repulsion_strength"] = self.params.repulsion_strength
        self.compute_shader["attraction_strength"] = self.params.attraction_strength
        self.compute_shader["max_force"] = self.params.max_force
        self.compute_shader["max_speed"] = self.params.max_speed
        self.compute_shader["friction"] = self.params.friction

        # Optional: Print the matrix to see the interactions
        print("Interaction Matrix:")
        print(self.interaction_matrix)


        # --- Create the FPS graph

        # # Enable timings for the performance graph
        # arcade.enable_timings()

        # # Create a sprite list to put the performance graph into
        # self.perf_graph_list = arcade.SpriteList()

        # # Create the FPS performance graph
        # graph = arcade.PerfGraph(GRAPH_WIDTH, GRAPH_HEIGHT, graph_data="FPS")
        # graph.position = GRAPH_WIDTH / 2, self.height - GRAPH_HEIGHT / 2
        # self.perf_graph_list.append(graph)


    def setup_ui(self):
        """
        Set up the user interface elements for simulation control.
        
        Creates a control panel with sliders for adjusting simulation
        parameters in real-time. The panel includes controls for:
        - Interaction and repulsion radii
        - Force strengths (repulsion and attraction)
        - Movement limits (max force and speed)
        - Friction
        Also adds a reset button to restore default parameters.
        """
        # Create a vertical box layout for our controls
        self.v_box = UIBoxLayout(
            space_between=20,  # Increased spacing between elements
            width=300,  # Fixed width for the side panel
        )

        # Add a title for the control panel
        title_label = UILabel(
            text="Simulation Controls",
            width=280,
            height=30,
            font_size=16,
            align="center"
        )
        self.v_box.add(title_label)

        # Helper function to create a slider row
        def create_slider_row(label_text: str, initial_value: float, min_val: float, max_val: float, 
                            callback, width: int = 200) -> Tuple[UISlider, UILabel]:
            # Create label
            label = UILabel(text=label_text, width=280, height=20)
            self.v_box.add(label)
            
            # Create row layout
            row = UIBoxLayout(vertical=False, space_between=10)
            
            # Create slider
            slider = UISlider(
                value=initial_value,
                min_value=min_val,
                max_value=max_val,
                width=width,
                height=20,
            )
            slider.on_change = callback
            row.add(slider)
            
            # Create value label
            value_label = UILabel(
                text=f"{initial_value:.1f}",
                width=60,
                height=20,
                align="right"
            )
            row.add(value_label)
            self.v_box.add(row)
            return slider, value_label

        # Create all sliders
        self.interaction_slider, self.interaction_value_label = create_slider_row(
            "Interaction Radius:", self.params.interaction_radius, 50, 400, self.on_interaction_radius_change)
            
        self.repulsion_slider, self.repulsion_value_label = create_slider_row(
            "Repulsion Radius:", self.params.repulsion_radius, 10, 100, self.on_repulsion_radius_change)
            
        self.repulsion_strength_slider, self.repulsion_strength_label = create_slider_row(
            "Repulsion Strength:", self.params.repulsion_strength, 0, 10, self.on_repulsion_strength_change)
            
        self.attraction_strength_slider, self.attraction_strength_label = create_slider_row(
            "Attraction Strength:", self.params.attraction_strength, 0, 5, self.on_attraction_strength_change)
            
        self.max_force_slider, self.max_force_label = create_slider_row(
            "Max Force:", self.params.max_force, 0.1, 2.0, self.on_max_force_change)
            
        self.max_speed_slider, self.max_speed_label = create_slider_row(
            "Max Speed:", self.params.max_speed, 1, 20, self.on_max_speed_change)
            
        self.friction_slider, self.friction_label = create_slider_row(
            "Friction:", self.params.friction, 0.5, 0.99, self.on_friction_change)

        # Add reset button
        from arcade.gui import UIFlatButton
        reset_button = UIFlatButton(
            text="Reset to Defaults",
            width=280,
            height=40,
        )
        reset_button.on_click = self.on_reset_click
        self.v_box.add(reset_button)

        # Add fresh start button
        fresh_start_button = UIFlatButton(
            text="Fresh Start",
            width=280,
            height=40,
        )
        fresh_start_button.on_click = self.on_fresh_start_click
        self.v_box.add(fresh_start_button)

        # Create an anchor layout for positioning
        anchor = UIAnchorLayout()
        
        # Add the box layout to the anchor with proper positioning
        anchor.add(
            child=self.v_box,
            anchor_x="right",
            anchor_y="center",
            align_x=-10,  # 10 pixels from the right edge
            align_y=0,
        )

        # Add the anchor layout to the UI manager
        self.ui_manager.add(anchor)

    def on_interaction_radius_change(self, event):
        """
        Handle changes to the interaction radius slider.
        
        Updates both the local parameter and the compute shader uniform
        when the user adjusts the interaction radius slider.
        
        Args:
            event: Slider change event containing the new value
        """
        self.params.interaction_radius = event.new_value
        self.compute_shader["interaction_radius"] = event.new_value
        self.interaction_value_label.text = f"{event.new_value:.1f}"
        print(f"Updated interaction radius to: {event.new_value}")

    def on_repulsion_radius_change(self, event):
        """
        Handle changes to the repulsion radius slider.
        
        Updates both the local parameter and the compute shader uniform
        when the user adjusts the repulsion radius slider.
        
        Args:
            event: Slider change event containing the new value
        """
        self.params.repulsion_radius = event.new_value
        self.compute_shader["repulsion_radius"] = event.new_value
        self.repulsion_value_label.text = f"{event.new_value:.1f}"
        print(f"Updated repulsion radius to: {event.new_value}")

    def on_repulsion_strength_change(self, event):
        """Handle changes to the repulsion strength slider."""
        self.params.repulsion_strength = event.new_value
        self.compute_shader["repulsion_strength"] = event.new_value
        self.repulsion_strength_label.text = f"{event.new_value:.1f}"

    def on_attraction_strength_change(self, event):
        """Handle changes to the attraction strength slider."""
        self.params.attraction_strength = event.new_value
        self.compute_shader["attraction_strength"] = event.new_value
        self.attraction_strength_label.text = f"{event.new_value:.1f}"

    def on_max_force_change(self, event):
        """Handle changes to the max force slider."""
        self.params.max_force = event.new_value
        self.compute_shader["max_force"] = event.new_value
        self.max_force_label.text = f"{event.new_value:.1f}"

    def on_max_speed_change(self, event):
        """Handle changes to the max speed slider."""
        self.params.max_speed = event.new_value
        self.compute_shader["max_speed"] = event.new_value
        self.max_speed_label.text = f"{event.new_value:.1f}"

    def on_friction_change(self, event):
        """Handle changes to the friction slider."""
        self.params.friction = event.new_value
        self.compute_shader["friction"] = event.new_value
        self.friction_label.text = f"{event.new_value:.1f}"

    def on_reset_click(self, event):
        """
        Reset all simulation parameters to their default values.
        
        Creates a new SimulationParams instance with default values and
        updates all UI elements and shader uniforms accordingly.
        
        Args:
            event: Button click event (unused)
        """
        # Create new params object with defaults
        default_params = SimulationParams()
        
        # Update all sliders and labels
        self.interaction_slider.value = default_params.interaction_radius
        self.repulsion_slider.value = default_params.repulsion_radius
        self.repulsion_strength_slider.value = default_params.repulsion_strength
        self.attraction_strength_slider.value = default_params.attraction_strength
        self.max_force_slider.value = default_params.max_force
        self.max_speed_slider.value = default_params.max_speed
        self.friction_slider.value = default_params.friction
        
        # Update all shader parameters
        self.params = default_params
        self.compute_shader["interaction_radius"] = default_params.interaction_radius
        self.compute_shader["repulsion_radius"] = default_params.repulsion_radius
        self.compute_shader["repulsion_strength"] = default_params.repulsion_strength
        self.compute_shader["attraction_strength"] = default_params.attraction_strength
        self.compute_shader["max_force"] = default_params.max_force
        self.compute_shader["max_speed"] = default_params.max_speed
        self.compute_shader["friction"] = default_params.friction
        
        # Update all value labels
        self.interaction_value_label.text = f"{default_params.interaction_radius:.1f}"
        self.repulsion_value_label.text = f"{default_params.repulsion_radius:.1f}"
        self.repulsion_strength_label.text = f"{default_params.repulsion_strength:.1f}"
        self.attraction_strength_label.text = f"{default_params.attraction_strength:.1f}"
        self.max_force_label.text = f"{default_params.max_force:.1f}"
        self.max_speed_label.text = f"{default_params.max_speed:.1f}"
        self.friction_label.text = f"{default_params.friction:.1f}"

    def generate_video_filename(self) -> str:
        """
        Generate a unique filename for video recording.
        
        Creates a filename based on current timestamp and simulation
        parameters, ensuring each recording has a unique identifier.
        
        Returns:
            str: Path to the video file in the videos directory
        """
        # Create videos directory if it doesn't exist
        videos_dir = Path("videos")
        videos_dir.mkdir(exist_ok=True)
        
        # Format: particle_life_YYYYMMDD_HHMMSS_30kp_5t.mp4
        # Where 30kp means 30,000 particles, 5t means 5 particle types
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        particles_str = f"{NUM_PARTICLES//1000}kp" if NUM_PARTICLES >= 1000 else f"{NUM_PARTICLES}p"
        types_str = f"{N_PARTICLE_TYPES}t"
        filename = f"particle_life_{timestamp}_{particles_str}_{types_str}.mp4"
        
        # Return full path in videos directory
        return str(videos_dir / filename)

    def start_recording(self, output_path: Optional[str] = None):
        """
        Start recording the simulation to video.
        
        Initializes video recording with the specified or auto-generated
        filename. Records at 30 FPS in MP4 format.
        
        Args:
            output_path (Optional[str]): Custom path for the video file
        """
        if not self.recording:
            # Generate unique filename if none provided
            if output_path is None:
                output_path = self.generate_video_filename()
            
            # Get the actual framebuffer size
            fb_width, fb_height = self.get_framebuffer_size()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                30.0,  # FPS
                (fb_width, fb_height)
            )
            self.recording = True
            self.frame_count = 0
            self.start_time = time.time()
            # Clear any previous output and print start message
            print("\033[K", end="")
            print(f"Started recording to {output_path}")
            print("\033[K", end="", flush=True)  # Clear line and prepare for progress updates

    def stop_recording(self):
        """
        Stop the current video recording.
        
        Finalizes the video file and prints recording statistics
        including duration and frame count.
        """
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            duration = time.time() - self.start_time
            # Clear the progress line and print final message
            print("\033[K", end="")
            print(f"\tRecording stopped. Saved {self.frame_count} frames ({duration:.1f} seconds)")
            # Add a newline to ensure clean state for next recording
            print("")

    def on_key_press(self, key: int, modifiers: int):
        """
        Handle keyboard input for video recording control.
        
        Ctrl+R toggles video recording on/off.
        
        Args:
            key (int): The key that was pressed
            modifiers (int): Modifier keys (ctrl, shift, etc.)
        """
        if key == arcade.key.R and modifiers & arcade.key.MOD_CTRL:
            # Ctrl+R to toggle recording
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()

    def on_draw(self):
        """
        Main rendering method called every frame.
        
        Performs the following steps:
        1. Clears the screen
        2. Updates particle positions using compute shader
        3. Renders particles
        4. Updates and draws UI elements
        5. Handles video recording if active
        6. Swaps particle buffers for next frame
        """
        # Clear the screen
        self.clear()
        
        # Set viewport for particle simulation
        self.ctx.viewport = (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
        
        # Update compute shader screen size if window is resized
        self.compute_shader["screen_size"] = (CANVAS_WIDTH, CANVAS_HEIGHT)
        
        # Enable blending
        self.ctx.enable(self.ctx.BLEND)

        # Bind buffers and run compute shader
        self.ssbo_previous.bind_to_storage_buffer(binding=0)
        self.ssbo_current.bind_to_storage_buffer(binding=1)
        
        # Ensure matrix data is properly maintained
        matrix_data = self.interaction_matrix.flatten()
        max_size = 8*8
        if len(matrix_data) < max_size:
            matrix_data = np.pad(matrix_data, (0, max_size - len(matrix_data)))
        matrix_list = matrix_data.tolist()
        
        # Update shader uniforms
        self.compute_shader["interaction_matrix"] = matrix_list
        
        # Run the compute shader
        self.compute_shader.run(group_x=self.group_x, group_y=self.group_y)

        # Draw particles
        self.vao_current.render(self.program)

        # Reset viewport for UI
        self.ctx.viewport = (0, 0, self.width, self.height)
        
        # Draw UI
        self.ui_manager.draw()

        # Record frame if recording
        if self.recording and self.video_writer:
            # Get the actual framebuffer size
            fb_width, fb_height = self.get_framebuffer_size()
            
            # Read the frame buffer (4 components: RGBA)
            image_buffer = self.ctx.screen.read(components=4)
            # Convert to numpy array and reshape
            frame = np.frombuffer(image_buffer, dtype=np.uint8)
            frame = frame.reshape((fb_height, fb_width, 4))
            # Extract only RGB components
            frame = frame[:, :, :3]
            # OpenCV uses BGR format, so convert from RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Flip the image vertically (OpenGL coordinate system is different)
            frame = cv2.flip(frame, 0)
            # Write the frame
            self.video_writer.write(frame)
            self.frame_count += 1
            
            # Update progress
            duration = time.time() - self.start_time
            print(f"\r\033[KRecording: {self.frame_count} frames ({duration:.1f} seconds) - Press Ctrl+R to stop", end="", flush=True)

        # Swap the buffer pairs.
        self.ssbo_previous, self.ssbo_current = self.ssbo_current, self.ssbo_previous
        self.vao_previous, self.vao_current = self.vao_current, self.vao_previous

        # Draw the graphs
        # self.perf_graph_list.draw()

    def on_fresh_start_click(self, event):
        """
        Reinitialize the simulation for a fresh start.
        
        Args:
            event: Button click event (unused)
        """
        # Close the current window and start a new instance
        self.close()
        new_app = ParticleLifeWindow()
        arcade.run()


if __name__ == "__main__":
    app = ParticleLifeWindow()
    arcade.run()