"""
Particle Life
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


# Window dimensions
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000

# Performance graph dimensions 
GRAPH_WIDTH = 200
GRAPH_HEIGHT = 120
GRAPH_MARGIN = 5

# Simulation parameters
NUM_PARTICLES: int = 30000
N_PARTICLE_TYPES: int = 5


def gen_initial_data(
        screen_size: Tuple[int, int],
        num_particles: int = NUM_PARTICLES,
        num_particle_types: int = 4,
        palette: str = "husl"  # Good for distinguishable colors
) -> array:
    width, height = screen_size
    colors = sns.color_palette(palette, num_particle_types)
    def _data_generator() -> Generator[float, None, None]:
        for i in range(num_particles):
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

            # Color (now represents particle type)
            type_id = random.randint(0, num_particle_types - 1)
            r, g, b = colors[type_id]
            yield r
            yield g
            yield b
            yield 1.0  # Alpha channel

    # Use the generator function to fill an array in RAM
    return array('f', _data_generator())

class NBodyGravityWindow(arcade.Window):

    def __init__(self):
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
        # Video recording properties
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.frame_count = 0
        
        # Attempt to put the window in the center of the screen.
        self.center_window()

        # --- Create buffers

        # Create pairs of buffers for the compute & visualization shaders.
        # We will swap which buffer instance is the initial value and
        # which is used as the current value to write to.

        # ssbo = shader storage buffer object
        initial_data = gen_initial_data(self.get_size(), num_particle_types=N_PARTICLE_TYPES)
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

        vertex_shader_source = Path("shaders/vertex_shader.glsl").read_text()
        fragment_shader_source = Path("shaders/fragment_shader.glsl").read_text()
        geometry_shader_source = Path("shaders/geometry_shader.glsl").read_text()

        # Create the complete shader program which will draw the stars
        self.program = self.ctx.program(
            vertex_shader=vertex_shader_source,
            geometry_shader=geometry_shader_source,
            fragment_shader=fragment_shader_source,
        )

        # --- Create our compute shader

        # Load in the raw source code safely & auto-close the file
        compute_shader_source = Path("shaders/compute_shader.glsl").read_text()

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
        self.compute_shader["screen_size"] = self.get_size()
        self.compute_shader["NUM_PARTICLES"] = NUM_PARTICLES
        self.compute_shader["N_PARTICLE_TYPES"] = N_PARTICLE_TYPES
        self.compute_shader["interaction_matrix"] = matrix_data.tolist()  # Convert to list for arcade

        
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


    def generate_video_filename(self) -> str:
        """Generate a unique video filename based on current time and simulation parameters."""
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
        """Start recording the simulation to video."""
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
            print(f"\nStarted recording to {output_path}")

    def stop_recording(self):
        """Stop recording and save the video."""
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            duration = time.time() - self.start_time
            # Clear the current line and print the stop message
            print(f"\033[K\rRecording stopped. Saved {self.frame_count} frames ({duration:.1f} seconds)", flush=True)

    def on_key_press(self, key: int, modifiers: int):
        """Handle key press events for video recording control."""
        if key == arcade.key.R and modifiers & arcade.key.MOD_CTRL:
            # Ctrl+R to toggle recording
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()

    def on_draw(self):
        # Clear the screen
        self.clear()
        # Enable blending so our alpha channel works
        self.ctx.enable(self.ctx.BLEND)

        # Bind buffers
        self.ssbo_previous.bind_to_storage_buffer(binding=0)
        self.ssbo_current.bind_to_storage_buffer(binding=1)

        # Run compute shader to calculate new positions for this frame
        self.compute_shader.run(group_x=self.group_x, group_y=self.group_y)

        # Draw the current star positions
        self.vao_current.render(self.program)

        # Record frame if we're recording
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
            
            # Update progress (clear the line first)
            duration = time.time() - self.start_time
            print(f"\033[K\rRecording: {self.frame_count} frames ({duration:.1f} seconds) - Press Ctrl+R to stop", end="", flush=True)

        # Swap the buffer pairs.
        self.ssbo_previous, self.ssbo_current = self.ssbo_current, self.ssbo_previous
        self.vao_previous, self.vao_current = self.vao_current, self.vao_previous

        # Draw the graphs
        # self.perf_graph_list.draw()



if __name__ == "__main__":
    app = NBodyGravityWindow()
    arcade.run()