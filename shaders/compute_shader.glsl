#version 430

layout(local_size_x=256, local_size_y=1) in;

// Input uniforms
uniform vec2 screen_size;
uniform int NUM_PARTICLES;
uniform int N_PARTICLE_TYPES;
uniform float interaction_matrix[64];  // Changed to 1D array for easier passing

// Simulation parameters
const float INTERACTION_RADIUS = 200.0;    // Maximum radius for interaction
const float REPULSION_RADIUS = 50;      // Distance where repulsion starts
const float REPULSION_STRENGTH = 4.0;     // Base strength of repulsion force
const float ATTRACTION_STRENGTH = 1.0;    // Base strength of attraction force
const float MAX_FORCE = 0.7;             // Maximum force magnitude
const float MAX_SPEED = 10.0;             // Maximum velocity magnitude
const float FRICTION = 0.9;             // Velocity dampening


// Structure of the particle data
struct Particle {
    vec4 pos;    // xyz = position, w = radius
    vec4 vel;    // xyz = velocity, w = unused
    vec4 color;  // rgba = color/type
};

// Input buffer
layout(std430, binding=0) buffer particles_in {
    Particle particles[];
} In;

// Output buffer
layout(std430, binding=1) buffer particles_out {
    Particle particles[];
} Out;


// Add this function before main()
vec2 toroidal_diff(vec2 pos1, vec2 pos2, vec2 world_size) {
    vec2 diff = pos2 - pos1;
    
    // Check for shorter distance across screen borders
    diff.x = diff.x - world_size.x * round(diff.x / world_size.x);
    diff.y = diff.y - world_size.y * round(diff.y / world_size.y);
    
    return diff;
}


void main() {
    uint id = gl_GlobalInvocationID.x;
    if(id >= NUM_PARTICLES) return;

    // Get current particle data
    vec2 pos = In.particles[id].pos.xy;
    vec2 vel = In.particles[id].vel.xy;
    vec4 color = In.particles[id].color;

    // Calculate particle type from color
    int type1 = int(color.r + color.g * 2.0 + color.b * 4.0);
    
    vec2 force_sum = vec2(0.0);

    // Calculate forces from other particles
    for(int i = 0; i < NUM_PARTICLES; i++) {
        if(i == int(id)) continue;

        vec2 other_pos = In.particles[i].pos.xy;
        vec4 other_color = In.particles[i].color;
        int type2 = int(other_color.r + other_color.g * 2.0 + other_color.b * 4.0);

        vec2 diff = toroidal_diff(pos, other_pos, screen_size);
        float dist = length(diff);
        vec2 direction = normalize(diff);

        if(dist > 0.0) {  // Avoid division by zero
            // Universal repulsion (inverse proportional to distance)
            if(dist < REPULSION_RADIUS) {
                float repulsion = REPULSION_STRENGTH * (1.0 - dist/REPULSION_RADIUS);
                force_sum -= direction * repulsion;
            }
            
            // Attraction/repulsion based on particle types
            if(dist < INTERACTION_RADIUS && dist > REPULSION_RADIUS) {
                // Convert 2D index to 1D for matrix lookup
                int matrix_index = type1 * N_PARTICLE_TYPES + type2;
                float interaction = interaction_matrix[matrix_index];
                
                // Smooth transition between repulsion and interaction radius
                float factor = ATTRACTION_STRENGTH * interaction * 
                             (1.0 - (dist - REPULSION_RADIUS)/(INTERACTION_RADIUS - REPULSION_RADIUS));
                             
                force_sum += direction * factor;
            }
        }
    }

    // Apply maximum force limit
    float force_mag = length(force_sum);
    if(force_mag > MAX_FORCE) {
        force_sum = normalize(force_sum) * MAX_FORCE;
    }

    // Update velocity with friction
    vel = vel * FRICTION + force_sum;
    
    // Apply speed limit
    float speed = length(vel);
    if(speed > MAX_SPEED) {
        vel = normalize(vel) * MAX_SPEED;
    }

    // Update position with wraparound
    pos = mod(pos + vel, screen_size);

     // Write output
    Out.particles[id].pos = vec4(pos, 0.0, In.particles[id].pos.w);
    Out.particles[id].vel = vec4(vel, 0.0, 0.0);
    Out.particles[id].color = color;
}