#version 430

layout(local_size_x=256, local_size_y=1) in;

// Input uniforms
uniform vec2 screen_size;
uniform int NUM_PARTICLES;
uniform int N_PARTICLE_TYPES;
uniform float interaction_matrix[64];  // Changed to 1D array for easier passing

// Simulation parameters (now as uniforms)
uniform float interaction_radius;    // Maximum radius for interaction
uniform float repulsion_radius;      // Distance where repulsion starts
uniform float repulsion_strength;    // Base strength of repulsion force
uniform float attraction_strength;   // Base strength of attraction force
uniform float max_force;            // Maximum force magnitude
uniform float max_speed;            // Maximum velocity magnitude
uniform float friction;             // Velocity dampening

// Structure of the particle data
struct Particle {
    vec4 pos;    // xyz = position, w = radius
    vec4 vel;    // xyz = velocity, w = unused
    vec4 color;  // rgb = color, a = type_id
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

float get_interaction_strength(int type1, int type2) {
    int matrix_index = type1 * N_PARTICLE_TYPES + type2;
    return interaction_matrix[matrix_index];
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if(id >= NUM_PARTICLES) return;

    // Get current particle data
    vec2 pos = In.particles[id].pos.xy;
    vec2 vel = In.particles[id].vel.xy;
    vec4 color = In.particles[id].color;
    
    // Get particle type directly from alpha channel
    int type1 = int(color.a);
    
    vec2 force_sum = vec2(0.0);

    // Calculate forces from other particles
    for(int i = 0; i < NUM_PARTICLES; i++) {
        if(i == int(id)) continue;

        vec2 other_pos = In.particles[i].pos.xy;
        vec4 other_color = In.particles[i].color;
        
        // Get other particle's type directly from alpha channel
        int type2 = int(other_color.a);

        vec2 diff = toroidal_diff(pos, other_pos, screen_size);
        float dist = length(diff);
        
        if(dist > 0.0 && dist < interaction_radius) {  // Only process if within range
            vec2 direction = normalize(diff);
            
            // Get interaction strength from matrix
            float interaction = get_interaction_strength(type1, type2);
            
            // Universal short-range repulsion
            if(dist < repulsion_radius) {
                float repulsion = repulsion_strength * (1.0 - dist/repulsion_radius);
                force_sum -= direction * repulsion;
            }
            // Type-based attraction/repulsion
            else {
                // Smooth transition between repulsion and interaction radius
                float factor = attraction_strength * interaction * 
                             (1.0 - (dist - repulsion_radius)/(interaction_radius - repulsion_radius));
                             
                force_sum += direction * factor;
            }
        }
    }

    // Apply maximum force limit
    float force_mag = length(force_sum);
    if(force_mag > max_force) {
        force_sum = normalize(force_sum) * max_force;
    }

    // Update velocity with friction
    vel = vel * friction + force_sum;
    
    // Apply speed limit
    float speed = length(vel);
    if(speed > max_speed) {
        vel = normalize(vel) * max_speed;
    }

    // Update position with wraparound
    pos = mod(pos + vel, screen_size);

    // Write output
    Out.particles[id].pos = vec4(pos, 0.0, In.particles[id].pos.w);
    Out.particles[id].vel = vec4(vel, 0.0, 0.0);
    Out.particles[id].color = color;
}