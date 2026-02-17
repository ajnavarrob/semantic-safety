#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include <cstring> // For memcpy
#include <stdexcept> // For std::runtime_error

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For std::pair

// Global variables, now mutable
int IMAX = 120; // Grid X Size
int JMAX = 120; // Grid Y Size
float DS = 0.0294f; // X-Y Grid Resolution

// Poisson Variables
// float *hgrid; // Will be managed by Python wrapper / calling code
// int h_iters;  // Will be returned by the solver function
float h0 = 0.0f; // Set boundary level set value
float dh0 = 1.0f; // Set dh Value  
std::vector<float> dh0_classes = {1.0f}; // Default: one class

// Robot kernel parameters (now global and mutable)
float robot_length = 0.13f; // Crazyflie Drone default
float robot_width = 0.13f;
// Setter and Getter functions for global parameters
void set_IMAX(int val) { IMAX = val; }
int get_IMAX() { return IMAX; }

void set_JMAX(int val) { JMAX = val; }
int get_JMAX() { return JMAX; }

void set_DS(float val) { DS = val; }
float get_DS() { return DS; }

void set_h0_val(float val) { h0 = val; } // Renamed to avoid conflict with get_h0 function
float get_h0_val() { return h0; }

void set_dh0_classes(const std::vector<float>& vals) { dh0_classes = vals; }
std::vector<float> get_dh0_classes() { return dh0_classes; }
void set_dh0_val(float val) { dh0 = val; if (!dh0_classes.empty()) dh0_classes[0] = val; }
float get_dh0_val() { return dh0; }
float get_dh0_for_class(int class_idx) {
    if (class_idx >= 0 && class_idx < (int)dh0_classes.size())
        return dh0_classes[class_idx];
    return dh0_classes.empty() ? dh0 : dh0;
}

// Getter and setter for robot_length
void set_robot_length(float val) { robot_length = val; }
float get_robot_length() { return robot_length; }

// Getter and setter for robot_width
void set_robot_width(float val) { robot_width = val; }
float get_robot_width() { return robot_width; }

/* Perform a bilinear interpolation on a 2-D grid */
float bilinear_interpolation(const float *grid, const float i, const float j){

    const float i1f = floorf(i);
    const float j1f = floorf(j);
    const float i2f = ceilf(i);
    const float j2f = ceilf(j);

    const int i1 = (int)i1f;
    const int j1 = (int)j1f;
    const int i2 = (int)i2f;
    const int j2 = (int)j2f;

    if((i1 != i2) && (j1 != j2)){
        const float f1 = (i2f - i) * grid[i1*JMAX+j1] + (i - i1f) * grid[i2*JMAX+j1];
        const float f2 = (i2f - i) * grid[i1*JMAX+j2] + (i - i1f) * grid[i2*JMAX+j2];
        return (j2f - j) * f1 + (j - j1f) * f2;
    }
    else if(i1 != i2){
        return (i2f - i) * grid[i1*JMAX+(int)j] + (i - i1f) * grid[i2*JMAX+(int)j];
    }
    else if(j1 != j2){
        return (j2f - j) * grid[(int)i*JMAX+j1] + (j - j1f) * grid[(int)i*JMAX+j2];
    }
    else{
        return grid[(int)i*JMAX+(int)j];
    }

};

/* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
void find_boundary(float *bound){
    
    // Set Border
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
        }
    }

    float b0[IMAX*JMAX];
    memcpy(b0, bound, IMAX*JMAX*sizeof(float));
    for(int n = 0; n < IMAX*JMAX; n++){
        if(b0[n]==1.0f){
            if(b0[n+1]==-1.0f || 
                b0[n-1]==-1.0f || 
                b0[n+JMAX]==-1.0f || 
                b0[n-JMAX]==-1.0f || 
                b0[n+JMAX+1]==-1.0f || 
                b0[n-JMAX+1]==-1.0f || 
                b0[n+JMAX-1]==-1.0f || 
                b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
        }
    }

};

/* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
void find_and_fix_boundary(float *grid, float *bound){
    
    // Set Border
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
        }
    }

    float b0[IMAX*JMAX];
    memcpy(b0, bound, IMAX*JMAX*sizeof(float));
    for(int n = 0; n < IMAX*JMAX; n++){
        if(b0[n]==1.0f){
            if(b0[n+1]==-1.0f || 
                b0[n-1]==-1.0f || 
                b0[n+JMAX]==-1.0f || 
                b0[n-JMAX]==-1.0f || 
                b0[n+JMAX+1]==-1.0f || 
                b0[n-JMAX+1]==-1.0f || 
                b0[n+JMAX-1]==-1.0f || 
                b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
        }
        if(!bound[n]) grid[n] = h0;
    }

};

/* Buffer Occupancy Grid with 2-D Robot Shape */
void inflate_occupancy_grid(float *bound, const float yawk){

    /* Step 1: Create Robot Kernel */
    const float length = robot_length;
    const float width = robot_width;
    
    const float D = sqrtf(length*length + width*width); // Max Robot Dimension to Define Kernel Size
    const int dim = ceilf((ceilf(D / DS) + 1.0f) / 2.0f) * 2.0f - 1.0f;
    float robot_grid[dim*dim];

    const float MOS = 1.2f;
    const float ar = MOS * length / 2.0f;
    const float br = MOS * width / 2.0f;

    const float expo = 2.0f;
    for(int i = 0; i < dim; i++){
        const float yi = (float)i*DS - D/2.0f;
        for(int j = 0; j < dim; j++){
            robot_grid[i*dim+j] = 0.0;
            const float xi = (float)j*DS - D/2.0f;
            const float xb = cosf(yawk)*xi + sinf(yawk)*yi;
            const float yb = -sinf(yawk)*xi + cosf(yawk)*yi;
            const float dist = powf(fabsf(xb/ar), expo) + powf(fabsf(yb/br), expo);
            if(dist <= 1.0f) robot_grid[i*dim+j] = -1.0f;
            //if(fabsf(xb/ar) <= 1.0f && fabsf(yb/br) <= 1.0f) robot_grid[i*dim+j] = -1.0f;
        }
    }

    /* Step 2: Convolve Robot Kernel with Occupancy Grid, Along the Boundary */
    float b0[IMAX*JMAX];
    memcpy(b0, bound, IMAX*JMAX*sizeof(float));

    int lim = (dim - 1)/2;
    for(int i = 1; i < IMAX-1; i++){
        int ilow = std::max(i - lim, 0);
        int itop = std::min(i + lim, IMAX);
        for(int j = 1; j < JMAX-1; j++){
            int jlow = std::max(j - lim, 0);
            int jtop = std::min(j + lim, JMAX);
            if(!b0[i*JMAX+j]){
                for(int p = ilow; p < itop; p++){
                    for(int q = jlow; q < jtop; q++){
                        bound[p*JMAX+q] += robot_grid[(p-i+lim)*dim+(q-j+lim)];
                    }
                }
            }
        }
    }
    for(int n = 0; n < IMAX*JMAX; n++){
        if(bound[n] < -1.0f) bound[n] = -1.0f;
    }

};

/* Compute Forcing Function for Average Flux */
void compute_fast_forcing_function(float *force, const float *bound){

    float perimeter_c = 0.0f;
    float area_c = 0.0f;
    
    for(int i = 1; i < IMAX-1; i++){
        for(int j = 1; j < JMAX-1; j++){
            if(bound[i*JMAX+j] == 0.0f) perimeter_c += DS;
            else if(bound[i*JMAX+j] < 0.0f) area_c += DS*DS;
        }
    }
    
    float perimeter_o = 2.0f*(float)IMAX*DS + 2.0f*(float)JMAX*DS + perimeter_c;
    float area_o = (float)IMAX*(float)JMAX*DS*DS - area_c;
    float force_o = -dh0 * perimeter_o / area_o * DS*DS;
    float force_c = 0.0f;
    if(area_c != 0.0f) force_c = dh0 * perimeter_c / area_c * DS*DS;
    
    for(int n = 0; n < IMAX*JMAX; n++){
        if(bound[n] > 0.0f){
            force[n] = force_o;
        }
        else if(bound[n] < 0.0f){
            force[n] = force_c;
        }
        else{
            force[n] = 0.0f;
        }
    }

};

/* Solve Poisson's Equation -- Checkerboard Successive Overrelaxation (SOR) Method */
int poisson(float *grid, const float *force, const float *bound, const float relTol = 1.0e-4f, const float N = 25.0f){
    
    const float w_SOR = 2.0f/(1.0f+sinf(M_PI/(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5

    int iters = 0;
    const int max_iters = 10000;
    for(int n = 0; n < max_iters; n++){

        float rss = 0.0f;
       
        // Red Pass
        #pragma omp parallel for
        for(int i = 1; i < IMAX-1; i++){
            for(int j = 1; j < JMAX-1; j++){
                const bool red = (((i%2)+(j%2))%2) == 0;
                if(bound[i*JMAX+j] && red){
                    float dg = 0.0f;
                    dg += (grid[(i+1)*JMAX+j] + grid[(i-1)*JMAX+j]);
                    dg += (grid[i*JMAX+(j+1)] + grid[i*JMAX+(j-1)]);
                    dg -= force[i*JMAX+j];
                    dg /= 4.0f;
                    dg -= grid[i*JMAX+j];
                    grid[i*JMAX+j] += w_SOR * dg;
                    rss += dg * dg;
                }
            }
        }
        // Black Pass
        #pragma omp parallel for
        for(int i = 1; i < IMAX-1; i++){
            for(int j = 1; j < JMAX-1; j++){
                const bool black = (((i%2)+(j%2))%2) == 1;
                if(bound[i*JMAX+j] && black){
                    float dg = 0.0f;
                    dg += (grid[(i+1)*JMAX+j] + grid[(i-1)*JMAX+j]);
                    dg += (grid[i*JMAX+(j+1)] + grid[i*JMAX+(j-1)]);
                    dg -= force[i*JMAX+j];
                    dg /= 4.0f;
                    dg -= grid[i*JMAX+j];
                    grid[i*JMAX+j] += w_SOR * dg;
                    rss += dg * dg;
                }
            }
        }

        rss = sqrtf(rss) * DS;
        iters++;
        if(rss < relTol) break;

    }

    return iters;

};

// Global social navigation parameters (to match semantic_poisson.cpp)
bool enable_social_navigation_ = false;  // Pass on human's right (robot goes left)
float social_tangent_bias_ = 0.5f;       // Tangential bias strength when enabled
float dh0_human = 1.0f;                  // Default dh0 for human class
float dh0_obstacle = 0.3f;               // Default dh0 for obstacle class

// Getter and setter for social navigation
void set_enable_social_navigation(bool val) { enable_social_navigation_ = val; }
bool get_enable_social_navigation() { return enable_social_navigation_; }
void set_social_tangent_bias(float val) { social_tangent_bias_ = val; }
float get_social_tangent_bias() { return social_tangent_bias_; }
void set_dh0_human(float val) { dh0_human = val; }
float get_dh0_human() { return dh0_human; }
void set_dh0_obstacle(float val) { dh0_obstacle = val; }
float get_dh0_obstacle() { return dh0_obstacle; }

// Compute search radius based on robot kernel size
int get_robot_kernel_dim() {
    const float D = sqrtf(robot_length*robot_length + robot_width*robot_width);
    const int dim = ceilf((ceilf(D / DS) + 1.0f) / 2.0f) * 2.0f - 1.0f;
    return dim;
}

void compute_boundary_gradients(float *guidance_x, float *guidance_y, const float *bound, const int* class_map = nullptr){
    // Set Border Gradients
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            if(i==0) guidance_x[i*JMAX+j] = dh0;
            if(j==0) guidance_y[i*JMAX+j] = dh0;
            if(i==(IMAX-1)) guidance_x[i*JMAX+j] = -dh0;
            if(j==(JMAX-1)) guidance_y[i*JMAX+j] = -dh0;
        }
    }
    // Set Additional Boundary Gradients
    for(int i = 1; i < IMAX-1; i++){
        for(int j = 1; j < JMAX-1; j++){
            if(!bound[i*JMAX+j]){ // bound value is 0 so it's checking for existing boundary
                guidance_x[i*JMAX+j] = 0.0f;
                guidance_y[i*JMAX+j] = 0.0f;
                for(int p = -1; p <= 1; p++){
                    for(int q = -1; q <= 1; q++){
                        if(q > 0){
                            guidance_x[i*JMAX+j] += bound[(i+q)*JMAX+(j+p)];
                            guidance_y[i*JMAX+j] += bound[(i+p)*JMAX+(j+q)];
                        }
                        else if (q < 0){
                            guidance_x[i*JMAX+j] -= bound[(i+q)*JMAX+(j+p)];
                            guidance_y[i*JMAX+j] -= bound[(i+p)*JMAX+(j+q)];
                        }
                    }
                }
            }
        }
    }
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            if(!bound[i*JMAX+j]){  // Boundary cell
                const float V = sqrtf(guidance_x[i*JMAX+j]*guidance_x[i*JMAX+j] + guidance_y[i*JMAX+j]*guidance_y[i*JMAX+j]);
                if(V != 0.0f){
                    guidance_x[i*JMAX+j] /= V;
                    guidance_y[i*JMAX+j] /= V;
                }
                float local_dh0 = dh0_obstacle;  // Default
                bool is_human = false;
                if (class_map) {
                    // Check if ANY cell within search radius has a human label
                    // Use larger radius for more uniform human detection
                    int search_radius = get_robot_kernel_dim() / 2;
                    for(int di = -search_radius; di <= search_radius && local_dh0 != dh0_human; di++){
                        for(int dj = -search_radius; dj <= search_radius; dj++){
                            int ni = i + di;
                            int nj = j + dj;
                            if(ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX){
                                // Check if this cell has a human label (don't require it to be occupied)
                                if(class_map[ni*JMAX+nj] == 1){
                                    local_dh0 = dh0_human;
                                    is_human = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // Apply social navigation tangential bias for humans
                // The tangent perpendicular to (gx, gy) for CCW rotation is (-gy, gx)
                // This causes the robot to pass on the human's RIGHT (robot goes LEFT)
                if (is_human && enable_social_navigation_) {
                    float gx = guidance_x[i*JMAX+j];
                    float gy = guidance_y[i*JMAX+j];
                    // Add CCW tangential component: (-gy, gx)
                    // The tangent_bias controls the blend ratio
                    guidance_x[i*JMAX+j] = gx + social_tangent_bias_ * (-gy);
                    guidance_y[i*JMAX+j] = gy + social_tangent_bias_ * gx;
                    // Re-normalize to unit vector
                    const float V2 = sqrtf(guidance_x[i*JMAX+j]*guidance_x[i*JMAX+j] + 
                                           guidance_y[i*JMAX+j]*guidance_y[i*JMAX+j]);
                    if(V2 != 0.0f){
                        guidance_x[i*JMAX+j] /= V2;
                        guidance_y[i*JMAX+j] /= V2;
                    }
                }
                
                guidance_x[i*JMAX+j] *= local_dh0;
                guidance_y[i*JMAX+j] *= local_dh0;
            }
        }
    }
};

/* Compute Forcing Function from Guidance Field */
void compute_optimal_forcing_function(float *force, const float *guidance_x, const float *guidance_y, const float *bound){
    const float max_div = 10.0f;
    const float alpha = 3.0f;
    for(int i = 1; i < (IMAX-1); i++){
        for(int j = 1; j < (JMAX-1); j++){
            force[i*JMAX+j] = (guidance_x[(i+1)*JMAX+j] - guidance_x[(i-1)*JMAX+j]) / (2.0f*DS) + (guidance_y[i*JMAX+(j+1)] - guidance_y[i*JMAX+(j-1)]) / (2.0f*DS);
            if(bound[i*JMAX+j] > 0.0f){
                // force[i*JMAX+j] = softMin(force[i*JMAX+j], -max_div, alpha);
                // force[i*JMAX+j] = softMax(force[i*JMAX+j], 0.0f, alpha);
            
            }
            else if(bound[i*JMAX+j] < 0.0f){
                // force[i*JMAX+j] = softMax(force[i*JMAX+j], max_div, alpha);
                force[i*JMAX+j] = std::max(force[i*JMAX+j], max_div);
                force[i*JMAX+j] = std::min(force[i*JMAX+j], 0.0f);
            }
            else{
                force[i*JMAX+j] = 0.0f;
            }
        }
    }
};

/* Compute boundary mask from occupancy (1 = boundary/occupied, 0 = interior free) */
void compute_boundary_mask_from_occ(float *boundary_mask, const float *occ_grid) {
    // Initialize to 0
    std::fill(boundary_mask, boundary_mask + IMAX*JMAX, 0.0f);

    // Mark occupied cells that are adjacent to free space or domain boundary
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            if(occ_grid[i*JMAX+j] == 1.0f) {
                // It is occupied. Check if it is on the boundary of the occupied region. // TODO change this to check corners also and the free space should be boundary
                bool is_boundary = false;
                
                // Check 4-neighbors
                // Up (i-1)
                if (i == 0 || occ_grid[(i-1)*JMAX+j] == 0.0f) is_boundary = true;
                // Down (i+1)
                else if (i == IMAX-1 || occ_grid[(i+1)*JMAX+j] == 0.0f) is_boundary = true;
                // Left (j-1)
                else if (j == 0 || occ_grid[i*JMAX+(j-1)] == 0.0f) is_boundary = true;
                // Right (j+1)
                else if (j == JMAX-1 || occ_grid[i*JMAX+(j+1)] == 0.0f) is_boundary = true;
                
                if (is_boundary) {
                    boundary_mask[i*JMAX+j] = 1.0f;
                }
            }
        }
    }
    
    // Mark domain edges as boundary (Python does this explicitly at the end)
    for(int j = 0; j < JMAX; j++) {
        boundary_mask[0*JMAX+j] = 1.0f;
        boundary_mask[(IMAX-1)*JMAX+j] = 1.0f;
    }
    for(int i = 0; i < IMAX; i++) {
        boundary_mask[i*JMAX+0] = 1.0f;
        boundary_mask[i*JMAX+(JMAX-1)] = 1.0f;
    }
}

/* Compute outward normals from occupancy using Sobel-like gradient */
void compute_boundary_normals(float *nx, float *ny, const float *occ_grid) {
    // Helper function to get value with reflect boundary mode (mimic scipy's sobel default)
    // Scipy's 'reflect' mode: (d c b a | a b c d | d c b a)
    // The input is extended by reflecting about the edge of the last pixel
    auto get_val = [&](int i, int j) -> float {
        // Reflect mode for scipy:
        // - For index < 0: reflect as -i - 1 (e.g. -1 -> 0, -2 -> 1)
        // - For index >= size: reflect as 2*size - i - 1 (e.g. size -> size-1)
        if (i < 0) {
            i = -i - 1;
        } else if (i >= IMAX) {
            i = 2 * IMAX - i - 1;
        }
        
        if (j < 0) {
            j = -j - 1;
        } else if (j >= JMAX) {
            j = 2 * JMAX - j - 1;
        }
        
        // Clamp to valid range (in case reflection goes too far for very small grids)
        i = std::max(0, std::min(i, IMAX-1));
        j = std::max(0, std::min(j, JMAX-1));
        
        return occ_grid[i*JMAX+j];
    };
    
    // Apply Sobel filter to all points including boundaries
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            // Sobel in y-direction (vertical, row direction)
            float gy = -get_val(i-1, j-1) - 2.0f*get_val(i-1, j) - get_val(i-1, j+1)
                      + get_val(i+1, j-1) + 2.0f*get_val(i+1, j) + get_val(i+1, j+1);
            
            // Sobel in x-direction (horizontal, column direction)
            float gx = -get_val(i-1, j-1) - 2.0f*get_val(i, j-1) - get_val(i+1, j-1)
                      + get_val(i-1, j+1) + 2.0f*get_val(i, j+1) + get_val(i+1, j+1);
            
            float mag = sqrtf(gx*gx + gy*gy);
            // Match Python's mag[mag == 0] = 1  
            if(mag == 0.0f) mag = 1.0f;
            
            // Negative for outward normals
            nx[i*JMAX+j] = -gx / mag;
            ny[i*JMAX+j] = -gy / mag;
        }
    }
}

/* Convert flux map to boundary conditions: v = flux * normal */
void flux_to_bc(float *bc_x, float *bc_y, const float *flux_map, const float *nx, const float *ny) {
    for(int i = 0; i < IMAX; i++){
        for(int j = 0; j < JMAX; j++){
            int n = i*JMAX+j;
            
            // Check if this is a domain boundary
            bool is_domain_boundary = (i == 0 || i == IMAX-1 || j == 0 || j == JMAX-1);
            
            if(is_domain_boundary){
                // Inward pointing flux with magnitude 1
                // Negate normals to point inward instead of outward
                bc_x[n] = -nx[n];
                bc_y[n] = -ny[n];
            } else {
                // Normal flux-to-BC conversion for obstacle boundaries
                bc_x[n] = flux_map[n] * nx[n];
                bc_y[n] = flux_map[n] * ny[n];
            }
        }
    }
}

/* Solve Laplace equation using SOR for a single component */
int solve_laplace_sor(float *v, const float *boundary_mask, const float *bc_values, 
                       const float relTol = 1.0e-5f, const int max_iters = 50000) {
    // Compute optimal omega for SOR
    const float nmax = (float)std::max(IMAX, JMAX);
    float omega = 2.0f / (1.0f + sinf(M_PI / nmax));
    omega = std::max(1.0f, std::min(omega, 1.99f));
    
    // Warm start: preserve interior v passed in; only enforce boundary values
    for(int n = 0; n < IMAX*JMAX; n++){
        if(boundary_mask[n] > 0.5f) {
            v[n] = bc_values[n];
        }
    }
    
    int iters = 0;
    for(int iter = 0; iter < max_iters; iter++){
        float diff_sum = 0.0f;
        
        // Red-black SOR sweeps
        // Red sweep
        #pragma omp parallel for
        for(int i = 1; i < IMAX-1; i++){
            for(int j = 1; j < JMAX-1; j++){
                const bool red = (((i%2)+(j%2))%2) == 0;
                if(red && boundary_mask[i*JMAX+j] < 0.5f) { // Interior point
                    float neighbor_sum = v[(i+1)*JMAX+j] + v[(i-1)*JMAX+j] + 
                                        v[i*JMAX+(j+1)] + v[i*JMAX+(j-1)];
                    float v_new = 0.25f * neighbor_sum;
                    float delta = v_new - v[i*JMAX+j];
                    v[i*JMAX+j] += omega * delta;
                    diff_sum += delta * delta;
                }
            }
        }
        
        // Black sweep
        #pragma omp parallel for
        for(int i = 1; i < IMAX-1; i++){
            for(int j = 1; j < JMAX-1; j++){
                const bool black = (((i%2)+(j%2))%2) == 1;
                if(black && boundary_mask[i*JMAX+j] < 0.5f) { // Interior point
                    float neighbor_sum = v[(i+1)*JMAX+j] + v[(i-1)*JMAX+j] + 
                                        v[i*JMAX+(j+1)] + v[i*JMAX+(j-1)];
                    float v_new = 0.25f * neighbor_sum;
                    float delta = v_new - v[i*JMAX+j];
                    v[i*JMAX+j] += omega * delta;
                    diff_sum += delta * delta;
                }
            }
        }
        
        iters++;
        float diff = sqrtf(diff_sum);
        if(diff < relTol) {
            break;
        }
    }
    
    return iters;
}

/* Flux-based solver implementation following guidance_field.py approach */
int solve_poisson_safety_function_flux_impl(float *grid_out, float *guidance_x, float *guidance_y, 
                                             const float *occ_in, float yaw_param, 
                                             const float *flux_map) {
    // Persistent buffers for warm start
    static float *hgrid = nullptr;
    static float *guidance_x_buf = nullptr;
    static float *guidance_y_buf = nullptr;
    static int last_IMAX = -1, last_JMAX = -1;

    if (hgrid == nullptr || guidance_x_buf == nullptr || guidance_y_buf == nullptr ||
        last_IMAX != IMAX || last_JMAX != JMAX) {
        if (hgrid) free(hgrid);
        if (guidance_x_buf) free(guidance_x_buf);
        if (guidance_y_buf) free(guidance_y_buf);
        hgrid = (float *)malloc(IMAX * JMAX * sizeof(float));
        guidance_x_buf = (float *)malloc(IMAX * JMAX * sizeof(float));
        guidance_y_buf = (float *)malloc(IMAX * JMAX * sizeof(float));
        last_IMAX = IMAX;
        last_JMAX = JMAX;
        for (int n = 0; n < IMAX * JMAX; n++) {
            hgrid[n] = h0;
            guidance_x_buf[n] = 0.0f;
            guidance_y_buf[n] = 0.0f;
        }
    }

    // Copy warm-start values
    memcpy(grid_out, hgrid, IMAX * JMAX * sizeof(float));
    memcpy(guidance_x, guidance_x_buf, IMAX * JMAX * sizeof(float));
    memcpy(guidance_y, guidance_y_buf, IMAX * JMAX * sizeof(float));

    // Allocate temporary buffers
    float *boundary_mask = (float *)calloc(IMAX*JMAX, sizeof(float));
    float *nx = (float *)calloc(IMAX*JMAX, sizeof(float));
    float *ny = (float *)calloc(IMAX*JMAX, sizeof(float));
    float *bc_x = (float *)calloc(IMAX*JMAX, sizeof(float));
    float *bc_y = (float *)calloc(IMAX*JMAX, sizeof(float));
    float *bound = (float *)malloc(IMAX*JMAX*sizeof(float));
    float *force = (float *)malloc(IMAX*JMAX*sizeof(float));

    if (!boundary_mask || !nx || !ny || !bc_x || !bc_y || !bound || !force) {
        if (boundary_mask) free(boundary_mask);
        if (nx) free(nx);
        if (ny) free(ny);
        if (bc_x) free(bc_x);
        if (bc_y) free(bc_y);
        if (bound) free(bound);
        if (force) free(force);
        throw std::runtime_error("Memory allocation failed in solve_poisson_safety_function_flux_impl");
    }

    // Step 1: Compute boundary mask from occupancy
    compute_boundary_mask_from_occ(boundary_mask, occ_in);
    
    // Step 2: Compute boundary normals
    compute_boundary_normals(nx, ny, occ_in);
    
    // Step 3: Convert flux to boundary conditions
    flux_to_bc(bc_x, bc_y, flux_map, nx, ny);
    
    // Step 4: Solve Laplace for guidance field components
    // int vx_iters = solve_laplace_sor(guidance_x, boundary_mask, bc_x);
    // int vy_iters = solve_laplace_sor(guidance_y, boundary_mask, bc_y);
    
    // Step 5: Inflate occupancy for robot size and compute h-field boundaries
    for(int n=0; n<IMAX*JMAX; n++){
        if(occ_in[n] == 1.0f){
            bound[n] = -1.0f;
        }
        else{
            bound[n] = 1.0f;
        }
    }
    find_boundary(bound);
    inflate_occupancy_grid(bound, yaw_param);
    find_and_fix_boundary(grid_out, bound);
    compute_boundary_gradients(guidance_x, guidance_y, bound, nullptr);
    float *f0 = (float *)calloc(IMAX*JMAX, sizeof(float)); // Initialize f0 with zeros
    const float v_RelTol = 1.0e-4f;
    int vx_iters = poisson(guidance_x, f0, bound, v_RelTol, 25.0f);
    int vy_iters = poisson(guidance_y, f0, bound, v_RelTol, 25.0f);

    compute_optimal_forcing_function(force, guidance_x, guidance_y, bound);
    for(int n=0; n<IMAX*JMAX; n++){
        force[n] *= DS*DS;
    }

    const float h_RelTol = 1.0e-4f;
    int h_iters = poisson(grid_out, force, bound, h_RelTol, 25.0f);
    
    // Save to persistent buffers
    memcpy(hgrid, grid_out, IMAX * JMAX * sizeof(float));
    memcpy(guidance_x_buf, guidance_x, IMAX * JMAX * sizeof(float));
    memcpy(guidance_y_buf, guidance_y, IMAX * JMAX * sizeof(float));

    free(boundary_mask);
    free(nx);
    free(ny);
    free(bc_x);
    free(bc_y);
    free(bound);
    free(force);

    return h_iters;
}

/* Compute the Poisson Safety Function - Implementation Detail */
int solve_poisson_safety_function_impl(float *grid_out, float *guidance_x, float *guidance_y, const float *occ_in, float yaw_param, const int* class_map = nullptr){
    // --- Persistent buffers for warm start ---
    static float *hgrid = nullptr;
    static float *guidance_x_buf = nullptr;
    static float *guidance_y_buf = nullptr;
    static int last_IMAX = -1, last_JMAX = -1;

    // Allocate or reallocate persistent buffers if grid size changed
    if (hgrid == nullptr || guidance_x_buf == nullptr || guidance_y_buf == nullptr ||
        last_IMAX != IMAX || last_JMAX != JMAX) {
        if (hgrid) free(hgrid);
        if (guidance_x_buf) free(guidance_x_buf);
        if (guidance_y_buf) free(guidance_y_buf);
        hgrid = (float *)malloc(IMAX * JMAX * sizeof(float));
        guidance_x_buf = (float *)malloc(IMAX * JMAX * sizeof(float));
        guidance_y_buf = (float *)malloc(IMAX * JMAX * sizeof(float));
        last_IMAX = IMAX;
        last_JMAX = JMAX;
        // Initialize to h0/0
        for (int n = 0; n < IMAX * JMAX; n++) {
            hgrid[n] = h0;
            guidance_x_buf[n] = 0.0f;
            guidance_y_buf[n] = 0.0f;
        }
    }

    // Copy persistent buffers to output pointers for Python
    memcpy(grid_out, hgrid, IMAX * JMAX * sizeof(float));
    memcpy(guidance_x, guidance_x_buf, IMAX * JMAX * sizeof(float));
    memcpy(guidance_y, guidance_y_buf, IMAX * JMAX * sizeof(float));

    float *bound, *force;
    bound = (float *)malloc(IMAX*JMAX*sizeof(float));
    force = (float *)malloc(IMAX*JMAX*sizeof(float));
    float *f0 = (float *)calloc(IMAX*JMAX, sizeof(float)); // Initialize f0 with zeros

    if (!bound || !force || !f0) {
        if (bound) free(bound);
        if (force) free(force);
        if (f0) free(f0);
        throw std::runtime_error("Memory allocation failed in solve_poisson_safety_function_impl");
    }

    memcpy(bound, occ_in, IMAX*JMAX*sizeof(float));
    find_boundary(bound);
    inflate_occupancy_grid(bound, yaw_param);
    find_and_fix_boundary(grid_out, bound);
    compute_boundary_gradients(guidance_x, guidance_y, bound, class_map);

    const float v_RelTol = 1.0e-4f;
    int vx_iters = poisson(guidance_x, f0, bound, v_RelTol, 25.0f);
    int vy_iters = poisson(guidance_y, f0, bound, v_RelTol, 25.0f);

    compute_optimal_forcing_function(force, guidance_x, guidance_y, bound);
    for(int n=0; n<IMAX*JMAX; n++){
        force[n] *= DS*DS;
    }

    const float h_RelTol = 1.0e-4f;
    int iterations = poisson(grid_out, force, bound, h_RelTol, 25.0f);

    // Save results to persistent buffers for next call (warm start)
    memcpy(hgrid, grid_out, IMAX * JMAX * sizeof(float));
    memcpy(guidance_x_buf, guidance_x, IMAX * JMAX * sizeof(float));
    memcpy(guidance_y_buf, guidance_y, IMAX * JMAX * sizeof(float));

    free(bound);
    free(force);
    free(f0);

    return iterations;
};

// Original get_h0 function, unchanged
float get_h0(const float *grid, const float rx, const float ry){

    // Fractional Index Corresponding to Current Position
    const float ir = ry / DS;
    const float jr = rx / DS;
    const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
    const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size
    
    return bilinear_interpolation(grid, ic, jc);

};

/*
// Original main function - commented out as this will be a Python module
int main(void){

    hgrid = (float *)malloc(IMAX*JMAX*sizeof(float));
    for(int n = 0; n < IMAX*JMAX; n++) hgrid[n] = h0;

    // 'occ', 'rx', 'ry', 'yaw' were not defined in the provided snippet for main
    // For example:
    // float occ[IMAX*JMAX]; 
    // // ... initialize occ ...
    // float example_rx = 10.0f, example_ry = 20.0f, example_yaw = 0.0f;

    // solve_poisson_safety_function(hgrid, occ, example_yaw); // Assuming modified signature
    
    // float h0_val = get_h0(hgrid, example_rx, example_ry);
    // std::cout << "h0_val: " << h0_val << " after " << h_iters << " iterations." << std::endl;
    
    free(hgrid);

    return 0;

}
*/

// Pybind11 module definition
namespace py = pybind11;

// Wrapper for solve_poisson_safety_function
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>, int>
py_solve_poisson_safety_function(py::array_t<float> occ_py, float yaw_param, py::object class_map_py = py::none()) {
    py::buffer_info occ_buf = occ_py.request();
    if (occ_buf.ndim != 1 || static_cast<size_t>(occ_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("Input occupancy grid 'occ' must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    const float *occ_ptr = static_cast<const float *>(occ_buf.ptr);

    // Create an output NumPy array for the grid (1D, size IMAX*JMAX)
    auto hgrid_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info hgrid_buf = hgrid_py.request();
    float *hgrid_ptr = static_cast<float *>(hgrid_buf.ptr);

    auto guidance_x_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info guidance_x_buf = guidance_x_py.request();
    float *guidance_x_ptr = static_cast<float *>(guidance_x_buf.ptr);

    auto guidance_y_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info guidance_y_buf = guidance_y_py.request();
    float *guidance_y_ptr = static_cast<float *>(guidance_y_buf.ptr);

    const int* class_map_ptr = nullptr;
    std::vector<int> class_map_vec;
    if (!class_map_py.is_none()) {
        py::array_t<int> class_map_arr = class_map_py.cast<py::array_t<int>>();
        py::buffer_info class_map_buf = class_map_arr.request();
        if (class_map_buf.ndim != 1 || static_cast<size_t>(class_map_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
            throw std::runtime_error("class_map must be 1D int array of size IMAX*JMAX");
        }
        class_map_ptr = static_cast<const int*>(class_map_buf.ptr);
    }

    int iterations_count = solve_poisson_safety_function_impl(
        hgrid_ptr, guidance_x_ptr, guidance_y_ptr, occ_ptr, yaw_param, class_map_ptr);

    return std::make_tuple(hgrid_py, guidance_x_py, guidance_y_py, iterations_count);
}

// Wrapper for flux-based solver
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>, int>
py_solve_poisson_safety_function_flux(py::array_t<float> occ_py, float yaw_param, py::array_t<float> flux_py) {
    py::buffer_info occ_buf = occ_py.request();
    if (occ_buf.ndim != 1 || static_cast<size_t>(occ_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("Input occupancy grid 'occ' must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    const float *occ_ptr = static_cast<const float *>(occ_buf.ptr);

    py::buffer_info flux_buf = flux_py.request();
    if (flux_buf.ndim != 1 || static_cast<size_t>(flux_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("Input flux map 'flux' must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    const float *flux_ptr = static_cast<const float *>(flux_buf.ptr);

    // Create output arrays
    auto hgrid_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info hgrid_buf = hgrid_py.request();
    float *hgrid_ptr = static_cast<float *>(hgrid_buf.ptr);

    auto guidance_x_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info guidance_x_buf = guidance_x_py.request();
    float *guidance_x_ptr = static_cast<float *>(guidance_x_buf.ptr);

    auto guidance_y_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info guidance_y_buf = guidance_y_py.request();
    float *guidance_y_ptr = static_cast<float *>(guidance_y_buf.ptr);

    int iterations_count = solve_poisson_safety_function_flux_impl(
        hgrid_ptr, guidance_x_ptr, guidance_y_ptr, occ_ptr, yaw_param, flux_ptr);

    return std::make_tuple(hgrid_py, guidance_x_py, guidance_y_py, iterations_count);
}

// Wrapper for get_h0
float py_get_h0(py::array_t<float> grid_py, float rx, float ry) {
    py::buffer_info grid_buf = grid_py.request();
    if (grid_buf.ndim != 1 || static_cast<size_t>(grid_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
         throw std::runtime_error("Input grid for 'get_h0' must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    const float *grid_ptr = static_cast<const float *>(grid_buf.ptr);

    return get_h0(grid_ptr, rx, ry); // Call the original C++ function
}

// Wrapper for compute_boundary_mask_from_occ
py::array_t<float> py_compute_boundary_mask_from_occ(py::array_t<float> occ_py) {
    py::buffer_info occ_buf = occ_py.request();
    if (occ_buf.ndim != 1 || static_cast<size_t>(occ_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("Input occupancy grid must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    const float *occ_ptr = static_cast<const float *>(occ_buf.ptr);
    
    auto boundary_mask_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info boundary_buf = boundary_mask_py.request();
    float *boundary_ptr = static_cast<float *>(boundary_buf.ptr);
    
    compute_boundary_mask_from_occ(boundary_ptr, occ_ptr);
    
    return boundary_mask_py;
}

// Wrapper for compute_boundary_normals
std::tuple<py::array_t<float>, py::array_t<float>> py_compute_boundary_normals(py::array_t<float> occ_py) {
    py::buffer_info occ_buf = occ_py.request();
    if (occ_buf.ndim != 1 || static_cast<size_t>(occ_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("Input occupancy grid must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    const float *occ_ptr = static_cast<const float *>(occ_buf.ptr);
    
    auto nx_py = py::array_t<float>(IMAX * JMAX);
    auto ny_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info nx_buf = nx_py.request();
    py::buffer_info ny_buf = ny_py.request();
    float *nx_ptr = static_cast<float *>(nx_buf.ptr);
    float *ny_ptr = static_cast<float *>(ny_buf.ptr);
    
    // Initialize to zero
    for (int n = 0; n < IMAX * JMAX; n++) {
        nx_ptr[n] = 0.0f;
        ny_ptr[n] = 0.0f;
    }
    
    compute_boundary_normals(nx_ptr, ny_ptr, occ_ptr);
    
    return std::make_tuple(nx_py, ny_py);
}

// Wrapper for flux_to_bc
std::tuple<py::array_t<float>, py::array_t<float>> py_flux_to_bc(
    py::array_t<float> flux_py, py::array_t<float> nx_py, py::array_t<float> ny_py) {
    
    py::buffer_info flux_buf = flux_py.request();
    py::buffer_info nx_buf = nx_py.request();
    py::buffer_info ny_buf = ny_py.request();
    
    if (flux_buf.ndim != 1 || static_cast<size_t>(flux_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("flux_map must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    if (nx_buf.ndim != 1 || static_cast<size_t>(nx_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("nx must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    if (ny_buf.ndim != 1 || static_cast<size_t>(ny_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("ny must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    
    const float *flux_ptr = static_cast<const float *>(flux_buf.ptr);
    const float *nx_ptr = static_cast<const float *>(nx_buf.ptr);
    const float *ny_ptr = static_cast<const float *>(ny_buf.ptr);
    
    auto bc_x_py = py::array_t<float>(IMAX * JMAX);
    auto bc_y_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info bc_x_buf = bc_x_py.request();
    py::buffer_info bc_y_buf = bc_y_py.request();
    float *bc_x_ptr = static_cast<float *>(bc_x_buf.ptr);
    float *bc_y_ptr = static_cast<float *>(bc_y_buf.ptr);
    
    flux_to_bc(bc_x_ptr, bc_y_ptr, flux_ptr, nx_ptr, ny_ptr);
    
    return std::make_tuple(bc_x_py, bc_y_py);
}

// Wrapper for solve_laplace_sor
std::tuple<py::array_t<float>, int> py_solve_laplace_sor(
    py::array_t<float> boundary_mask_py, py::array_t<float> bc_values_py,
    float relTol = 1.0e-5f, int max_iters = 50000) {
    
    py::buffer_info boundary_buf = boundary_mask_py.request();
    py::buffer_info bc_buf = bc_values_py.request();
    
    if (boundary_buf.ndim != 1 || static_cast<size_t>(boundary_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("boundary_mask must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    if (bc_buf.ndim != 1 || static_cast<size_t>(bc_buf.size) != static_cast<size_t>(IMAX * JMAX)) {
        throw std::runtime_error("bc_values must be a 1D array of size IMAX*JMAX (" + std::to_string(IMAX*JMAX) + ")");
    }
    
    const float *boundary_ptr = static_cast<const float *>(boundary_buf.ptr);
    const float *bc_ptr = static_cast<const float *>(bc_buf.ptr);
    
    auto v_py = py::array_t<float>(IMAX * JMAX);
    py::buffer_info v_buf = v_py.request();
    float *v_ptr = static_cast<float *>(v_buf.ptr);
    
    // Initialize v to zeros
    for (int n = 0; n < IMAX * JMAX; n++) {
        v_ptr[n] = 0.0f;
    }
    
    int iters = solve_laplace_sor(v_ptr, boundary_ptr, bc_ptr, relTol, max_iters);
    
    return std::make_tuple(v_py, iters);
}

PYBIND11_MODULE(poisson_solver, m) {
    m.doc() = "Python bindings for Poisson safety function solver";

    // Expose getters and setters for global parameters
    m.def("get_IMAX", &get_IMAX, "Get the current IMAX value (Grid X Size)");
    m.def("set_IMAX", &set_IMAX, "Set the IMAX value (Grid X Size)");

    m.def("get_JMAX", &get_JMAX, "Get the current JMAX value (Grid Y Size)");
    m.def("set_JMAX", &set_JMAX, "Set the JMAX value (Grid Y Size)");

    m.def("get_DS", &get_DS, "Get the current DS value (X-Y Grid Resolution)");
    m.def("set_DS", &set_DS, "Set the DS value (X-Y Grid Resolution)");

    m.def("get_h0_val", &get_h0_val, "Get the current h0 value (Boundary Level Set Value)");
    m.def("set_h0_val", &set_h0_val, "Set the h0 value (Boundary Level Set Value)");

    m.def("get_dh0_val", &get_dh0_val, "Get the current dh0 value (Set dh Value)");
    m.def("set_dh0_val", &set_dh0_val, "Set the dh0 value (Set dh Value)");

    m.def("get_dh0_classes", &get_dh0_classes, "Get the list of dh0 values for all object classes");
    m.def("set_dh0_classes", &set_dh0_classes, "Set the list of dh0 values for all object classes");

    // Expose robot kernel parameters
    m.def("get_robot_length", &get_robot_length, "Get the robot length used for kernel inflation");
    m.def("set_robot_length", &set_robot_length, "Set the robot length used for kernel inflation");
    m.def("get_robot_width", &get_robot_width, "Get the robot width used for kernel inflation");
    m.def("set_robot_width", &set_robot_width, "Set the robot width used for kernel inflation");


    m.def("solve_poisson_safety_function", &py_solve_poisson_safety_function,
          "Solves the Poisson safety function. Takes an occupancy grid (1D NumPy array of size IMAX*JMAX), a yaw angle (float), and an optional class_map (1D int NumPy array). Returns a tuple: (solved h-grid (1D NumPy array), guidance_x, guidance_y, number_of_iterations (int)).",
          py::arg("occ_grid"), py::arg("yaw"), py::arg("class_map") = py::none());

    m.def("solve_poisson_safety_function_flux", &py_solve_poisson_safety_function_flux,
          "Solves the Poisson safety function using flux-based boundary conditions following guidance_field.py approach. Takes an occupancy grid (1D NumPy array of size IMAX*JMAX), a yaw angle (float), and a flux map (1D NumPy array of size IMAX*JMAX). Returns a tuple: (solved h-grid (1D NumPy array), guidance_x, guidance_y, number_of_iterations (int)).",
          py::arg("occ_grid"), py::arg("yaw"), py::arg("flux_map"));

    m.def("get_h0", &py_get_h0,
          "Performs bilinear interpolation on a given grid. Takes the grid (1D NumPy array of size IMAX*JMAX), rx (float), and ry (float). Returns the interpolated h-value (float).",
          py::arg("grid"), py::arg("rx"), py::arg("ry"));

    m.def("compute_boundary_mask_from_occ", &py_compute_boundary_mask_from_occ,
          "Computes the boundary mask from an occupancy grid. Takes the grid (1D NumPy array of size IMAX*JMAX). Returns the boundary mask (1D NumPy array of size IMAX*JMAX).",
          py::arg("occ_grid"));
    
    m.def("compute_boundary_normals", &py_compute_boundary_normals,
          "Computes the boundary normal from an occupancy grid. Takes the grid (1D NumPy array of size IMAX*JMAX). Returns the boundary normal (1D NumPy array of size IMAX*JMAX).",
          py::arg("occ_grid"));
    
    m.def("flux_to_bc", &py_flux_to_bc,
          "Converts flux to boundary conditions. Takes the flux (1D NumPy array of size IMAX*JMAX). Returns the boundary conditions (1D NumPy array of size IMAX*JMAX).",
          py::arg("flux"), py::arg("nx"), py::arg("ny"));
    
    m.def("solve_laplace_sor", &py_solve_laplace_sor,
          "Solves the Laplace equation using SOR. Takes the boundary mask (1D NumPy array of size IMAX*JMAX), boundary values (1D NumPy array of size IMAX*JMAX), relative tolerance (float), and maximum iterations (int). Returns a tuple: (solved v-grid (1D NumPy array), number_of_iterations (int)).",
          py::arg("boundary_mask"), py::arg("bc_values"), py::arg("relTol") = 1.0e-5f, py::arg("max_iters") = 50000);

    // Social navigation parameters (ported from semantic_poisson.cpp)
    m.def("get_enable_social_navigation", &get_enable_social_navigation, "Get whether social navigation is enabled (pass on human's right)");
    m.def("set_enable_social_navigation", &set_enable_social_navigation, "Set whether social navigation is enabled (pass on human's right)");
    m.def("get_social_tangent_bias", &get_social_tangent_bias, "Get the social navigation tangential bias strength");
    m.def("set_social_tangent_bias", &set_social_tangent_bias, "Set the social navigation tangential bias strength");
    m.def("get_dh0_human", &get_dh0_human, "Get the dh0 value for human class");
    m.def("set_dh0_human", &set_dh0_human, "Set the dh0 value for human class");
    m.def("get_dh0_obstacle", &get_dh0_obstacle, "Get the dh0 value for obstacle class");
    m.def("set_dh0_obstacle", &set_dh0_obstacle, "Set the dh0 value for obstacle class");
    m.def("get_robot_kernel_dim", &get_robot_kernel_dim, "Get the robot kernel dimension used for human label search radius");
}