#include "utils.h"
#include "poisson.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>

Timer::Timer(bool print) : print_(print){}

void Timer::start() {
    start_time = std::chrono::steady_clock::now();
}

float Timer::time() {
    end_time = std::chrono::steady_clock::now();
    duration = end_time - start_time;
    float dur_ms =  duration.count()*1.0e3f;
    if (print_)
        std::cout << dur_ms << " ms" << std::endl;
    start_time = end_time;
    return dur_ms;
}

float Timer::time(std::string info) {
    if (print_)
        std::cout << info;
    return time();
}

/* Compute difference between two angles wrapped between [-pi, pi] */
float ang_diff(const float a1, const float a2){
    
    float a3 = a1 - a2;
    while(a3 <= -M_PI){
        a3 += 2.0f*M_PI;
    }
    while(a3 > M_PI){
        a3 -= 2.0f*M_PI;
    }
    return a3;

}

/* Wrap Angle Indices */
float q_wrap(float qi){

    float qf = qi;
    while(qf < 0.0f) qf += (float)QMAX;
    while(qf >= (float)QMAX) qf -= (float)QMAX;
    return qf;

}

/* Convert x Coordinate to j Index*/
float x_to_j(const float x, const float xc){

    return (x-xc) / DS;

}
  
/* Convert y Coordinate to i Index */
float y_to_i(const float y, const float yc){

    return (y-yc) / DS;

}

/* Convert j Index to x Coordinate (inverse of x_to_j) */
float j_to_x(const int j, const float xc){

    return (float)j * DS + xc;

}

/* Convert i Index to y Coordinate (inverse of y_to_i) */
float i_to_y(const int i, const float yc){

    return (float)i * DS + yc;

}

/* Convert q Index to yaw Coordinate */
float q_to_yaw(const int q, const float yawc){

    return ang_diff((float)q*DQ, -yawc);

}

/* Convert yaw Coordinate to q Index */
float yaw_to_q(const float yaw, const float yawc){

    return q_wrap((yaw-yawc) / DQ);

}

/* Low Pass Filter */
void low_pass(std::vector<float>& v_filter, const std::vector<float>& v_new, const float wc, const float dt){

    const float kc = 1.0f - std::exp(-wc*dt);
    for(char i=0; i<v_filter.size(); i++){
        v_filter[i] *= (1.0f - kc);
        v_filter[i] += kc * v_new[i];
    }

}

/* Perform a trilinear interpolation on a 3-D grid */
float trilinear_interpolation(const float *grid, const float i, const float j, const float k){

    const float i1f = std::floor(i);
    const float j1f = std::floor(j);
    const float k1f = std::floor(k);
    const float i2f = std::ceil(i);
    const float j2f = std::ceil(j);
    const float k2f = std::ceil(k);

    const int i1 = (int)i1f;
    const int j1 = (int)j1f;
    const int k1 = (int)q_wrap(k1f);
    const int i2 = (int)i2f;
    const int j2 = (int)j2f;
    const int k2 = (int)q_wrap(k2f);

    if((i1 != i2) && (j1 != j2) && (k1 != k2)){
        const float f11 = (k2f - k) * grid[k1*IMAX*JMAX+i1*JMAX+j1] + (k - k1f) * grid[k2*IMAX*JMAX+i1*JMAX+j1];
        const float f12 = (k2f - k) * grid[k1*IMAX*JMAX+i1*JMAX+j2] + (k - k1f) * grid[k2*IMAX*JMAX+i1*JMAX+j2];
        const float f21 = (k2f - k) * grid[k1*IMAX*JMAX+i2*JMAX+j1] + (k - k1f) * grid[k2*IMAX*JMAX+i2*JMAX+j1];
        const float f22 = (k2f - k) * grid[k1*IMAX*JMAX+i2*JMAX+j2] + (k - k1f) * grid[k2*IMAX*JMAX+i2*JMAX+j2];
        const float f1 = (i2f - i) * f11 + (i - i1f) * f21;
        const float f2 = (i2f - i) * f12 + (i - i1f) * f22;
        return (j2f - j) * f1 + (j - j1f) * f2;
    }
    else if((i1 != i2) && (k1 != k2)){
        const float f1 = (i2f - i) * grid[k1*IMAX*JMAX+i1*JMAX+(int)j] + (i - i1f) * grid[k1*IMAX*JMAX+i2*JMAX+(int)j];
        const float f2 = (i2f - i) * grid[k2*IMAX*JMAX+i1*JMAX+(int)j] + (i - i1f) * grid[k2*IMAX*JMAX+i2*JMAX+(int)j];
        return (k2f - k) * f1 + (k - k1f) * f2;
    }
    else if((j1 != j2) && (k1 != k2)){
        const float f1 = (j2f - j) * grid[k1*IMAX*JMAX+(int)i*JMAX+j1] + (j - j1f) * grid[k1*IMAX*JMAX+(int)i*JMAX+j2];
        const float f2 = (j2f - j) * grid[k2*IMAX*JMAX+(int)i*JMAX+j1] + (j - j1f) * grid[k2*IMAX*JMAX+(int)i*JMAX+j2];
        return (k2f - k) * f1 + (k - k1f) * f2;
    }
    else if((i1 != i2) && (j1 != j2)){
        const float f1 = (i2f - i) * grid[(int)k*IMAX*JMAX+i1*JMAX+j1] + (i - i1f) * grid[(int)k*IMAX*JMAX+i2*JMAX+j1];
        const float f2 = (i2f - i) * grid[(int)k*IMAX*JMAX+i1*JMAX+j2] + (i - i1f) * grid[(int)k*IMAX*JMAX+i2*JMAX+j2];
        return (j2f - j) * f1 + (j - j1f) * f2;
    }
    else if(k1 != k2){
        return (k2f - k) * grid[k1*IMAX*JMAX+(int)i*JMAX+(int)j] + (k - k1f) * grid[k2*IMAX*JMAX+(int)i*JMAX+(int)j];
    }
    else if(i1 != i2){
        return (i2f - i) * grid[(int)k*IMAX*JMAX+i1*JMAX+(int)j] + (i - i1f) * grid[(int)k*IMAX*JMAX+i2*JMAX+(int)j];
    }
    else if(j1 != j2){
        return (j2f - j) * grid[(int)k*IMAX*JMAX+(int)i*JMAX+j1] + (j - j1f) * grid[(int)k*IMAX*JMAX+(int)i*JMAX+j2];
    }
    else{
        return grid[(int)k*IMAX*JMAX+(int)i*JMAX+(int)j];
    }

}

/* Perform a bilinear interpolation on a 2-D grid */
float bilinear_interpolation(const float *grid, const float i, const float j){

    const float i1f = std::floor(i);
    const float j1f = std::floor(j);
    const float i2f = std::ceil(i);
    const float j2f = std::ceil(j);

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

}

int8_t bilinear_interpolation_int8(const int8_t *grid, const float i, const float j){

    const float i1f = std::floor(i);
    const float j1f = std::floor(j);
    const float i2f = std::ceil(i);
    const float j2f = std::ceil(j);

    const int i1 = (int)i1f;
    const int j1 = (int)j1f;
    const int i2 = (int)i2f;
    const int j2 = (int)j2f;

    if((i1 != i2) && (j1 != j2)){
        const float f1 = (i2f - i) * (float)grid[i1*JMAX+j1] + (i - i1f) * (float)grid[i2*JMAX+j1];
        const float f2 = (i2f - i) * (float)grid[i1*JMAX+j2] + (i - i1f) * (float)grid[i2*JMAX+j2];
        return (int)std::round((j2f - j) * f1 + (j - j1f) * f2);
    }
    else if(i1 != i2){
        return (int)std::round((i2f - i) * (float)grid[i1*JMAX+(int)j] + (i - i1f) * (float)grid[i2*JMAX+(int)j]);
    }
    else if(j1 != j2){
        return (int)std::round((j2f - j) * (float)grid[(int)i*JMAX+j1] + (j - j1f) * (float)grid[(int)i*JMAX+j2]);
    }
    else{
        return grid[(int)i*JMAX+(int)j];
    }

}

std::string getCurrentDateTime(void){
    
    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // Convert to tm struct
    std::tm now_tm;
    localtime_r(&now_time_t, &now_tm); // Linux/macOS

    // Format date/time string
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();

}