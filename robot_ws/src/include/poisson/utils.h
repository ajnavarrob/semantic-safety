#pragma once
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

#define PRINT_TIMING false

class Timer{
public: 
    Timer(bool print);
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::chrono::duration<float> duration;

    void start();
    float time();
    float time(std::string info);

    bool print_;
};

float x_to_j(const float x, const float xc);
float y_to_i(const float y, const float yc);
float j_to_x(const int j, const float xc);
float i_to_y(const int i, const float yc);
float q_to_yaw(const int q, const float yawc);
float yaw_to_q(const float yaw, const float yawc);
float ang_diff(const float a1, const float a2);
float q_wrap(float qi);
void low_pass(std::vector<float>& v_filter, const std::vector<float>& v_new, const float wc, const float dt);
float trilinear_interpolation(const float *grid, const float i, const float j, const float k);
float bilinear_interpolation(const float *grid, const float i, const float j);
int8_t bilinear_interpolation_int8(const int8_t *grid, const float i, const float j);
std::string getCurrentDateTime(void);