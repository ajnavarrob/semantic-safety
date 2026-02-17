#pragma once
#ifndef KERNEL_H
#define KERNEL_H

#include <vector>

namespace Kernel{
    void poissonInit(void);
    int poissonSolve(float *h, const float *f, const float *b, const float relTol, const float w_SOR);
}

#endif