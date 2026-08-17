#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "poisson.h"
#include "utils.h"
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Sparse>

#define STATES 3
#define INPUTS 3

class MPC3D {

public:

    MPC3D(void) {

        N_HORIZON = TMAX - 1;

        nX = (N_HORIZON + 1) * STATES;
        nU = N_HORIZON * INPUTS;

        // One scalar alpha variable.
        nAlpha = 1;
        idAlpha = nX + nU;

        nZ = nX + nU + nAlpha;
        nC = nZ + N_HORIZON;

        cost_P = Eigen::MatrixXd::Zero(nZ, nZ);
        cost_q = Eigen::VectorXd::Zero(nZ);

        constraint_A = Eigen::MatrixXd::Zero(nC, nZ);
        constraint_upper = Eigen::VectorXd::Zero(nC);
        constraint_lower = Eigen::VectorXd::Zero(nC);

        sol = Eigen::VectorXd::Zero(nZ);
        xbar = Eigen::VectorXd::Zero(nX);

        Pu.setIdentity(INPUTS, INPUTS);
        Pu.row(0) << 5.0, 0.0, 0.0;
        Pu.row(1) << 0.0, 5.0, 0.0;
        Pu.row(2) << 0.0, 0.0, 1.0;

        for (int k = 0; k < N_HORIZON; k++) {
            const int idu = nX + k * INPUTS;
            cost_P.block<INPUTS, INPUTS>(idu, idu) = Pu;
        }

        cost_P.block<INPUTS, INPUTS>(
            nX + (N_HORIZON - 1) * INPUTS,
            nX + (N_HORIZON - 1) * INPUTS
        ) *= 2.0f;

        // Alpha cost: penalize deviation from alpha_ref.
        cost_P(idAlpha, idAlpha) = alpha_weight_;
        cost_q(idAlpha) = -alpha_weight_ * alpha_ref_;

        // Initial condition constraint.
        constraint_A.block<STATES, STATES>(0, 0).setIdentity();

        for (int k = 0; k < N_HORIZON; k++) {

            const int idx = k * STATES;
            const int idxp1 = (k + 1) * STATES;
            const int idu = nX + k * INPUTS;

            // Dynamics: x_{k+1} = x_k + DT * u_k.
            constraint_A.block<STATES, STATES>(idxp1, idxp1) =
                -Eigen::MatrixXd::Identity(STATES, STATES);

            constraint_A.block<STATES, STATES>(idxp1, idx) =
                Eigen::MatrixXd::Identity(STATES, STATES);

            constraint_A.block<STATES, INPUTS>(idxp1, idu) =
                DT * Eigen::MatrixXd::Identity(STATES, INPUTS);

            // Input saturation constraints.
            constraint_A.block<INPUTS, INPUTS>(idu, idu) =
                Eigen::MatrixXd::Identity(INPUTS, INPUTS);

            constraint_upper.segment(idu, INPUTS) << 0.8f, 0.8f, 0.8f;
            constraint_lower.segment(idu, INPUTS) << -0.8f, -0.8f, -0.8f;

            // Safety constraint rows.
            const int idsf = nZ + k;
            constraint_upper(idsf) = OSQP_INFTY;
            constraint_lower(idsf) = -OSQP_INFTY;
        }

        // Alpha variable row.
        constraint_A(idAlpha, idAlpha) = 1.0;

        // By default, alpha is fixed, so this behaves like normal MPC.
        constraint_lower(idAlpha) = alpha_fixed_;
        constraint_upper(idAlpha) = alpha_fixed_;
    }

    int N_HORIZON;

    int nX;
    int nU;
    int nAlpha;
    int idAlpha;
    int nZ;
    int nC;

    Eigen::MatrixXd Pu;
    Eigen::MatrixXd cost_P;
    Eigen::VectorXd cost_q;

    Eigen::MatrixXd constraint_A;
    Eigen::VectorXd constraint_lower;
    Eigen::VectorXd constraint_upper;

    Eigen::VectorXd sol;
    Eigen::VectorXd xbar;

    OsqpEigen::Solver solver;

    float cost0 = 1.0e23f;
    float cost1 = 1.0e23f;
    float resid = 1.0e23f;

    // Alpha settings.
    float alpha_fixed_ = 0.95f;
    float alpha_min_ = 0.0f;
    float alpha_max_ = 10000000.0f;
    float alpha_ref_ = 0.95f;
    float alpha_weight_ = 10.0f;
    float alpha_solution_ = 0.95f;
    bool optimize_alpha_ = false;

    int setup_QP(void) {

        solver.settings()->setVerbosity(false);
        solver.settings()->setWarmStart(true);
        solver.settings()->setMaxIteration(1000);

        solver.data()->setNumberOfVariables(nZ);
        solver.data()->setNumberOfConstraints(nC);

        Eigen::SparseMatrix<double> cost_P_sparse =
            cost_P.sparseView();

        Eigen::SparseMatrix<double> constraint_A_sparse =
            constraint_A.sparseView();

        if (!solver.data()->setHessianMatrix(cost_P_sparse)) return 1;
        if (!solver.data()->setGradient(cost_q)) return 1;
        if (!solver.data()->setLinearConstraintsMatrix(constraint_A_sparse)) return 1;
        if (!solver.data()->setLowerBound(constraint_lower)) return 1;
        if (!solver.data()->setUpperBound(constraint_upper)) return 1;
        if (!solver.initSolver()) return 1;

        return 0;
    }

    void clear_QP(void) {

        solver.clearSolver();
        solver.data()->clearHessianMatrix();
        solver.data()->clearLinearConstraintsMatrix();
    }

    void update_cost(const std::vector<float> ud) {

        Eigen::VectorXd input_goal(INPUTS);
        input_goal << ud[0], ud[1], ud[2];

        for (int k = 0; k < N_HORIZON; k++) {
            const int idu = nX + k * INPUTS;
            cost_q.segment(idu, INPUTS) = -(Pu * input_goal);
        }

        cost_q(idAlpha) = -alpha_weight_ * alpha_ref_;

        solver.updateGradient(cost_q);
    }

    void set_velocity_bounds(
        float vx_fwd,
        float vx_bwd,
        float vy_max,
        float vyaw_max
    ) {
        for (int k = 0; k < N_HORIZON; k++) {
            const int idu = nX + k * INPUTS;

            constraint_upper.segment(idu, INPUTS)
                << vx_fwd, vy_max, vyaw_max;

            constraint_lower.segment(idu, INPUTS)
                << -vx_bwd, -vy_max, -vyaw_max;
        }
    }

    void set_alpha_bounds(float alpha_min, float alpha_max) {
        alpha_min_ = alpha_min;
        alpha_max_ = alpha_max;
    }

    void set_alpha_optimization_enabled(
        bool enabled,
        float alpha_fixed
    ) {
        optimize_alpha_ = enabled;

        alpha_fixed_ = std::clamp(alpha_fixed, alpha_min_, alpha_max_);
        alpha_ref_ = alpha_fixed_;

        if (optimize_alpha_) {
            constraint_lower(idAlpha) = alpha_min_;
            constraint_upper(idAlpha) = alpha_max_;
        } else {
            constraint_lower(idAlpha) = alpha_fixed_;
            constraint_upper(idAlpha) = alpha_fixed_;
        }

        cost_q(idAlpha) = -alpha_weight_ * alpha_ref_;

        solver.updateGradient(cost_q);
        solver.updateBounds(constraint_lower, constraint_upper);
    }

    float get_alpha() const {
        return alpha_solution_;
    }

    float line_search(
        const float* h_grid,
        const float* dhdt_grid,
        const std::vector<float> xc,
        const float grid_age,
        const float wn
    ) {
        float best_violation = -1.0e10f;
        float best_alpha = 0.0f;

        const int N = 10;
        const float rho = std::exp(-wn * DT);

        for (int n = 0; n <= N; n++) {

            float h[TMAX];
            const float alpha = static_cast<float>(n) / static_cast<float>(N);

            Eigen::VectorXd xbar_test =
                (1.0f - alpha) * xbar + alpha * sol.segment(0, nX);

            float total_violation = 0.0f;

            for (int k = 0; k <= N_HORIZON; k++) {

                const float tk = k * DT + grid_age;

                const float ir = y_to_i(xbar_test(k * STATES + 1), xc[1]);
                const float jr = x_to_j(xbar_test(k * STATES + 0), xc[0]);
                const float qc = yaw_to_q(xbar_test(k * STATES + 2), xc[2]);

                const float ic =
                    std::clamp(ir, 0.0f, static_cast<float>(IMAX - 1));

                const float jc =
                    std::clamp(jr, 0.0f, static_cast<float>(JMAX - 1));

                if ((ir == ic) && (jr == jc)) {
                    h[k] =
                        trilinear_interpolation(h_grid, ic, jc, qc)
                        + tk * trilinear_interpolation(dhdt_grid, ic, jc, qc);
                } else {
                    h[k] =
                        -std::sqrt(
                            (ir - ic) * (ir - ic)
                            + (jr - jc) * (jr - jc)
                        ) * DS;
                }
            }

            for (int k = 0; k < N_HORIZON; k++) {
                total_violation += std::fmin(
                    0.0f,
                    h[k + 1] - rho * h[k]
                );
            }

            if (total_violation >= best_violation) {
                best_violation = total_violation;
                best_alpha = alpha;
            }
        }

        return best_alpha;
    }

    int update_constraints(
        const float* h_grid,
        const float* dhdt_grid,
        const float* beta_grid_,
        const float* guidance_x,
        const float* guidance_y,
        const std::vector<float> x,
        const std::vector<float> xc,
        const float grid_age,
        const float wn,
        const float issf,
        const float sigma_epsilon,
        const float sigma_kappa
    ) {
        (void)beta_grid_;

        // Initial condition.
        constraint_lower.segment(0, STATES) << x[0], x[1], x[2];
        constraint_upper.segment(0, STATES) << x[0], x[1], x[2];

        const float q_eps = 1.0f;

        const float sqp_alpha =
            line_search(h_grid, dhdt_grid, xc, grid_age, wn);

        xbar *= 1.0f - sqp_alpha;
        xbar += sqp_alpha * sol.segment(0, nX);

        const float alpha_nominal = std::exp(-wn * DT);

        for (int k = 0; k <= N_HORIZON; k++) {

            const int idx = k * STATES;
            const int idu = k * INPUTS + nX;

            const float tk = k * DT + grid_age;

            const float rxk = xbar(idx + 0);
            const float ryk = xbar(idx + 1);
            const float yawk = xbar(idx + 2);

            const float ir = y_to_i(ryk, xc[1]);
            const float jr = x_to_j(rxk, xc[0]);
            const float qc = yaw_to_q(yawk, xc[2]);

            const float ic =
                std::clamp(ir, 0.0f, static_cast<float>(IMAX - 1));

            const float jc =
                std::clamp(jr, 0.0f, static_cast<float>(JMAX - 1));

            const float h1 =
                trilinear_interpolation(h_grid, ic, jc, qc);

            const float dhdt =
                trilinear_interpolation(dhdt_grid, ic, jc, qc);

            const float vx =
                trilinear_interpolation(guidance_y, ic, jc, qc);

            const float vy =
                trilinear_interpolation(guidance_x, ic, jc, qc);

            const float v_norm =
                std::sqrt(vx * vx + vy * vy);

            const float h_eps = 1.0f;

            const float hip =
                trilinear_interpolation(h_grid, ic + h_eps, jc, qc);

            const float him =
                trilinear_interpolation(h_grid, ic - h_eps, jc, qc);

            const float hjp =
                trilinear_interpolation(h_grid, ic, jc + h_eps, qc);

            const float hjm =
                trilinear_interpolation(h_grid, ic, jc - h_eps, qc);

            const float Dh_x =
                (hjp - hjm) / (2.0f * h_eps * DS);

            const float Dh_y =
                (hip - him) / (2.0f * h_eps * DS);

            const float qp = q_wrap(qc + q_eps);
            const float qm = q_wrap(qc - q_eps);

            float hqp =
                trilinear_interpolation(h_grid, ic, jc, qp);

            float hqm =
                trilinear_interpolation(h_grid, ic, jc, qm);

            const float dhqpdt =
                trilinear_interpolation(dhdt_grid, ic, jc, qp);

            const float dhqmdt =
                trilinear_interpolation(dhdt_grid, ic, jc, qm);

            hqp += dhqpdt * tk;
            hqm += dhqmdt * tk;

            float dhdyaw =
                (hqp - hqm) / (2.0f * q_eps * DQ);

            float dhdx = vx;
            float dhdy = vy;

            const float Dh_norm =
                std::sqrt(
                    Dh_x * Dh_x
                    + Dh_y * Dh_y
                    + dhdyaw * dhdyaw
                );

            const float sigma_h =
                sigma_epsilon
                * (1.0f - std::exp(
                    -sigma_kappa * std::max(0.0f, h1)
                ));

            const float dhdt_scale =
                std::min(
                    v_norm / (Dh_norm + sigma_h + 1.0e-6f),
                    1.0f
                );

            float h = h1 + dhdt_scale * dhdt * tk;

            if ((ir != ic) || (jr != jc)) {
                h =
                    -std::sqrt(
                        (ir - ic) * (ir - ic)
                        + (jr - jc) * (jr - jc)
                    ) * DS;

                if (jr > jc) dhdx = -1.0f;
                if (jr < jc) dhdx = 1.0f;
                if (ir > ic) dhdy = -1.0f;
                if (ir < ic) dhdy = 1.0f;

                const float norm =
                    std::sqrt(dhdx * dhdx + dhdy * dhdy);

                dhdx /= norm;
                dhdy /= norm;
                dhdyaw = 0.0f;
            }

            // Safety constraints:
            // h_{k+1}(x_{k+1}) - alpha * h_k(x_k) >= ISS term.
            if (k != N_HORIZON) {

                const int idsf = nZ + k;

                constraint_A.block<1, STATES>(idsf, idx)
                    << -alpha_nominal * dhdx,
                       -alpha_nominal * dhdy,
                       -alpha_nominal * dhdyaw;

                // First-order affine dependence on alpha:
                // alpha * h_k(x_k) ≈ alpha_nominal * grad_h * x_k
                //                    + h_k_bar * alpha
                //                    - alpha_nominal * grad_h * xbar_k.
                constraint_A(idsf, idAlpha) = -h;

                constraint_lower(idsf) =
                    -alpha_nominal
                    * (dhdx * rxk + dhdy * ryk + dhdyaw * yawk);

                const float ISSf1 = issf * (0.5f * static_cast<float>(k) + 1.0f);
                const float ISSf2 = issf * (0.5f * static_cast<float>(k) + 1.0f);

                const float Lgh_norm =
                    std::sqrt(dhdx * dhdx + dhdy * dhdy + dhdyaw * dhdyaw);

                constraint_lower(idsf) +=
                    (Lgh_norm / ISSf1 + Lgh_norm * Lgh_norm / ISSf2) * DT;
            }

            if (k != 0) {
                const int idsfm1 = nZ + k - 1;

                constraint_A.block<1, STATES>(idsfm1, idx)
                    << dhdx, dhdy, dhdyaw;

                constraint_lower(idsfm1) +=
                    dhdx * rxk + dhdy * ryk + dhdyaw * yawk - h;
            }

            // Update body-frame input saturation linearization.
            if (k != N_HORIZON) {
                constraint_A.block<1, INPUTS>(idu + 0, idu)
                    << std::cos(yawk), std::sin(yawk), 0.0f;

                constraint_A.block<1, INPUTS>(idu + 1, idu)
                    << -std::sin(yawk), std::cos(yawk), 0.0f;

                constraint_A.block<1, INPUTS>(idu + 2, idu)
                    << 0.0f, 0.0f, 1.0f;
            }
        }

        Eigen::SparseMatrix<double> constraint_A_sparse =
            constraint_A.sparseView();

        solver.updateLinearConstraintsMatrix(constraint_A_sparse);
        solver.updateBounds(constraint_lower, constraint_upper);

        return 1;
    }

    void solve(void) {

        solver.solveProblem();

        OsqpEigen::Status status = solver.getStatus();

        if (
            status == OsqpEigen::Status::Solved ||
            status == OsqpEigen::Status::SolvedInaccurate ||
            status == OsqpEigen::Status::MaxIterReached
        ) {
            sol = solver.getSolution();
            cost1 = solver.getObjValue();
        } else {
            switch (status) {
                case OsqpEigen::Status::PrimalInfeasible:
                    std::cout << "QP Solver Error: Primal Infeasible!" << std::endl;
                    break;

                case OsqpEigen::Status::DualInfeasible:
                    std::cout << "QP Solver Error: Dual Infeasible!" << std::endl;
                    break;

                case OsqpEigen::Status::NonCvx:
                    std::cout << "QP Solver Error: Non-Convex!" << std::endl;
                    break;

                default:
                    std::cout << "QP Solver Error: Unknown status "
                              << static_cast<int>(status)
                              << std::endl;
                    break;
            }

            clear_QP();
            setup_QP();
            cost1 = 1.0e23f;
        }
    }

    float update_residual(void) {

        resid = (cost1 - cost0) * (cost1 - cost0);
        cost0 = cost1;

        return resid;
    }

    void set_input(std::vector<float>& u) {

        u = {
            static_cast<float>(sol(nX + 0)),
            static_cast<float>(sol(nX + 1)),
            static_cast<float>(sol(nX + 2))
        };

        alpha_solution_ = static_cast<float>(sol(idAlpha));
    }
};
