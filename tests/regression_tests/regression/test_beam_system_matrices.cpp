#include <array>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "math/gl_quadrature.hpp"
#include "math/gll_quadrature.hpp"
#include "model/model.hpp"
#include "step/extract_system_matrices.hpp"
#include "step/step.hpp"
#include "test_utilities.hpp"

namespace {
template <typename T>
void WriteMatrixToFile(const std::vector<std::vector<T>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << "\n";
        return;
    }
    for (const auto& innerVector : data) {
        for (const auto& element : innerVector) {
            file << element << ",";
        }
        file << "\n";
    }
    file.close();
}

}  // namespace

namespace kynema_fmb::tests {

TEST(DynamicBeamTest, SystemMatrices) {

    const double length(10.0);

    // Mass matrix for uniform composite beam section
    constexpr auto mass_matrix = std::array{
        std::array{8.538e-2, 0., 0., 0., 0., 0.},
        std::array{0., 8.538e-2, 0., 0., 0., 0.},
        std::array{0., 0., 8.538e-2, 0., 0., 0.},
        std::array{0., 0., 0., 1.4433e-2, 0., 0.},
        std::array{0., 0., 0., 0., 0.40972e-2, 0.},
        std::array{0., 0., 0., 0., 0., 1.0336e-2},
    };

    // Stiffness matrix for uniform composite beam section
    // Diagonal with high shear stiffness to match Euler-Bernoulli Theory
    constexpr auto stiffness_matrix = std::array{
        std::array{1368.17e3, 0., 0., 0., 0., 0.},
        std::array{0., 88.56e3*1e9, 0., 0., 0., 0.},
        std::array{0., 0., 38.78e3*1e9, 0., 0., 0.},
        std::array{0., 0., 0., 16.9600e3, 0., 0.},
        std::array{0., 0., 0., 0., 59.1200e3, 0.},
        std::array{0., 0., 0., 0., 0., 141.470e3},
    };

    // Node locations (GLL quadrature)
    const auto num_nodes = 10UL;
    const auto gll_locations = math::GetGllLocations(num_nodes - 1);
    std::vector<double> node_s(gll_locations.size());
    std::ranges::transform(gll_locations, node_s.begin(), [](auto xi) { return 0.5 * (xi + 1.0); });

    // Create model for managing nodes and constraints
    auto model = Model();

    // Set gravity in model
    model.SetGravity(0., 0., 0.);

    // Build vector of nodes (straight along x axis, no rotation)
    std::vector<size_t> beam_node_ids;
    std::ranges::transform(node_s, std::back_inserter(beam_node_ids), [&](auto s) {
        return model.AddNode()
            .SetElemLocation(s)
            .SetPosition(length * s, 0., 0., 1., 0., 0., 0.)
            .Build();
    });

    const double scalar_mu(0.0001);  // 1/s

    // const auto array_mu = std::array{0.0001, 0.0004, 0.0002,
    //                                  0.0003, 0.0002, 0.0004};
    const auto array_mu = std::array{0.0001, 0.0001, 0.0001,
                                     0.0001, 0.0001, 0.0001};

    const auto quad_order = 20UL;
    const auto gl_locations = math::GetGlLocations(quad_order);
    const auto gl_weights = math::GetGlWeights(quad_order);

    std::vector<std::array<double, 2>> quad_points(gl_locations.size());
    for (size_t i = 0; i < gl_locations.size(); ++i) {
        quad_points[i] = {gl_locations[i], gl_weights[i]};
    }

    // Add beam element
    model.AddBeamElement(
        beam_node_ids,
        std::array{
            BeamSection(0., mass_matrix, stiffness_matrix),
            BeamSection(1., mass_matrix, stiffness_matrix),
        },
        quad_points,
        array_mu
    );

    // Fix first node position
    model.AddFixedBC(beam_node_ids[0]);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const size_t max_iter(5);
    const double step_size(0.001);  // seconds
    const double rho_inf(1.0);
    const int num_steps(1000);

    // Create solver parameters
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Test that the tangent operator is correctly handled (not included)
    // if q_delta is 0, then tangent operator is identity, so need to change it for test.
    auto q_delta_host = Kokkos::create_mirror_view(state.q_delta);
    Kokkos::deep_copy(q_delta_host, state.q_delta);
    for (size_t i = 0; i < state.num_system_nodes; ++i) {
        for (int j = 3; j < 6; ++j) {
            q_delta_host(i, j) = 1.0;
        }
    }
    Kokkos::deep_copy(state.q_delta, q_delta_host);

    // Extract the system matrices here to verify them.
    auto matrices = step::ExtractSystemMatrices(parameters, solver, elements, state, constraints);

    // Check total mass and rotational inertia.
    // Add all entries in translation in a given direction or axial rotation to compare.
    const auto row_map = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, solver.A.graph.row_map);
    const auto mass_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrices.mass_matrix_values);

    std::array<double, 6> total_inertia{0., 0., 0., 0., 0., 0.};

    for (size_t i = 0; i < 6; ++i) {
        for (size_t n = 0; n < node_s.size(); ++n) {
            const auto row = i + 6 * n;
            for (auto j = row_map(row); j < row_map(row + 1); ++j) {
                total_inertia[i] += mass_vals(j);
            }
        }
    }

    // Translational mass in x,y,z
    ASSERT_NEAR(total_inertia[0], mass_matrix[0][0]*length, 1e-12*mass_matrix[0][0]*length);
    ASSERT_NEAR(total_inertia[1], mass_matrix[1][1]*length, 1e-12*mass_matrix[1][1]*length);
    ASSERT_NEAR(total_inertia[2], mass_matrix[2][2]*length, 1e-12*mass_matrix[2][2]*length);

    // Rotational Inertia
    ASSERT_NEAR(total_inertia[3], mass_matrix[3][3]*length, 1e-12*mass_matrix[3][3]*length);
    ASSERT_NEAR(total_inertia[4], mass_matrix[4][4]*length, 1e-12*mass_matrix[4][4]*length);
    ASSERT_NEAR(total_inertia[5], mass_matrix[5][5]*length, 1e-12*mass_matrix[5][5]*length);

    // Check natural frequencies (undamped)
    const auto col_ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, solver.A.graph.entries);
    const auto stiff_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrices.stiffness_matrix_values);
    const auto damp_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrices.damping_matrix_values);

    // Manually applying the root node constraint by eliminating the first node.
    const auto num_reduced_dofs = 6 * (node_s.size() - 1);

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_reduced_dofs, num_reduced_dofs);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(num_reduced_dofs, num_reduced_dofs);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(num_reduced_dofs, num_reduced_dofs);
    for (size_t row = 6; row < num_reduced_dofs + 6; ++row) {
        for (auto j = row_map(row); j < row_map(row + 1); ++j) {
            const auto col = static_cast<size_t>(col_ids(j));
            if (col >= 6 && col < num_reduced_dofs + 6) {
                M(row - 6, col - 6) = mass_vals(j);
                K(row - 6, col - 6) = stiff_vals(j);
                C(row - 6, col - 6) = damp_vals(j);
            }
        }
    }

    const Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> eigensolver(K, M);
    const auto raw_eigenvalues = eigensolver.eigenvalues();
    const auto raw_eigenvectors = eigensolver.eigenvectors();

    // Reorder the eigenvalues and eigenvectors so the real parts are ascending.
    // Afterwards, the eigenvalues should still be in an array and the eigenvectors in a matrix
    std::vector<Eigen::Index> indices(raw_eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](Eigen::Index i, Eigen::Index j) {
        return raw_eigenvalues(i).real() < raw_eigenvalues(j).real();
    });

    Eigen::VectorXcd eigenvalues(raw_eigenvalues.size());
    Eigen::MatrixXcd eigenvectors(raw_eigenvectors.rows(), raw_eigenvectors.cols());
    for (Eigen::Index i = 0; i < raw_eigenvalues.size(); ++i) {
        eigenvalues(i) = raw_eigenvalues(indices[i]);
        eigenvectors.col(i) = raw_eigenvectors.col(indices[i]);
    }

    // Assertions for natural frequencies (undamped)
    // Tests are against numerical eigenvalues (omega**2) with 10 nodes.
    // Analytical frequencies and eigenvalues are noted in comments.

    // Bending in y
    // Analytical (rad/s): [3.52, 22.0, 61.7] * sqrt(stiffness_matrix[4][4] / (mass[1][1] * length^4))
    // omega = [ 29.29083827, 183.06773922, 513.4217959]
    // omega**2 = [857.95320656,  33513.79714312, 263601.94050518]
    ASSERT_NEAR(eigenvalues[0].real(), 853.86, 1.0);
    ASSERT_NEAR(eigenvalues[3].real(), 33103.8, 10.0);
    ASSERT_NEAR(eigenvalues[5].real(), 254151., 100.0);

    // Bending in z
    // Analytical (rad/s): omega = [45.31028199, 283.18926242, 794.2171587]
    // omega**2 = [2053.02165401,  80196.15834998, 630780.8951735]
    ASSERT_NEAR(eigenvalues[1].real(), 2036.71, 1.0);
    ASSERT_NEAR(eigenvalues[4].real(), 77406.1, 10.0);
    ASSERT_NEAR(eigenvalues[8].real(), 576762., 100.0);

    // Torsion
    // Analytical (rad/s): [0.5, 1.5, 2.5]* pi * sqrt(stiffness_matrix[3][3] / (mass[3][3] * length^2))
    // omega = [170.27641391, 510.82924172, 851.38206954]
    // omega**2 = [28994.05713405, 260946.51419623, 724851.42833421]
    ASSERT_NEAR(eigenvalues[2].real(), 28994.1, 1.0);
    ASSERT_NEAR(eigenvalues[6].real(), 260947., 10.0);
    ASSERT_NEAR(eigenvalues[9].real(), 724852., 10.0);

    // Axial
    // Analytical (rad/s): [0.5, 1.5, 2.5]* pi * sqrt(stiffness_matrix[0][0] / (mass[0][0] * length^2))
    // omega = [628.79898715, 1886.39696145, 3143.99493575]
    // omega**2 = [395388.16624087, 3558493.49616779, 9884704.15602165]
    ASSERT_NEAR(eigenvalues[7].real(), 395388., 1.0);
    ASSERT_NEAR(eigenvalues[15].real(), 3.55849e6, 10.0);
    ASSERT_NEAR(eigenvalues[20].real(), 9.88471e6, 10.0);


    // Check damping ratios from matrix
    double omega;
    double scalar_mu_mode;
    Eigen::Index max_idx;
    Eigen::VectorXd eigvec;

    double zeta_mu;
    double modal_damping;
    double zeta_matrix;

    for (size_t i = 0; i < 10; ++i) {

        eigvec = eigenvectors.col(i).real();

        // Find which mode direction and then the analytical damping from mu
        max_idx = 0;
        eigvec.head(6).cwiseAbs().maxCoeff(&max_idx);

        scalar_mu_mode = array_mu[static_cast<size_t>(max_idx)];
        omega = std::sqrt(eigenvalues[i].real());
        zeta_mu = scalar_mu_mode * omega / 2.0;

        // Damping from the extracted matrix
        modal_damping = eigvec.dot(C * eigvec) / eigvec.dot(M * eigvec);
        zeta_matrix = modal_damping / (2.0 * omega);

        ASSERT_NEAR(zeta_mu, zeta_matrix, 1e-6 * zeta_mu);
    }


    // THIS IS JUST FOR EXTERNAL DEBUGGING, TEST SHOULD FINISH ABOVE THIS.
    
    // Save the eigenvalues to a csv file
    std::ofstream ev_file("eigenvalues.csv");
    if (ev_file.is_open()) {
        for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
            ev_file << eigenvalues(i).real() << "," << eigenvalues(i).imag() << "\n";
        }
        ev_file.close();
    }

    // save the eigenvectors to a csv file
    std::ofstream vec_file("eigenvectors.csv");
    if (vec_file.is_open()) {
        for (Eigen::Index r = 0; r < eigenvectors.rows(); ++r) {
            for (Eigen::Index c = 0; c < eigenvectors.cols(); ++c) {
                vec_file << eigenvectors(r, c).real() << ",";
            }
            vec_file << "\n";
        }
        vec_file.close();
    }

    // Convert sparse CRS values to dense matrices and save to CSV files.
    const auto num_dofs = solver.num_dofs;
    const auto constr_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrices.constraint_matrix_values);

    auto to_dense = [&](const auto& vals) {
        std::vector<std::vector<double>> dense(num_dofs, std::vector<double>(num_dofs, 0.));
        for (size_t row = 0; row < num_dofs; ++row) {
            for (auto j = row_map(row); j < row_map(row + 1); ++j) {
                dense[row][static_cast<size_t>(col_ids(j))] = vals(j);
            }
        }
        return dense;
    };

    WriteMatrixToFile(to_dense(mass_vals), "mass_matrix.csv");
    WriteMatrixToFile(to_dense(stiff_vals), "stiffness_matrix.csv");
    WriteMatrixToFile(to_dense(damp_vals), "damping_matrix.csv");
    WriteMatrixToFile(to_dense(constr_vals), "constraint_matrix.csv");
    

}

}  // namespace kynema_fmb::tests
