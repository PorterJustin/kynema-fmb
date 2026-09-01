#include <array>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

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
    constexpr auto stiffness_matrix = std::array{
        std::array{1368.17e3, 0., 0., 0., 0., 0.},
        std::array{0., 88.56e3, 0., 0., 0., 0.},
        std::array{0., 0., 38.78e3, 0., 0., 0.},
        std::array{0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
        std::array{0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
        std::array{0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
    };

    // Node locations (GLL quadrature)
    const auto node_s = std::vector{0., 0.17267316464601146, 0.5, 0.8273268353539885, 1.};

    // Create model for managing nodes and constraints
    auto model = Model();

    // Set gravity in model
    model.SetGravity(0., 0., 0.);

    // Build vector of nodes (straight along x axis, no rotation)
    std::vector<size_t> beam_node_ids;
    std::ranges::transform(node_s, std::back_inserter(beam_node_ids), [&](auto s) {
        return model.AddNode()
            .SetElemLocation(s)
            .SetPosition(10 * s, 0., 0., 1., 0., 0., 0.)
            .Build();
    });


    const double scalar_mu(0.0001);  // 1/s

    // Add beam element
    model.AddBeamElement(
        beam_node_ids,
        std::array{
            BeamSection(0., mass_matrix, stiffness_matrix),
            BeamSection(1., mass_matrix, stiffness_matrix),
        },
        std::array{
            std::array{-0.9491079123427585, 0.1294849661688697},
            std::array{-0.7415311855993943, 0.27970539148927664},
            std::array{-0.40584515137739696, 0.3818300505051189},
            std::array{6.123233995736766e-17, 0.4179591836734694},
            std::array{0.4058451513773971, 0.3818300505051189},
            std::array{0.7415311855993945, 0.27970539148927664},
            std::array{0.9491079123427585, 0.1294849661688697},
        },
        std::array{scalar_mu, scalar_mu, scalar_mu, scalar_mu, scalar_mu, scalar_mu}
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

    // Convert sparse CRS values to dense matrices and save to CSV files.
    const auto num_dofs = solver.num_dofs;
    const auto row_map = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, solver.A.graph.row_map);
    const auto col_ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, solver.A.graph.entries);
    const auto mass_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrices.mass_matrix_values);
    const auto stiff_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrices.stiffness_matrix_values);
    const auto damp_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrices.damping_matrix_values);
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
