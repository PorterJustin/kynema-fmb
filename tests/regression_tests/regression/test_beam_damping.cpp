#include <array>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "model/model.hpp"
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

namespace kynema::tests {

TEST(DynamicBeamTest, Damping) {
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
    const double step_size(0.0001);  // seconds
    const double rho_inf(1.0);
    const int num_steps(1000);

    // Create solver parameters
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Eventually will want this to only track the tip displacement
    // // Get ID of last node
    // const auto last_node_id = beam_node_ids.back();

    // Ideally would pull from an eigenanalysis to get the mode shapes
    // and initial velocity, just prototyping here.
    // (x/L)^3 is a rough approx of first bending mode,
    // technically need rotational velocity as well

    // Start the beam with an initial velocity in the first component
    for (size_t i = 0; i < beam_node_ids.size(); ++i) {
        auto node_v = Kokkos::subview(state.v, beam_node_ids[i], Kokkos::ALL);
        node_v(0) = node_s[i] * node_s[i] * node_s[i];
        node_v(1) = 0.;
        node_v(2) = 0.;
        node_v(3) = 0.;
        node_v(4) = 0.;
        node_v(5) = 0.;
    }

    // Time Stepping Loop for num_steps time steps saving the response at each step

    std::vector<std::vector<double>> displacement_history;
    displacement_history.reserve(num_steps);
    std::vector<std::vector<double>> velocity_history;
    velocity_history.reserve(num_steps);

    for ([[maybe_unused]] auto i : std::views::iota(0, num_steps)) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);

        const auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q);
        auto& displacement_row = displacement_history.emplace_back();
        displacement_row.reserve(1 + q_host.extent(0) * q_host.extent(1));
        displacement_row.push_back(static_cast<double>(state.time_step) * step_size);
        for (size_t node = 0; node < q_host.extent(0); ++node) {
            for (size_t dof = 0; dof < q_host.extent(1); ++dof) {
                displacement_row.push_back(q_host(node, dof));
            }
        }

        const auto v_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.v);
        auto& velocity_row = velocity_history.emplace_back();
        velocity_row.reserve(1 + v_host.extent(0) * v_host.extent(1));
        velocity_row.push_back(static_cast<double>(state.time_step) * step_size);
        for (size_t node = 0; node < v_host.extent(0); ++node) {
            for (size_t dof = 0; dof < v_host.extent(1); ++dof) {
                velocity_row.push_back(v_host(node, dof));
            }
        }
    }

    // WriteMatrixToFile(displacement_history, "beam_damping_displacement_history.csv");
    // WriteMatrixToFile(velocity_history, "beam_damping_velocity_history.csv");

    // Calculate the damping values based on log decrement
    // and verify against the analytical stiffness proportional damping

    // This should match the direction of the initial velocity
    constexpr size_t dir_ind = 0;

    // 3 translations + 4 quaternions per node
    const auto tip_displacement_col = 1 + dir_ind + 7*(beam_node_ids.size() - 1);

    std::vector<size_t> peak_inds;
    // Assume that should have at least 50 steps per cycle
    // for sizing the memory (100 is rule of thumb to be accurate)
    peak_inds.reserve(num_steps / 50);

    for (size_t i = 1; i + 1 < displacement_history.size(); ++i) {
        const auto prev = displacement_history[i - 1][tip_displacement_col];
        const auto curr = displacement_history[i][tip_displacement_col];
        const auto next = displacement_history[i + 1][tip_displacement_col];
        if (curr > 0. && curr > prev && curr > next) {
            // Require peaks > 0 to avoid undefined log below.
            peak_inds.push_back(i);
        }
    }

    // Need at least two peaks for log decrement
    ASSERT_GE(peak_inds.size(), 2U);

    std::vector<double> peak_vals;
    peak_vals.reserve(peak_inds.size());

    // Peak times are really just (peak_inds + 1) * step_size
    std::vector<double> peak_times;
    peak_times.reserve(peak_inds.size());
    
    for (const auto peak_ind : peak_inds) {
        peak_vals.push_back(displacement_history[peak_ind][tip_displacement_col]);
        peak_times.push_back(displacement_history[peak_ind][0]);
    }

    std::vector<double> zeta;
    zeta.reserve(peak_vals.size() - 1);

    std::vector<double> omega_n;
    omega_n.reserve(peak_vals.size() - 1);

    constexpr auto two_pi = 2. * std::numbers::pi;
    
    for (size_t i = 0; i + 1 < peak_vals.size(); ++i) {

        const auto delta_i = std::log(peak_vals[i] / peak_vals[i + 1]);
        
        const auto zeta_i = delta_i / std::sqrt(two_pi * two_pi +
                                    delta_i * delta_i);

        const auto omega_n_i = (two_pi / (peak_times[i + 1] - peak_times[i]))
                        / std::sqrt(1. - zeta_i * zeta_i);

        zeta.push_back(zeta_i);
        omega_n.push_back(omega_n_i);
    }

    // Check only the last few damping values starting from the end
    // The initialization is imperfect, so the early ones have other transients
    // If above gets updated to an eigensolution for the initialization
    // then should be able to run fewer cycles and immediately evaluate
    // the damping and at tighter tolerance.
    for (size_t eval_count = 0, idx = zeta.size() - 1;
            eval_count < 3 && idx > 4;
            ++eval_count, --idx) 
    {
        ASSERT_NEAR(zeta[idx], scalar_mu * omega_n[idx] / 2.0, 1e-3*zeta[idx]);
    }
}

}  // namespace kynema::tests
