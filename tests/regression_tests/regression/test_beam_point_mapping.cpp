/*
This file contains a regression test that demonstrates how a point in space
should be moved to have a consistent mapping to a deformed beam.

Steps:
    1. A beam should be defined with a fixed node at the origin and an
    initial orientation with a non-trivial quaternion.
    2. A second beam should be defined at a new rotation and with the base fixed
    to the tip node of the first beam (all 6-DOFs constrained or reuse the same node).
    3. An initial static solve should be performed to verify no deformation.
    4. A moment vector with all three components should be applied to the tip node of
    the first beam and the static problem resolved.
    5. Verify that the new tip node of the second beam is combination of translation of
    tip node of first beam plus a rotation of the initial position vector by solely
    the displacement quaternion of the tip node of the first beam.
*/

#include <array>
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "math/gl_quadrature.hpp"
#include "math/interpolation.hpp"
#include "math/quaternion_operations.hpp"
#include "model/model.hpp"
#include "step/step.hpp"

namespace kynema_fmb::tests {

namespace {

/// Normalize a quaternion so it represents a valid rotation
std::array<double, 4> NormalizeQuaternion(std::array<double, 4> q) {
    const auto norm = std::sqrt((q[0] * q[0]) + (q[1] * q[1]) + (q[2] * q[2]) + (q[3] * q[3]));
    return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}

/// Build the {point, weight} Gauss-Legendre quadrature rule on [-1, 1] with n points
std::vector<std::array<double, 2>> GaussLegendreQuadrature(size_t n) {
    const auto locations = math::GetGlLocations(n);
    const auto weights = math::GetGlWeights(n);
    auto quadrature = std::vector<std::array<double, 2>>{};
    quadrature.reserve(n);
    for (auto i = 0U; i < n; ++i) {
        quadrature.push_back({locations[i], weights[i]});
    }
    return quadrature;
}

}  // namespace

TEST(BeamPointMappingTest, RigidFollowerBeam) {
    // Solver device type
    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

    // Uniform section properties (mass is irrelevant for a static solve, but required)
    constexpr auto mass_matrix = std::array{
        std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
        std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
        std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.},
    };
    constexpr auto stiffness_matrix = std::array{
        std::array{1770.e3, 0., 0., 0., 0., 0.}, std::array{0., 1770.e3, 0., 0., 0., 0.},
        std::array{0., 0., 1770.e3, 0., 0., 0.}, std::array{0., 0., 0., 8.16e3, 0., 0.},
        std::array{0., 0., 0., 0., 86.9e3, 0.},  std::array{0., 0., 0., 0., 0., 215.e3},
    };

    // 10th order beam elements: 11 GLL nodes, Gauss-Legendre quadrature
    constexpr size_t element_order{10};
    const auto sections = std::vector{
        BeamSection(0., mass_matrix, stiffness_matrix),
        BeamSection(1., mass_matrix, stiffness_matrix),
    };
    const auto quadrature = GaussLegendreQuadrature(12);

    // Non-trivial initial orientations for the two beams
    const auto q_beam1 = NormalizeQuaternion({0.9, 0.1, 0.2, 0.3});
    const auto q_beam2 = NormalizeQuaternion({0.8, -0.3, 0.15, 0.25});
    constexpr auto origin = std::array{0., 0., 0.};

    // Create model with no gravity so the follower beam carries no load
    auto model = Model();
    model.SetGravity(0., 0., 0.);

    // Build a straight beam of the given length along the local x-axis using GLL node locations
    const auto build_straight_beam = [&](double length) {
        const auto gll_points = math::GenerateGLLPoints(element_order);
        auto node_ids = std::vector<size_t>{};
        node_ids.reserve(gll_points.size());
        for (const auto xi : gll_points) {
            const auto s = (xi + 1.) / 2.;
            node_ids.push_back(model.AddNode()
                                   .SetElemLocation(s)
                                   .SetPosition(length * s, 0., 0., 1., 0., 0., 0.)
                                   .Build());
        }
        return node_ids;
    };

    // Step 1: First beam, fixed at the origin, rotated to a non-trivial orientation
    constexpr double beam1_length{10.};
    const auto beam1_node_ids = build_straight_beam(beam1_length);
    const auto beam1_elem_id = model.AddBeamElement(beam1_node_ids, sections, quadrature);
    model.RotateBeamAboutPoint(beam1_elem_id, q_beam1, origin);

    const auto beam1_tip_id = beam1_node_ids.back();
    const auto beam1_tip_position = std::array{
        model.GetNode(beam1_tip_id).x0[0],
        model.GetNode(beam1_tip_id).x0[1],
        model.GetNode(beam1_tip_id).x0[2],
    };

    // Step 2: Second beam, rotated to its own orientation and moved so its base
    // coincides with the tip of the first beam
    constexpr double beam2_length{5.};
    const auto beam2_node_ids = build_straight_beam(beam2_length);
    const auto beam2_elem_id = model.AddBeamElement(beam2_node_ids, sections, quadrature);
    model.RotateBeamAboutPoint(beam2_elem_id, q_beam2, origin);
    model.TranslateBeam(beam2_elem_id, beam1_tip_position);

    const auto beam2_base_id = beam2_node_ids.front();
    const auto beam2_tip_id = beam2_node_ids.back();

    // Fix the base of the first beam and rigidly join the two beams at the shared location
    model.AddFixedBC(beam1_node_ids.front());
    model.AddRigidJointConstraint(std::array{beam1_tip_id, beam2_base_id});

    // Static solve parameters
    constexpr bool is_dynamic_solve{false};
    constexpr size_t max_iter{20};
    constexpr double step_size{1.};
    constexpr double rho_inf{0.};
    constexpr double a_tol{1e-8};
    constexpr double r_tol{1e-8};
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf, a_tol, r_tol);

    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver<DeviceType>();

    // Step 3: Initial static solve with no applied loads should produce no deformation
    auto converged = Step(parameters, solver, elements, state, constraints);
    ASSERT_TRUE(converged);

    {
        auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q);
        for (const auto node_id : {beam1_tip_id, beam2_tip_id}) {
            EXPECT_NEAR(q_host(node_id, 0), 0., 1e-10);
            EXPECT_NEAR(q_host(node_id, 1), 0., 1e-10);
            EXPECT_NEAR(q_host(node_id, 2), 0., 1e-10);
            EXPECT_NEAR(q_host(node_id, 3), 1., 1e-10);
            EXPECT_NEAR(q_host(node_id, 4), 0., 1e-10);
            EXPECT_NEAR(q_host(node_id, 5), 0., 1e-10);
            EXPECT_NEAR(q_host(node_id, 6), 0., 1e-10);
        }
    }

    // Step 4: Apply a moment vector with all three components to the tip of the first beam.
    // External loads are placed in state.f so the static load-stepping can scale them.
    {
        auto host_f = Kokkos::create_mirror_view(state.f);
        Kokkos::deep_copy(host_f, state.f);
        host_f(beam1_tip_id, 3) = 5.e3;
        host_f(beam1_tip_id, 4) = 4.e3;
        host_f(beam1_tip_id, 5) = 3.e3;
        Kokkos::deep_copy(state.f, host_f);
    }

    converged = Step(parameters, solver, elements, state, constraints);
    ASSERT_TRUE(converged);
    {
        // Verify that there are real displacements and rotations relative to the start
        const auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q);
        ASSERT_GT(
            std::hypot(q_host(beam1_tip_id, 0), q_host(beam1_tip_id, 1), q_host(beam1_tip_id, 2)),
            1.e-4
        );
        ASSERT_GT(
            std::hypot(q_host(beam1_tip_id, 4), q_host(beam1_tip_id, 5), q_host(beam1_tip_id, 6)),
            1.e-4
        );
    }

    // Step 5: The deformed tip of the second beam should equal the deformed tip of the first
    // beam translated, plus the initial relative position vector rotated by only the
    // displacement quaternion of the first beam's tip node.
    const auto expected = Kokkos::View<double[3], DeviceType>("expected");
    const auto actual = Kokkos::View<double[3], DeviceType>("actual");
    const auto x0 = state.x0;
    const auto x = state.x;
    const auto q = state.q;
    Kokkos::parallel_for(
        "MapBeamTipPoint", Kokkos::RangePolicy<typename DeviceType::execution_space>(0, 1),
        KOKKOS_LAMBDA(int) {
            // Initial position vector from the first beam tip to the second beam tip
            auto r0_data = Kokkos::Array<double, 3>{
                x0(beam2_tip_id, 0) - x0(beam1_tip_id, 0),
                x0(beam2_tip_id, 1) - x0(beam1_tip_id, 1),
                x0(beam2_tip_id, 2) - x0(beam1_tip_id, 2),
            };
            const auto r0 = Kokkos::View<double[3], DeviceType>(r0_data.data());

            // Displacement quaternion of the first beam tip node
            const auto q_u = Kokkos::subview(q, beam1_tip_id, Kokkos::make_pair(3, 7));

            // Rotate the initial position vector by the tip displacement quaternion
            math::RotateVectorByQuaternion(q_u, r0, expected);

            // Add the deformed translation of the first beam tip
            for (auto i = 0; i < 3; ++i) {
                expected(i) += x(beam1_tip_id, i);
                actual(i) = x(beam2_tip_id, i);
            }
        }
    );

    auto expected_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected);
    auto actual_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
    for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(actual_host(i), expected_host(i), 1e-6);
    }
}

}  // namespace kynema_fmb::tests