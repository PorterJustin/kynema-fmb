#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>

#include "interfaces/components/beam.hpp"
#include "interfaces/components/beam_input.hpp"
#include "math/gl_quadrature.hpp"
#include "math/gll_quadrature.hpp"
#include "model/model.hpp"
#include "step/step.hpp"
#include "test_utilities.hpp"

namespace kynema_fmb::tests {

// A uniform prismatic beam hangs vertically along the global z-axis from a
// fixed top node at z = 0 down to a free bottom end at z = -L. Under its own
// weight it behaves like a bar hanging under self-weight, stretching purely in
// the axial (z) direction. The closed form vertical displacement, measuring
// zeta upward from the free (bottom) end, is
//
//     u_z(zeta) = -w * (L^2 - zeta^2) / (2 * E*A)
//
// where w = (mass/length) * |g| is the weight per unit length, L is the beam
// length, and E*A is the sectional axial stiffness. This test performs a static
// solve under gravity and compares the nodal displacements to this solution.
TEST(StaticBeamTest, BeamUnderGravity) {
    // Mass matrix for a uniform composite beam section (diagonal).
    // Entry [0][0] is the translational mass per unit length.
    constexpr auto mass_matrix = std::array{
        std::array{8.538, 0., 0., 0., 0., 0.},   std::array{0., 8.538, 0., 0., 0., 0.},
        std::array{0., 0., 8.538, 0., 0., 0.},   std::array{0., 0., 0., 1.4433e-2, 0., 0.},
        std::array{0., 0., 0., 0., 0.40972e-2, 0.}, std::array{0., 0., 0., 0., 0., 1.0336e-2},
    };

    // Stiffness matrix for a uniform composite beam section (diagonal).
    // Entry [0][0] is the sectional axial stiffness E*A.
    constexpr auto stiffness_matrix = std::array{
        std::array{1368.17, 0., 0., 0., 0., 0.}, std::array{0., 88.56e3, 0., 0., 0., 0.},
        std::array{0., 0., 38.78, 0., 0., 0.},   std::array{0., 0., 0., 16.9600e3, 0., 0.},
        std::array{0., 0., 0., 0., 59.1200, 0.}, std::array{0., 0., 0., 0., 0., 141.470e3},
    };

    constexpr double beam_length = 10.;        // beam length along z (m)
    constexpr double gravity = -9.81;          // gravitational acceleration along z (m/s^2)
    const double mass_per_length = mass_matrix[0][0];
    const double axial_stiffness = stiffness_matrix[0][0];

    // Node locations (GLL quadrature) mapped from [-1, 1] to [0, 1]
    const auto num_nodes = 5UL;
    const auto gll_locations = math::GetGllLocations(num_nodes - 1);
    std::vector<double> node_s(gll_locations.size());
    std::ranges::transform(gll_locations, node_s.begin(), [](auto xi) {
        return 0.5 * (xi + 1.);
    });

    // Create model for managing nodes and constraints
    auto model = Model();

    // Set gravity in model (acting downward along the z-axis)
    model.SetGravity(0., 0., gravity);

    // Orientation that rotates the section's local x-axis (the axial direction)
    // onto the global -z-axis (the direction of increasing node parameter): a
    // +90 degree rotation about the y-axis.
    constexpr double sqrt2_over_2 = 0.70710678118654752;
    const auto orientation = std::array{sqrt2_over_2, 0., sqrt2_over_2, 0.};

    // Build vector of nodes along the z axis from z = 0 (top) to z = -L (bottom)
    std::vector<size_t> beam_node_ids;
    std::ranges::transform(node_s, std::back_inserter(beam_node_ids), [&](auto s) {
        return model.AddNode()
            .SetElemLocation(s)
            .SetPosition(
                0., 0., -beam_length * s, orientation[0], orientation[1], orientation[2],
                orientation[3]
            )
            .Build();
    });

    // Gauss-Legendre quadrature points and weights for the beam element
    const auto quad_order = 7UL;
    const auto gl_locations = math::GetGlLocations(quad_order);
    const auto gl_weights = math::GetGlWeights(quad_order);
    std::vector<std::array<double, 2>> quad_points(gl_locations.size());
    for (size_t i = 0; i < gl_locations.size(); ++i) {
        quad_points[i] = {gl_locations[i], gl_weights[i]};
    }

    // Add beam element with uniform sections
    model.AddBeamElement(
        beam_node_ids,
        std::array{
            BeamSection(0., mass_matrix, stiffness_matrix),
            BeamSection(1., mass_matrix, stiffness_matrix),
        },
        quad_points
    );

    // Fix the top node (located at z = 0)
    const auto top_node_id = beam_node_ids.front();
    model.AddFixedBC(top_node_id);

    // Static solution parameters
    const bool is_dynamic_solve(false);
    const size_t max_iter(10);
    const double step_size(1.);  // pseudo-time step, only sets the preconditioner for a static solve
    const double rho_inf(0.);
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Solve for the static equilibrium under gravity
    const auto converged = Step(parameters, solver, elements, state, constraints);
    ASSERT_TRUE(converged);

    // Compare nodal displacements against the analytical hanging-bar solution.
    // zeta is measured upward from the free (bottom) end at z = -L, so for a node
    // at z = -L * s it is zeta = z + L = L * (1 - s).
    const double weight_per_length = mass_per_length * std::abs(gravity);
    const auto q = kokkos_view_2D_to_vector(state.q);

    for (size_t i = 0; i < beam_node_ids.size(); ++i) {
        const double zeta = beam_length * node_s[i];
        const double u_z_expected =
            (weight_per_length / (2. * axial_stiffness)) * (zeta * zeta  - 2. * beam_length * zeta);

        const auto& node_q = q[beam_node_ids[i]];

        EXPECT_NEAR(node_q[0], 0., 1e-11);            // no transverse (x) displacement
        EXPECT_NEAR(node_q[1], 0., 1e-11);            // no transverse (y) displacement
        EXPECT_NEAR(node_q[2], u_z_expected, 1e-10);  // axial (z) displacement
    }
}

// Same problem as BeamUnderGravity, but the beam is constructed through the
// high-level Beam/BeamInput interface (as used in the mooring_static example)
// instead of Model::AddBeamElement. The physical setup, parameters, and
// analytical check are identical: a uniform beam hangs from a fixed top node at
// z = 0 down to a free end at z = -L and is loaded by gravity.
TEST(StaticBeamTest, BeamUnderGravityBeamInterface) {
    // Mass matrix for a uniform composite beam section (diagonal).
    // Entry [0][0] is the translational mass per unit length.
    constexpr auto mass_matrix = std::array{
        std::array{8.538, 0., 0., 0., 0., 0.},      std::array{0., 8.538, 0., 0., 0., 0.},
        std::array{0., 0., 8.538, 0., 0., 0.},      std::array{0., 0., 0., 1.4433e-2, 0., 0.},
        std::array{0., 0., 0., 0., 0.40972e-2, 0.}, std::array{0., 0., 0., 0., 0., 1.0336e-2},
    };

    // Stiffness matrix for a uniform composite beam section (diagonal).
    // Entry [0][0] is the sectional axial stiffness E*A.
    constexpr auto stiffness_matrix = std::array{
        std::array{1368.17, 0., 0., 0., 0., 0.}, std::array{0., 88.56e3, 0., 0., 0., 0.},
        std::array{0., 0., 38.78, 0., 0., 0.},   std::array{0., 0., 0., 16.9600e3, 0., 0.},
        std::array{0., 0., 0., 0., 59.1200, 0.}, std::array{0., 0., 0., 0., 0., 141.470e3},
    };

    constexpr double beam_length = 10.;  // beam length along z (m)
    constexpr double gravity = -9.81;    // gravitational acceleration along z (m/s^2)
    const double mass_per_length = mass_matrix[0][0];
    const double axial_stiffness = stiffness_matrix[0][0];

    // Node arc-length fractions (GLL) matching the interface's node layout
    const auto num_nodes = 5UL;
    const auto gll_locations = math::GetGllLocations(num_nodes - 1);
    std::vector<double> node_s(gll_locations.size());
    std::ranges::transform(gll_locations, node_s.begin(), [](auto xi) {
        return 0.5 * (xi + 1.);
    });

    // Create model for managing nodes and constraints
    auto model = Model();

    // Set gravity in model (acting downward along the z-axis)
    model.SetGravity(0., 0., gravity);

    // Describe the beam using the high-level BeamInput format and let the Beam
    // interface build the nodes, element, and quadrature within the model.
    namespace components = interfaces::components;
    auto beam_input = components::BeamInput{};

    // Spectral element order (number of nodes - 1)
    beam_input.element_order = num_nodes - 1;

    // Straight reference axis along the x axis, from (0,0,0) to (L,0,0)
    beam_input.ref_axis.coordinate_grid = std::vector{0., 1.};
    beam_input.ref_axis.coordinates =
        std::vector{std::array{0., 0., 0.}, std::array{beam_length, 0., 0.}};
    beam_input.ref_axis.twist_grid = std::vector{0., 1.};
    beam_input.ref_axis.twist = std::vector{0., 0.};

    // Uniform sectional properties along the beam
    beam_input.sections = std::vector{
        components::Section(0., mass_matrix, stiffness_matrix),
        components::Section(1., mass_matrix, stiffness_matrix),
    };

    // Gauss-Lobatto segmented quadrature, as used in the mooring_static example.
    beam_input.quadrature_rule = components::BeamInput::QuadratureRule::GaussLobatto;
    beam_input.quadrature_style = components::BeamInput::QuadratureStyle::Segmented;
    beam_input.section_refinement = beam_input.element_order - 1;

    // Reorient the beam so it hangs along the -z direction. The beam is built
    // along +x, so a +90 degree rotation about the y axis maps +x -> -z. This
    // root orientation is applied by Beam::PositionBladeInSpace.
    constexpr double sqrt2_over_2 = 0.70710678118654752;
    beam_input.root.position = {0., 0., 0., sqrt2_over_2, 0., sqrt2_over_2, 0.};

    // Build the beam within the model
    auto beam = components::Beam(beam_input, model);

    // Node IDs of the beam element (in element order, root -> tip)
    const auto& beam_node_ids = model.GetBeamElement(beam.beam_element_id).node_ids;

    // Fix the top (root) node located at z = 0
    model.AddFixedBC(beam_node_ids.front());

    // Static solution parameters (identical to BeamUnderGravity)
    const bool is_dynamic_solve(false);
    const size_t max_iter(10);
    const double step_size(1.);  // pseudo-time step, only sets the preconditioner for a static solve
    const double rho_inf(0.);
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Solve for the static equilibrium under gravity
    const auto converged = Step(parameters, solver, elements, state, constraints);
    ASSERT_TRUE(converged);

    // Compare nodal displacements against the analytical hanging-bar solution.
    // zeta is measured downward from the fixed top (z = 0), so for a node at
    // z = -L * s it is zeta = -z = L * s.
    const double weight_per_length = mass_per_length * std::abs(gravity);
    const auto q = kokkos_view_2D_to_vector(state.q);

    for (size_t i = 0; i < beam_node_ids.size(); ++i) {
        const double zeta = beam_length * node_s[i];
        const double u_z_expected =
            (weight_per_length / (2. * axial_stiffness)) * (zeta * zeta - 2. * beam_length * zeta);

        const auto& node_q = q[beam_node_ids[i]];

        EXPECT_NEAR(node_q[0], 0., 1e-11);            // no transverse (x) displacement
        EXPECT_NEAR(node_q[1], 0., 1e-11);            // no transverse (y) displacement
        EXPECT_NEAR(node_q[2], u_z_expected, 1e-10);  // axial (z) displacement
    }
}

// A horizontal cantilever beam, built with the high-level Beam/BeamInput
// interface, is fixed at its root (x = 0), extends along +x to a free tip at
// x = L, and is loaded by gravity acting transverse to its axis (-z). In the
// small-deflection limit this is the classic Euler-Bernoulli cantilever under a
// uniformly distributed load w = (mass/length) * |g|, whose transverse
// deflection measured from the fixed root is
//
//     v(x) = w / (24 * E*I) * (x^4 - 4*L*x^3 + 6*L^2*x^2)
//
// (downward, so u_z = -v), with tip deflection w*L^4 / (8*E*I). The section is
// made stiff in shear/axial/torsion so that the bending response dominates and
// the geometrically-exact beam reproduces the Euler-Bernoulli solution.
TEST(StaticBeamTest, CantileverUnderGravityBeamInterface) {
    // Mass matrix for a uniform beam section (diagonal).
    // Entry [0][0] is the translational mass per unit length.
    constexpr auto mass_matrix = std::array{
        std::array{8.538, 0., 0., 0., 0., 0.},      std::array{0., 8.538, 0., 0., 0., 0.},
        std::array{0., 0., 8.538, 0., 0., 0.},      std::array{0., 0., 0., 1.4433e-2, 0., 0.},
        std::array{0., 0., 0., 0., 0.40972e-2, 0.}, std::array{0., 0., 0., 0., 0., 1.0336e-2},
    };

    // Stiffness matrix for a uniform beam section (diagonal). Bending about the
    // y-axis (entry [4][4]) governs the x-z plane deflection. The axial, shear,
    // and torsion stiffnesses are made large so that shear/axial deformation is
    // negligible and the response is dominated by bending.
    constexpr double bending_stiffness = 1.0e8;   // E*I (about y, drives z deflection)
    constexpr double shear_stiffness = 1.0e12;    // G*A
    constexpr double axial_stiffness = 1.0e12;    // E*A
    constexpr double torsion_stiffness = 1.0e12;  // G*J

    constexpr auto stiffness_matrix = std::array{
        std::array{axial_stiffness, 0., 0., 0., 0., 0.},
        std::array{0., shear_stiffness, 0., 0., 0., 0.},
        std::array{0., 0., shear_stiffness, 0., 0., 0.},
        std::array{0., 0., 0., torsion_stiffness, 0., 0.},
        std::array{0., 0., 0., 0., bending_stiffness, 0.},
        std::array{0., 0., 0., 0., 0., bending_stiffness},
    };

    constexpr double beam_length = 10.;  // beam length along x (m)
    constexpr double gravity = -9.81;    // gravitational acceleration along z (m/s^2)
    const double mass_per_length = mass_matrix[0][0];

    // Node arc-length fractions (GLL) matching the interface's node layout
    const auto num_nodes = 5UL;
    const auto gll_locations = math::GetGllLocations(num_nodes - 1);
    std::vector<double> node_s(gll_locations.size());
    std::ranges::transform(gll_locations, node_s.begin(), [](auto xi) {
        return 0.5 * (xi + 1.);
    });

    // Create model for managing nodes and constraints
    auto model = Model();

    // Set gravity in model (acting downward along the z-axis, transverse to the beam)
    model.SetGravity(0., 0., gravity);

    // Describe the beam using the high-level BeamInput format and let the Beam
    // interface build the nodes, element, and quadrature within the model.
    namespace components = interfaces::components;
    auto beam_input = components::BeamInput{};

    // Spectral element order (number of nodes - 1)
    beam_input.element_order = num_nodes - 1;

    // Straight reference axis along the x axis, from (0,0,0) to (L,0,0)
    beam_input.ref_axis.coordinate_grid = std::vector{0., 1.};
    beam_input.ref_axis.coordinates =
        std::vector{std::array{0., 0., 0.}, std::array{beam_length, 0., 0.}};
    beam_input.ref_axis.twist_grid = std::vector{0., 1.};
    beam_input.ref_axis.twist = std::vector{0., 0.};

    // Uniform sectional properties along the beam
    beam_input.sections = std::vector{
        components::Section(0., mass_matrix, stiffness_matrix),
        components::Section(1., mass_matrix, stiffness_matrix),
    };

    // Gauss-Lobatto segmented quadrature, as used in the mooring_static example.
    beam_input.quadrature_rule = components::BeamInput::QuadratureRule::GaussLobatto;
    beam_input.quadrature_style = components::BeamInput::QuadratureStyle::Segmented;
    beam_input.section_refinement = beam_input.element_order - 1;

    // Leave the beam along +x (horizontal) with the default identity root
    // orientation, so gravity (-z) acts transverse to the beam axis.

    // Build the beam within the model
    auto beam = components::Beam(beam_input, model);

    // Node IDs of the beam element (in element order, root -> tip)
    const auto& beam_node_ids = model.GetBeamElement(beam.beam_element_id).node_ids;

    // Fix the root (cantilever) node located at x = 0
    model.AddFixedBC(beam_node_ids.front());

    // Static solution parameters
    const bool is_dynamic_solve(false);
    const size_t max_iter(10);
    const double step_size(1.);  // pseudo-time step, only sets the preconditioner for a static solve
    const double rho_inf(0.);
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Solve for the static equilibrium under gravity
    const auto converged = Step(parameters, solver, elements, state, constraints);
    ASSERT_TRUE(converged);

    // Compare nodal deflections against the analytical Euler-Bernoulli solution
    // for a cantilever under a uniformly distributed load. The small residual
    // difference (~1e-4 relative) comes from the beam's finite shear stiffness.
    const double weight_per_length = mass_per_length * std::abs(gravity);
    const auto q = kokkos_view_2D_to_vector(state.q);

    for (size_t i = 0; i < beam_node_ids.size(); ++i) {
        const double x = beam_length * node_s[i];
        const double v = (weight_per_length * x * x / (24. * bending_stiffness)) *
                         (x * x - 4. * beam_length * x + 6. * beam_length * beam_length);
        const double u_z_expected = -v;

        const auto& node_q = q[beam_node_ids[i]];

        EXPECT_NEAR(node_q[1], 0., 1e-10);  // no out-of-plane (y) displacement
        EXPECT_NEAR(
            node_q[2], u_z_expected, 2e-6 * std::abs(u_z_expected) + 1e-9
        );  // transverse (z) deflection
    }
}

}  // namespace kynema_fmb::tests