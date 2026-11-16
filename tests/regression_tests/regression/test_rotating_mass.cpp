#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numbers>

#include <gtest/gtest.h>

#include "model/model.hpp"
#include "step/step.hpp"
#include "test_utilities.hpp"

namespace kynema_fmb::tests {

TEST(RotatingMass, InitializeRevoluteJoint) {
    // Create model object
    auto model = Model();

    // Set gravity
    model.SetGravity(0., 0., 0.);

    // Add node
    const auto mass_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Add mass element
    const auto m = 1.;
    const auto j = 2.;
    model.AddMassElement(
        mass_node_id, {{
                     {m, 0., 0., 0., 0., 0.},
                     {0., m, 0., 0., 0., 0.},
                     {0., 0., m, 0., 0., 0.},
                     {0., 0., 0., j, 0., 0.},
                     {0., 0., 0., 0., j, 0.},
                     {0., 0., 0., 0., 0., j},
                 }}
    );

    // Fixed Node
    const auto fixed_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Fix the fixed_node_id
    model.AddFixedBC(fixed_node_id);

    // Revolute joint between the nodes along the y-axis
    auto torque = 6.;
    auto axis = std::array{0., 1., 0.};

    model.AddRevoluteJointConstraint(
        std::array{fixed_node_id, mass_node_id},
        axis,
        &torque);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Create solution parameters
    const auto time_step = 0.001;
    const auto max_iter = 5;
    const auto rho_inf = 0.;
    auto parameters = StepParameters(true, max_iter, time_step, rho_inf);

    const auto force_x = 5.;
    auto host_f = Kokkos::create_mirror_view(state.f);
    host_f(mass_node_id, 0) = force_x;
    Kokkos::deep_copy(state.f, host_f);

    // Run simulation for 1000 steps, corresponds to 1.0 s
    const auto n_steps = 1000;
    for ([[maybe_unused]] auto iteration : std::views::iota(0, n_steps)) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
    }

    // Get velocity
    auto v_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.v);

    EXPECT_NEAR(v_host(0, 0), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 3), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 4), 2.9985, 1.e-12);
    EXPECT_NEAR(v_host(0, 5), 0., 1.e-12);
}


TEST(RotatingMass, UpdateRevoluteJoint) {
    // Create model object
    auto model = Model();

    // Set gravity
    model.SetGravity(0., 0., 0.);

    // Add node
    const auto mass_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Add mass element
    const auto m = 1.;
    const auto j = 4.;
    model.AddMassElement(
        mass_node_id, {{
                     {m, 0., 0., 0., 0., 0.},
                     {0., m, 0., 0., 0., 0.},
                     {0., 0., m, 0., 0., 0.},
                     {0., 0., 0., j, 0., 0.},
                     {0., 0., 0., 0., j, 0.},
                     {0., 0., 0., 0., 0., j},
                 }}
    );

    // Fixed Node
    const auto fixed_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Fix the fixed_node_id
    model.AddFixedBC(fixed_node_id);

    // Revolute joint between the nodes along the y-axis
    auto torque = 6.;
    auto axis = std::array{0., 1., 0.};

    model.AddRevoluteJointConstraint(
        std::array{fixed_node_id, mass_node_id},
        axis,
        &torque);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Create solution parameters
    const auto time_step = 0.001;
    const auto max_iter = 5;
    const auto rho_inf = 0.;
    auto parameters = StepParameters(true, max_iter, time_step, rho_inf);

    const auto force_x = 5.;
    auto host_f = Kokkos::create_mirror_view(state.f);
    host_f(mass_node_id, 0) = force_x;
    Kokkos::deep_copy(state.f, host_f);

    // Update RevoluteJoint Torque
    torque = 12.0;

    // Run simulation for 1000 steps, corresponds to 1.0 s
    const auto n_steps = 1000;
    for ([[maybe_unused]] auto iteration : std::views::iota(0, n_steps)) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
    }

    // Get velocity
    auto v_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.v);

    EXPECT_NEAR(v_host(0, 0), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 3), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 4), 2.9985, 1.e-12);
    EXPECT_NEAR(v_host(0, 5), 0., 1.e-12);
}


TEST(RotatingMass, FlippedRevoluteJoint) {
    // Create model object
    auto model = Model();

    // Set gravity
    model.SetGravity(0., 0., 0.);

    // Add node
    const auto mass_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Add mass element
    const auto m = 1.;
    const auto j = 2.;
    model.AddMassElement(
        mass_node_id, {{
                     {m, 0., 0., 0., 0., 0.},
                     {0., m, 0., 0., 0., 0.},
                     {0., 0., m, 0., 0., 0.},
                     {0., 0., 0., j, 0., 0.},
                     {0., 0., 0., 0., j, 0.},
                     {0., 0., 0., 0., 0., j},
                 }}
    );

    // Fixed Node
    const auto fixed_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Fix the fixed_node_id
    model.AddFixedBC(fixed_node_id);

    // Revolute joint between the nodes along the y-axis
    auto torque = 6.;
    auto axis = std::array{0., 1., 0.};

    model.AddRevoluteJointConstraint(
        std::array{mass_node_id, fixed_node_id},
        axis,
        &torque);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Create solution parameters
    const auto time_step = 0.001;
    const auto max_iter = 5;
    const auto rho_inf = 0.;
    auto parameters = StepParameters(true, max_iter, time_step, rho_inf);

    const auto force_x = 5.;
    auto host_f = Kokkos::create_mirror_view(state.f);
    host_f(mass_node_id, 0) = force_x;
    Kokkos::deep_copy(state.f, host_f);

    // Run simulation for 1000 steps, corresponds to 1.0 s
    const auto n_steps = 1000;
    for ([[maybe_unused]] auto iteration : std::views::iota(0, n_steps)) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
    }

    // Get velocity
    auto v_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.v);

    EXPECT_NEAR(v_host(0, 0), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 3), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 4), -2.9985, 1.e-12);
    EXPECT_NEAR(v_host(0, 5), 0., 1.e-12);
}


TEST(RotatingMass, InitializeRotationControl) {
    // Create model object
    auto model = Model();

    // Set gravity
    model.SetGravity(0., 0., 0.);

    // Add node
    const auto mass_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Add mass element
    const auto m = 1.;
    const auto j = 2.;
    model.AddMassElement(
        mass_node_id, {{
                     {m, 0., 0., 0., 0., 0.},
                     {0., m, 0., 0., 0., 0.},
                     {0., 0., m, 0., 0., 0.},
                     {0., 0., 0., j, 0., 0.},
                     {0., 0., 0., 0., j, 0.},
                     {0., 0., 0., 0., 0., j},
                 }}
    );

    // Fixed Node
    const auto fixed_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Fix the fixed_node_id
    model.AddFixedBC(fixed_node_id);


    // RotationControl joint between the nodes along the y-axis
    auto angle = std::numbers::pi / 3.;
    auto axis = std::array{0., 1., 0.};

    model.AddRotationControl(
        std::array{fixed_node_id, mass_node_id},
        axis,
        &angle);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Create solution parameters
    const auto time_step = 0.001;
    const auto max_iter = 5;
    const auto rho_inf = 0.;
    const auto a_tol = 1.e-7;
    const auto r_tol = 1.e-7;
    auto parameters = StepParameters(false, max_iter, time_step, rho_inf, a_tol, r_tol);

    const auto force_x = 5.;
    const auto torque_y = 6.;
    auto host_f = Kokkos::create_mirror_view(state.f);
    host_f(mass_node_id, 0) = force_x;
    host_f(mass_node_id, 4) = torque_y;
    Kokkos::deep_copy(state.f, host_f);

    // Solve one dynamic step to find the fully constrained position
    auto converged = Step(parameters, solver, elements, state, constraints);
    EXPECT_TRUE(converged);

    // Get position
    auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x);

    // Expected angle
    const auto q_expect = Eigen::Quaternion<double>(
        Eigen::AngleAxis<double>(angle, Eigen::Matrix<double, 3, 1>(axis[0], axis[1], axis[2]))
    );

    EXPECT_NEAR(x_host(0, 0), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 3), q_expect.w(), 1.e-12);
    EXPECT_NEAR(x_host(0, 4), q_expect.x(), 1.e-12);
    EXPECT_NEAR(x_host(0, 5), q_expect.y(), 1.e-12);
    EXPECT_NEAR(x_host(0, 6), q_expect.z(), 1.e-12);
}


TEST(RotatingMass, UpdateRotationControl) {
    // Create model object
    auto model = Model();

    // Set gravity
    model.SetGravity(0., 0., 0.);

    // Add node
    const auto mass_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Add mass element
    const auto m = 1.;
    const auto j = 2.;
    model.AddMassElement(
        mass_node_id, {{
                     {m, 0., 0., 0., 0., 0.},
                     {0., m, 0., 0., 0., 0.},
                     {0., 0., m, 0., 0., 0.},
                     {0., 0., 0., j, 0., 0.},
                     {0., 0., 0., 0., j, 0.},
                     {0., 0., 0., 0., 0., j},
                 }}
    );

    // Fixed Node
    const auto fixed_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Fix the fixed_node_id
    model.AddFixedBC(fixed_node_id);


    // RotationControl joint between the nodes along the y-axis
    auto angle = std::numbers::pi / 2.;
    auto axis = std::array{0., 1., 0.};

    model.AddRotationControl(
        std::array{fixed_node_id, mass_node_id},
        axis,
        &angle);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Create solution parameters
    const auto time_step = 0.001;
    const auto max_iter = 5;
    const auto rho_inf = 0.;
    const auto a_tol = 1.e-7;
    const auto r_tol = 1.e-7;
    auto parameters = StepParameters(false, max_iter, time_step, rho_inf, a_tol, r_tol);

    const auto force_x = 5.;
    const auto torque_y = 6.;
    auto host_f = Kokkos::create_mirror_view(state.f);
    host_f(mass_node_id, 0) = force_x;
    host_f(mass_node_id, 4) = torque_y;
    Kokkos::deep_copy(state.f, host_f);

    // Update angle
    angle = std::numbers::pi / 3.;

    // Solve one dynamic step to find the fully constrained position
    auto converged = Step(parameters, solver, elements, state, constraints);
    EXPECT_TRUE(converged);

    // Get position
    auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x);

    // Expected angle
    const auto q_expect = Eigen::Quaternion<double>(
        Eigen::AngleAxis<double>(std::numbers::pi / 3.,
        Eigen::Matrix<double, 3, 1>(axis[0], axis[1], axis[2]))
    );

    EXPECT_NEAR(x_host(0, 0), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 3), q_expect.w(), 1.e-12);
    EXPECT_NEAR(x_host(0, 4), q_expect.x(), 1.e-12);
    EXPECT_NEAR(x_host(0, 5), q_expect.y(), 1.e-12);
    EXPECT_NEAR(x_host(0, 6), q_expect.z(), 1.e-12);
}


TEST(RotatingMass, FlippedRotationControl) {
    // Create model object
    auto model = Model();

    // Set gravity
    model.SetGravity(0., 0., 0.);

    // Add node
    const auto mass_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Add mass element
    const auto m = 1.;
    const auto j = 2.;
    model.AddMassElement(
        mass_node_id, {{
                     {m, 0., 0., 0., 0., 0.},
                     {0., m, 0., 0., 0., 0.},
                     {0., 0., m, 0., 0., 0.},
                     {0., 0., 0., j, 0., 0.},
                     {0., 0., 0., 0., j, 0.},
                     {0., 0., 0., 0., 0., j},
                 }}
    );

    // Fixed Node
    const auto fixed_node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Fix the fixed_node_id
    model.AddFixedBC(fixed_node_id);


    // RotationControl joint between the nodes along the y-axis
    auto angle = std::numbers::pi / 3.;
    auto axis = std::array{0., 1., 0.};

    model.AddRotationControl(
        std::array{mass_node_id, fixed_node_id},
        axis,
        &angle);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver<>(state, elements, constraints);

    // Create solution parameters
    const auto time_step = 0.001;
    const auto max_iter = 5;
    const auto rho_inf = 0.;
    const auto a_tol = 1.e-7;
    const auto r_tol = 1.e-7;
    auto parameters = StepParameters(false, max_iter, time_step, rho_inf, a_tol, r_tol);

    const auto force_x = 5.;
    const auto torque_y = 6.;
    auto host_f = Kokkos::create_mirror_view(state.f);
    host_f(mass_node_id, 0) = force_x;
    host_f(mass_node_id, 4) = torque_y;
    Kokkos::deep_copy(state.f, host_f);

    // Solve one dynamic step to find the fully constrained position
    auto converged = Step(parameters, solver, elements, state, constraints);
    EXPECT_TRUE(converged);

    // Get position
    auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x);

    // Expected angle
    const auto q_expect = Eigen::Quaternion<double>(
        Eigen::AngleAxis<double>(-1.*angle, Eigen::Matrix<double, 3, 1>(axis[0], axis[1], axis[2]))
    );

    EXPECT_NEAR(x_host(0, 0), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(x_host(0, 3), q_expect.w(), 1.e-12);
    EXPECT_NEAR(x_host(0, 4), q_expect.x(), 1.e-12);
    EXPECT_NEAR(x_host(0, 5), q_expect.y(), 1.e-12);
    EXPECT_NEAR(x_host(0, 6), q_expect.z(), 1.e-12);
}

}