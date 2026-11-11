#include "generalized_structure_interface.hpp"
#include "model/model.hpp"
#include "step/step.hpp"

#include <yaml-cpp/yaml.h>

namespace kynema_fmb::interfaces {

GeneralStructureInterface::GeneralStructureInterface(
    const YAML::Node& structural_description):
    model(BuildModel(Model(), structural_description)),
    state(model.CreateState<DeviceType>()),
    elements(model.CreateElements<DeviceType>()),
    constraints(model.CreateConstraints<DeviceType>()),
    parameters(StepParameters(
        structural_description["simulation"]["is_dynamic_solve"].as<bool>(),
        structural_description["simulation"]["max_nonlinear_iterations"].as<size_t>(),
        structural_description["simulation"]["step_size"].as<double>(),
        structural_description["simulation"]["rho_inf"].as<double>())
    ),
    solver(CreateSolver(state, elements, constraints)),
    state_save(CloneState(state)),
    host_state(state),
    host_constraints(constraints) {

    /*
    Add everything to the model here
    */

    this->host_state.CopyFromState(this->state);

    

}

Model GeneralStructureInterface::BuildModel(
    Model model, const YAML::Node& structural_description
) {
    // Loop over structural_description["components"]
    // if item["type"] == "turbine", call AddTurbine

    const auto& components = structural_description["components"];

    if (!components || !components.IsSequence()) {
        throw std::runtime_error("No structural components in input.");
    }

    for (const auto& item : components)
    {
        const auto type = item["type"].as<std::string>();
        if (type == "turbine") {
            AddTurbine(model, item);
        }
        else if (type == "6dof")
        {
            Add6DOF(model, item);
        }
        else {
            throw std::runtime_error("Unknown component type: " + type);
        }
    }

    return model;
}

void GeneralStructureInterface::AddTurbine(
    Model& model, const YAML::Node& turbine_description
) {
    /* Do the full turbine initialization here */
}


void GeneralStructureInterface::Add6DOF(
    Model& model, const YAML::Node& dof6_description
) {
    /* Add a 6DOF mass, this is currently just a rough sketch,
    these values should come from the yaml node */

    constexpr auto mass = 15.;                                           // mass
    constexpr auto inertia = std::array{0.234375, 0.46875, 0.234375};    // inertia matrix
    const auto x = Eigen::Matrix<double, 3, 1>(0., 1., 0.);              // initial position
    const auto omega = Eigen::Matrix<double, 3, 1>(0., 150., -4.61538);  // initial angular velocity
    const auto x_dot = omega.cross(x);                                   // initial velocity

    auto mass_node_id =
        model.AddNode()
            .SetPosition(x(0), x(1), x(2), 1., 0., 0., 0.)
            .SetVelocity(x_dot(0), x_dot(1), x_dot(2), omega(0), omega(1), omega(2))
            .Build();
            // .SetAcceleration(
            //     x_ddot[0], x_ddot[1], x_ddot[2], omega_dot[0], omega_dot[1], omega_dot[2]
            // )
            // .Build();

    // Add masses element with m and J as mass matrix
    model.AddMassElement(
        mass_node_id, {{
                          {mass, 0., 0., 0., 0., 0.},        // mass in x-direction
                          {0., mass, 0., 0., 0., 0.},        // mass in y-direction
                          {0., 0., mass, 0., 0., 0.},        // mass in z-direction
                          {0., 0., 0., inertia[0], 0., 0.},  // inertia xx
                          {0., 0., 0., 0., inertia[1], 0.},  // inertia yy
                          {0., 0., 0., 0., 0., inertia[2]},  // inertia zz
                      }}
    );

}

}