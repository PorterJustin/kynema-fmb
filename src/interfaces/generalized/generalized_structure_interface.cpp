#include "generalized_structure_interface.hpp"
#include "model/model.hpp"
#include "step/step.hpp"

#include <yaml-cpp/yaml.h>

namespace kynema_fmb::interfaces {

using DeviceType =
    Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

GeneralStructureInterface::GeneralStructureInterface(
    const YAML::Node& structural_description):
    model(BuildModel(structural_description, model_description)),
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
    state_n(CloneState(state)),
    f_nm1(Kokkos::view_alloc("f_nm1", Kokkos::WithoutInitializing), state_n.f.extent(0)),
    host_state(state),
    host_constraints(constraints) {

    // Copy the forces from initialization
    Kokkos::deep_copy(this->f_nm1, this->state_n.f);
    this->host_state.CopyFromState(this->state);

    // TODO : step parameters may take inputs of atol and rtol.

}

//--------------------------------------------------------------------------
// Generalized API Functions
//--------------------------------------------------------------------------

YAML::Node GeneralStructureInterface::initialize_module(){
    return YAML::Clone(this->model_description);
}

void GeneralStructureInterface::predict_inputs(){
    using RangePolicy = Kokkos::RangePolicy<typename DeviceType::execution_space>;

    auto f     = this->state.f;
    auto f_n   = this->state_n.f;
    auto f_nm1 = this->f_nm1;

    Kokkos::parallel_for(
        "PredictForce", RangePolicy(0, f.extent(0)),
        KOKKOS_LAMBDA(size_t node) {
            for (auto component = 0U; component < 6U; ++component) {
                // TODO : This may need to be replaced with a higher order accuracy equation.
                f(node, component) = 2. * f_n(node, component) - f_nm1(node, component);
            }
        }
    );
}

template <typename ValueType>
using HostView = typename Kokkos::View<ValueType, DeviceType>::host_mirror_type;

void GeneralStructureInterface::receive_inputs(HostView<double* [6]> forces){
    Kokkos::deep_copy(this->state.f, forces);
}


void GeneralStructureInterface::advance_states(){
    // Back-up the forces at time n+1
    auto fnp1_temp = Kokkos::View<double* [6], DeviceType>(
    Kokkos::view_alloc("fnp1_temp", Kokkos::WithoutInitializing), this->state.f.extent(0));
    Kokkos::deep_copy(fnp1_temp, this->state.f);

    // Copy the state at time n -> state
    this->state = CloneState(this->state_n);

    // Copy forces back to state
    auto f     = this->state.f;
    Kokkos::deep_copy(f, fnp1_temp);

    auto converged = Step(this->parameters,
                          this->solver,
                          this->elements,
                          this->state,
                          this->constraints);

    // TODO : Throw an error if not converged?
}

HostState<DeviceType>& GeneralStructureInterface::calculate_outputs(){
    this->host_state.CopyFromState(this->state);
    return this->host_state;
}

void GeneralStructureInterface::finalize_time_step(){
    // Data from time n -> n-1
    // Data from time n+1 -> n
    auto f_nm1 = this->f_nm1;
    auto f_n   = this->state_n.f;
    Kokkos::deep_copy(f_nm1, f_n);

    this->state_n = CloneState(this->state);
}


//--------------------------------------------------------------------------
// Private Functions for Building Model
//--------------------------------------------------------------------------

Model GeneralStructureInterface::BuildModel(
    const YAML::Node& structural_description,
    YAML::Node& model_description
) {
    // High level global parameters
    auto gravity = structural_description["simulation"]["gravity"].as<std::array<double, 3>>();

    auto model = Model(gravity);

    // Checks for list of components for the model
    const auto& components = structural_description["components"];
    if (!components || !components.IsSequence()) {
        throw std::runtime_error("No structural components in input.");
    }

    // Append a new empty entry to the components sequence
    YAML::Node components_out = model_description["components"];
    
    // Add individual structural components to the model
    for (const auto& item : components)
    {
        const auto type = item["type"].as<std::string>();
        if (type == "turbine") {
            AddTurbine(model, item);
        }
        else if (type == "6dof")
        {
            // Add an entry to the output structure description
            components_out.push_back(YAML::Node());

            // Get the entry to add details to.
            YAML::Node output_description = components_out[components_out.size() - 1];

            output_description["type"] = "6dof";

            Add6DOF(model, item, output_description);
        }
        else {
            throw std::runtime_error("Unknown component type: " + type);
        }
    }

    model_description["model"]["num_nodes"] = model.NumNodes();

    return model;
}

void GeneralStructureInterface::AddTurbine(
    Model& model, const YAML::Node& turbine_description
) {
    /* Do the full turbine initialization here */
}


void GeneralStructureInterface::Add6DOF(Model& model,
            const YAML::Node& dof6_description,
            YAML::Node& output_description) {
    /* Add a 6DOF mass, this is currently just a rough sketch,
    these values should come from the yaml node */

    auto mass = dof6_description["mass"].as<double>();
    auto inertia = dof6_description["inertia"].as<std::array<double,9>>();    // inertia matrix
    auto x0 = dof6_description["x0"].as<std::array<double,3>>();          // translation position   
    auto x_dot0 = dof6_description["x_dot0"].as<std::array<double,3>>();  // translation velocity
    auto x_ddot0 = dof6_description["x_ddot0"].as<std::array<double,3>>();  // translation acceleration

    auto omega0 = dof6_description["omega0"].as<std::array<double,3>>();  // initial angular velocity  
    auto omega_dot0 = dof6_description["omega_dot0"].as<std::array<double,3>>();  // initial angular acceleration                                  // initial velocity

    auto mass_node_id =
        model.AddNode()
            .SetPosition(x0[0], x0[1], x0[2], 1., 0., 0., 0.)
            .SetVelocity(x_dot0[0], x_dot0[1], x_dot0[2], omega0[0], omega0[1], omega0[2])
            .SetAcceleration(
                x_ddot0[0], x_ddot0[1], x_ddot0[2], omega_dot0[0], omega_dot0[1], omega_dot0[2]
            )
            .Build();

    output_description["mass_node_id"] = mass_node_id;

    // Add masses element with m and J as mass matrix
    model.AddMassElement(
        mass_node_id, {{
                          {mass, 0., 0., 0., 0., 0.},        // mass in x-direction
                          {0., mass, 0., 0., 0., 0.},        // mass in y-direction
                          {0., 0., mass, 0., 0., 0.},        // mass in z-direction
                          {0., 0., 0., inertia[0], inertia[1], inertia[2]},  // inertia xx
                          {0., 0., 0., inertia[3], inertia[4], inertia[5]},  // inertia yy
                          {0., 0., 0., inertia[6], inertia[7], inertia[8]},  // inertia zz
                      }}
    );

    const auto& spring_set = dof6_description["springs"];

    if(spring_set){
        for (const auto& spring : spring_set)
        {
            auto fixed_coords = spring["fixed_point"].as<std::array<double,3>>();
            auto l0 = spring["l0"].as<double>();
            auto k = spring["k"].as<double>();

            auto fixed_node_id = model.AddNode()
                                    .SetPosition(fixed_coords[0], fixed_coords[1], fixed_coords[2],
                                                 1., 0., 0., 0.)
                                    .Build();

            model.AddFixedBC(fixed_node_id);
            model.AddSpringElement(fixed_node_id, mass_node_id, k, l0);
        }
    }
}

}