#include "generalized_structure_interface.hpp"
#include "model/model.hpp"
#include "step/step.hpp"

// Turbine includes:
#include "elements/beams/hollow_circle_properties.hpp"
#include "interfaces/components/turbine_builder.hpp"

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

    // Update atol and rtol if provided.
    if (structural_description["simulation"]["absolute_convergence_tol"]){
        this->parameters.absolute_convergence_tol
            = structural_description["simulation"]["absolute_convergence_tol"].as<double>();
    }
    if (structural_description["simulation"]["relative_convergence_tol"]){
        this->parameters.relative_convergence_tol
            = structural_description["simulation"]["relative_convergence_tol"].as<double>();
    }



    // TODO : will need to allocate memory for the different constraints.
    // then will need to update where the constraints in the model look to that memory.

    // Maybe after model construction, parse through the model to figure out how many
    // of each type of constraint there are. Then allocate std::array for the data.
    

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

void GeneralStructureInterface::reset_states(){
    // Back-up the forces at time n+1
    auto fnp1_temp = Kokkos::View<double* [6], DeviceType>(
    Kokkos::view_alloc("fnp1_temp", Kokkos::WithoutInitializing), this->state.f.extent(0));
    Kokkos::deep_copy(fnp1_temp, this->state.f);

    // Copy the state at time n -> state
    this->state = CloneState(this->state_n);

    // Copy forces back to state
    auto f     = this->state.f;
    Kokkos::deep_copy(f, fnp1_temp);
}

void GeneralStructureInterface::advance_states(){

    auto converged = Step(this->parameters,
                          this->solver,
                          this->elements,
                          this->state,
                          this->constraints);

    // TODO : Throw an error if not converged?
}

void GeneralStructureInterface::calculate_outputs(){
    this->host_state.CopyFromState(this->state);
}

HostState<DeviceType>& GeneralStructureInterface::get_outputs(){
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
        // Add an entry to the output structure description
        components_out.push_back(YAML::Node());

        // Get the entry to add details to.
        YAML::Node output_description = components_out[components_out.size() - 1];

        const auto type = item["type"].as<std::string>();

        if (type == "turbine") {
            output_description["type"] = "turbine";
            AddTurbine(model, item, output_description);
        }
        else if (type == "6dof")
        {
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
    Model& model,
    const YAML::Node& turbine_description,
    YAML::Node& output_description) {
    /* Do the full turbine initialization here */

    // Parameters from top level fmb input
    auto angular_speed = turbine_description["turbine"]["angular_speed"].as<double>();
    auto n_blades = turbine_description["blades"]["n_blades"].as<size_t>();
    auto elem_order_blades = turbine_description["blades"]["element_order"].as<size_t>();
    auto elem_order_tower = turbine_description["tower"]["element_order"].as<size_t>();
    auto write_outputs = turbine_description["outputs"]["write_output"].as<bool>();

    // Read WindIO yaml file
    const YAML::Node wio = YAML::LoadFile(turbine_description["windio_file"].as<std::string>());
    const auto& wio_tower = wio["components"]["tower"];
    const auto& wio_nacelle = wio["components"]["nacelle"];
    const auto& wio_blade = wio["components"]["blade"];
    const auto& wio_hub = wio["components"]["hub"];

    components::TurbineBuilder turbine_builder;

    //--------------------------------------------------------------------------
    // Build Turbine
    //--------------------------------------------------------------------------

    turbine_builder.SetAzimuthAngle(0.)
        .SetRotorApexToHub(0.)
        .SetHubDiameter(wio_hub["diameter"].as<double>())
        .SetConeAngle(wio_hub["cone_angle"].as<double>())
        //.SetBladePitchAngle(std::numbers:;pi / 2.)
        .SetShaftTiltAngle(wio_nacelle["drivetrain"]["uptilt"].as<double>())
        //.SetNacelleYawAngle(std::numbers:;pi)
        .SetTowerAxisToRotorApex(wio_nacelle["drivetrain"]["overhang"].as<double>())
        .SetTowerTopToRotorApex(wio_nacelle["drivetrain"]["distance_tt_hub"].as<double>());

    //--------------------------------------------------------------------------
    // Build Blades
    //--------------------------------------------------------------------------

    // Loop through blades and set parameters
    for (auto j : std::views::iota(0U, n_blades)) {
        // Get the blade builder
        auto& blade_builder = turbine_builder.Blade(j);

        // Set blade parameters
        blade_builder.SetElementOrder(elem_order_blades).PrescribedRootMotion(false);

        // Add reference axis coordinates (WindIO uses Z-axis as reference axis)
        const auto ref_axis = wio_blade["outer_shape_bem"]["reference_axis"];
        const auto axis_grid = ref_axis["x"]["grid"].as<std::vector<double>>();
        const auto x_values = ref_axis["x"]["values"].as<std::vector<double>>();
        const auto y_values = ref_axis["y"]["values"].as<std::vector<double>>();
        const auto z_values = ref_axis["z"]["values"].as<std::vector<double>>();
        for (auto i : std::views::iota(0U, axis_grid.size())) {
            blade_builder.AddRefAxisPoint(
                axis_grid[i], {x_values[i], y_values[i], z_values[i]},
                interfaces::components::ReferenceAxisOrientation::Z
            );
        }

        // Add reference axis twist
        const auto twist = wio_blade["outer_shape_bem"]["twist"];
        const auto twist_grid = twist["grid"].as<std::vector<double>>();
        const auto twist_values = twist["values"].as<std::vector<double>>();
        for (auto i : std::views::iota(0U, twist_grid.size())) {
            blade_builder.AddRefAxisTwist(twist_grid[i], twist_values[i] * std::numbers::pi / 180.);
        }

        // Add blade section properties
        const auto stiff_matrix = wio_blade["elastic_properties_mb"]["six_x_six"]["stiff_matrix"];
        const auto inertia_matrix =
            wio_blade["elastic_properties_mb"]["six_x_six"]["inertia_matrix"];
        const auto k_grid = stiff_matrix["grid"].as<std::vector<double>>();
        const auto m_grid = inertia_matrix["grid"].as<std::vector<double>>();
        const auto n_sections = k_grid.size();
        if (m_grid.size() != k_grid.size()) {
            throw std::runtime_error("stiffness and mass matrices not on same grid");
        }
        for (auto i : std::views::iota(0U, n_sections)) {
            if (abs(m_grid[i] - k_grid[i]) > 1e-8) {
                throw std::runtime_error("stiffness and mass matrices not on same grid");
            }
            const auto m = inertia_matrix["values"][i].as<std::vector<double>>();
            const auto k = stiff_matrix["values"][i].as<std::vector<double>>();
            blade_builder.AddSection(
                m_grid[i],
                {{
                    {m[0], m[1], m[2], m[3], m[4], m[5]},
                    {m[1], m[6], m[7], m[8], m[9], m[10]},
                    {m[2], m[7], m[11], m[12], m[13], m[14]},
                    {m[3], m[8], m[12], m[15], m[16], m[17]},
                    {m[4], m[9], m[13], m[16], m[18], m[19]},
                    {m[5], m[10], m[14], m[17], m[19], m[20]},
                }},
                {{
                    {k[0], k[1], k[2], k[3], k[4], k[5]},
                    {k[1], k[6], k[7], k[8], k[9], k[10]},
                    {k[2], k[7], k[11], k[12], k[13], k[14]},
                    {k[3], k[8], k[12], k[15], k[16], k[17]},
                    {k[4], k[9], k[13], k[16], k[18], k[19]},
                    {k[5], k[10], k[14], k[17], k[19], k[20]},
                }},
                interfaces::components::ReferenceAxisOrientation::Z
            );
        }
    }


    //--------------------------------------------------------------------------
    // Build Tower
    //--------------------------------------------------------------------------

    // Get the tower builder
    auto& tower_builder = turbine_builder.Tower();

    // Set tower parameters
    tower_builder
        .SetElementOrder(elem_order_tower)  // Set element order to num nodes -1
        .PrescribedRootMotion(false);        // Fix displacement of tower base node

    // Add reference axis coordinates (WindIO uses Z-axis as reference axis)
    const auto t_ref_axis = wio_tower["outer_shape_bem"]["reference_axis"];
    const auto axis_grid = t_ref_axis["x"]["grid"].as<std::vector<double>>();
    const auto x_values = t_ref_axis["x"]["values"].as<std::vector<double>>();
    const auto y_values = t_ref_axis["y"]["values"].as<std::vector<double>>();
    const auto z_values = t_ref_axis["z"]["values"].as<std::vector<double>>();
    for (auto i : std::views::iota(0U, axis_grid.size())) {
        tower_builder.AddRefAxisPoint(
            axis_grid[i], {x_values[i], y_values[i], z_values[i]},
            interfaces::components::ReferenceAxisOrientation::Z
        );
    }

    // Set tower base position from first reference axis point
    const auto tower_base_position =
        std::array<double, 7>{x_values[0], y_values[0], z_values[0], 1., 0., 0., 0.};
    turbine_builder.SetTowerBasePosition(tower_base_position);

    // Add reference axis twist (zero for tower)
    tower_builder.AddRefAxisTwist(0.0, 0.0).AddRefAxisTwist(1.0, 0.0);

    // Find the tower material properties
    const auto t_layer = wio_tower["internal_structure_2d_fem"]["layers"][0];
    const auto t_material_name = t_layer["material"].as<std::string>();
    YAML::Node t_material;
    for (const auto& m : wio["materials"]) {
        if (m["name"] && m["name"].as<std::string>() == t_material_name) {
            t_material = m.as<YAML::Node>();
            break;
        }
    }
    if (!t_material) {
        throw std::runtime_error(
            "Material '" + t_material_name + "' not found in materials section"
        );
    }

    // Add tower section properties
    const auto t_diameter = wio_tower["outer_shape_bem"]["outer_diameter"];
    const auto t_diameter_grid = t_diameter["grid"].as<std::vector<double>>();
    const auto t_diameter_values = t_diameter["values"].as<std::vector<double>>();
    const auto t_wall_thickness = t_layer["thickness"]["values"].as<std::vector<double>>();
    for (auto i : std::views::iota(0U, t_diameter_grid.size())) {
        // Create section mass and stiffness matrices
        const auto section = beams::GenerateHollowCircleSection(
            t_diameter_grid[i], t_material["E"].as<double>(), t_material["G"].as<double>(),
            t_material["rho"].as<double>(), t_diameter_values[i], t_wall_thickness[i],
            t_material["nu"].as<double>()
        );

        // Add section
        tower_builder.AddSection(
            t_diameter_grid[i], section.M_star, section.C_star,
            interfaces::components::ReferenceAxisOrientation::Z
        );
    }

    //--------------------------------------------------------------------------
    // Add mass elements
    //--------------------------------------------------------------------------

    // Get nacelle mass properties from WindIO
    const auto& nacelle_props = wio_nacelle["elastic_properties_mb"];
    const auto system_mass = nacelle_props["system_mass"].as<double>();
    const auto yaw_mass = nacelle_props["yaw_mass"].as<double>();
    const auto system_inertia_tt = nacelle_props["system_inertia_tt"].as<std::vector<double>>();

    // Construct 6x6 inertia matrix for yaw bearing node
    const auto total_mass = system_mass + yaw_mass;
    const auto nacelle_inertia_matrix = std::array<std::array<double, 6>, 6>{
        {{total_mass, 0., 0., 0., 0., 0.},
         {0., total_mass, 0., 0., 0., 0.},
         {0., 0., total_mass, 0., 0., 0.},
         {0., 0., 0., system_inertia_tt[0], system_inertia_tt[3], system_inertia_tt[4]},
         {0., 0., 0., system_inertia_tt[3], system_inertia_tt[1], system_inertia_tt[5]},
         {0., 0., 0., system_inertia_tt[4], system_inertia_tt[5], system_inertia_tt[2]}}
    };

    // Set the nacelle inertia matrix in the turbine builder
    turbine_builder.SetYawBearingInertiaMatrix(nacelle_inertia_matrix);

    // Get hub mass properties from WindIO
    const auto& hub_props = wio_hub["elastic_properties_mb"];
    const auto hub_mass = hub_props["system_mass"].as<double>();
    const auto hub_inertia = hub_props["system_inertia"].as<std::vector<double>>();

    // Construct 6x6 inertia matrix for hub node
    const auto hub_inertia_matrix = std::array<std::array<double, 6>, 6>{
        {{hub_mass, 0., 0., 0., 0., 0.},
         {0., hub_mass, 0., 0., 0., 0.},
         {0., 0., hub_mass, 0., 0., 0.},
         {0., 0., 0., hub_inertia[0], hub_inertia[3], hub_inertia[4]},
         {0., 0., 0., hub_inertia[3], hub_inertia[1], hub_inertia[5]},
         {0., 0., 0., hub_inertia[4], hub_inertia[5], hub_inertia[2]}}
    };

    // Set the hub inertia matrix in the turbine builder
    turbine_builder.SetHubInertiaMatrix(hub_inertia_matrix);

    components::Turbine turbine = turbine_builder.Build(model);

    // turbine now has names for all the different
    // components with associated node/beam/constraint id's
    //
    // However, it is going to go out of scope since the general interface
    // only cares about points and beams.
    //
    // Therefore, all of this information is going to be put into the output
    // yaml that can be shared with other modules and reused later.

    // Store Beam IDs
    YAML::Node blade_beam_ids(YAML::NodeType::Sequence);
    for (const auto& blade : turbine.blades) {
        blade_beam_ids.push_back(blade.beam_element_id);
    }
    output_description["blade_beam_ids"] = blade_beam_ids;
    output_description["tower_beam_id"] = turbine.tower.beam_element_id;

    // TODO : Likely need to store additional node ids for specific components here.
    //        However, I'm just going to wait and add those as needed.


    // Store Constraint IDs
    YAML::Node pitch_constraint_ids(YAML::NodeType::Sequence);
    for (const auto& constraint : turbine.blade_pitch) {
        pitch_constraint_ids.push_back(constraint.id);
    }
    output_description["pitch_constraint_ids"] = pitch_constraint_ids;
    output_description["rotor_constraint_id"] = turbine.shaft_base_to_azimuth.id;
    output_description["yaw_constraint_id"] = turbine.yaw_bearing_to_shaft_base.id;

}


void GeneralStructureInterface::Add6DOF(Model& model,
            const YAML::Node& dof6_description,
            YAML::Node& output_description) {

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