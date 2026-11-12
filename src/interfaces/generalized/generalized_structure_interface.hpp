#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/host_constraints.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/outputs.hpp"
#include "model/model.hpp"
#include "step/step_parameters.hpp"
#include "interfaces/turbine/turbine_interface.hpp"


namespace kynema_fmb::interfaces {

/**
 * @brief Interface for structural components of FMB for a generalized multi-physics framework.
 *
 * This class represents the generalized interface for structural simulations
 * used coupled to the Kynema suite of codes.
 */
class GeneralStructureInterface {
public:
    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

    template <typename ValueType>
    using HostView = typename Kokkos::View<ValueType, DeviceType>::host_mirror_type;

    /**
     * @brief Constructs a GeneralStructureInterface from a yaml description
     * @param structural_description Configuration parameters for the structure
     */
    explicit GeneralStructureInterface(const YAML::Node& structural_description);

    GeneralStructureInterface(GeneralStructureInterface& other) = delete;

    GeneralStructureInterface(GeneralStructureInterface&& other) = delete;

    GeneralStructureInterface& operator=(const GeneralStructureInterface&) = delete;

    GeneralStructureInterface& operator=(GeneralStructureInterface&&) = delete;

    ~GeneralStructureInterface() = default;

    /**
     * @brief Do any additional initialization and return a yaml of node ids etc.
     */
    YAML::Node initialize_module();

    /**
     * @brief Predict the inputs to the model at time n+1 from stored data.
     */
    void predict_inputs();

    /**
     * @brief Solve for the response at time n+1
     */
    void advance_states();

    /**
     * @brief Get actual values for inputs at time n+1
     *
     * TODO: this should receive a JSON
     * TODO: This may also need to get prescribed displacements
     */
    void receive_inputs(HostView<double* [6]> forces);

    /**
     * @brief Returns the entire state structures of outputs
     * TODO: should be updated to JSON in future.
     */
    HostState<DeviceType>& calculate_outputs();

    /**
     * @brief Finalize the stored data for the time step and shift time step index.
     */
    void finalize_time_step();


    /*
    1. `initialize_module`
    2. `initialize_stage1` - have multiple extra stages based on information sharing at init.
    3. `predict_inputs`
    4. `advance_states`
    5. `calculate_outputs`
    6. `finalize_time_step`

    TODO: Need a receive_inputs()
    */

private:
    template <typename ValueType>
    using View = Kokkos::View<ValueType, DeviceType>;

    YAML::Node model_description;
    Model model;                    ///< Kynema-FMB class for model construction
    State<DeviceType> state;        ///< Kynema-FMB class for storing system state (solved for n+1)
    Elements<DeviceType> elements;  ///< Kynema-FMB class for model elements (beams, masses, springs)
    Constraints<DeviceType> constraints;       ///< Kynema-FMB class for constraints tying elements together
    StepParameters parameters;         ///< Kynema-FMB class containing solution parameters
    Solver<DeviceType> solver;         ///< Kynema-FMB class for solving the dynamic system
    State<DeviceType> state_n;      ///< Kynema-FMB class state for storing data at time n
    View<double* [6]> f_nm1;           //< External forces at time n-1
    HostState<DeviceType> host_state;  ///< Host local copy of node state data
    HostConstraints<DeviceType> host_constraints;        ///< Host local copy of constraint data

    /**
     @brief Initializes everything for FMB
     @param structural_description Configuration parameters for the structure
     @param model_description Summary of information and indices of as built model
     */
    static Model BuildModel(const YAML::Node& structural_description,
                            YAML::Node& model_description);

    static void AddTurbine(Model& model, const YAML::Node& turbine_description);
    static void Add6DOF(Model& model,
                        const YAML::Node& dof6_description, 
                        YAML::Node& output_description);
};

}