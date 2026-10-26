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

    /*
    1. `initialize_module`
    2. `initialize_stage1` - have multiple extra stages based on information sharing at init.
    3. `predict_inputs`
    4. `advance_states`
    5. `calculate_outputs`
    6. `finalize_time_step`
    */

private:
    // Model model;                    ///< Kynema-FMB class for model construction
    // State<DeviceType> state;        ///< Kynema-FMB class for storing system state
    // Elements<DeviceType> elements;  ///< Kynema-FMB class for model elements (beams, masses, springs)
    // Constraints<DeviceType> constraints;       ///< Kynema-FMB class for constraints tying elements together
    // StepParameters parameters;         ///< Kynema-FMB class containing solution parameters
    // Solver<DeviceType> solver;         ///< Kynema-FMB class for solving the dynamic system
    // State<DeviceType> state_save;      ///< Kynema-FMB class state class for temporarily saving state
    // HostState<DeviceType> host_state;  ///< Host local copy of node state data
    // HostConstraints<DeviceType> host_constraints;        ///< Host local copy of constraint data
};

}