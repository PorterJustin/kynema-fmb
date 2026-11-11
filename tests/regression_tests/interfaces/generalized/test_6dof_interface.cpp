#include <array>
#include <numbers>
#include <ranges>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "elements/beams/hollow_circle_properties.hpp"
#include "interfaces/generalized/generalized_structure_interface.hpp"

namespace kynema_fmb::tests {

/*
This is intended to test a single 6 DOF model run through
the general structural interface
*/
TEST(GeneralizedInterface, DOF6) {

    const YAML::Node input_yaml =
        YAML::LoadFile("generalized_test_files/general_6dof_inputs.yaml");

    const auto duration =
        input_yaml["simulation"]["duration"].as<double>();  // Simulation duration in seconds
    const auto time_step =
        input_yaml["simulation"]["step_size"].as<double>();  // Time step for the simulation
    const auto write_output{false};                          // Write output file

    auto structure_interface = interfaces::GeneralStructureInterface(input_yaml);
}

}  // namespace kynema_fmb::tests
