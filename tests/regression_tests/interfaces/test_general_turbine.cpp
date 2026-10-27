#include <array>
#include <numbers>
#include <ranges>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "elements/beams/hollow_circle_properties.hpp"
#include "interfaces/generalized/generalized_structure_interface.hpp"

namespace kynema_fmb::tests {

/*
This test uses the generalized interface to simulate a structure (turbine)
and a controller.

This is intended to exactly recreate 'test_turbine_controller_interface.cpp'

'test_blade_interface.cpp'
*/
TEST(GeneralizedInterface, TurbineStructureController) {
    const YAML::Node input_yaml =
        YAML::LoadFile("interfaces_test_files/general_interface_turbine_inputs.yaml");

    const auto angular_speed = input_yaml["components"][0]["turbine"]["angular_speed"].as<double>();
    const auto duration =
        input_yaml["simulation"]["duration"].as<double>();  // Simulation duration in seconds
    const auto time_step =
        input_yaml["simulation"]["time_step"].as<double>();  // Time step for the simulation
    const auto write_output{false};                          // Write output file

    auto structure_interface = interfaces::GeneralStructureInterface(input_yaml);
}

}  // namespace kynema_fmb::tests
