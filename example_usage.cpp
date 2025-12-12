/**
 * Practical C++ Example: Using the Unified Decision Interface
 * Compile: g++ -std=c++17 example_usage.cpp -o decision_example
 */

#include "decision_interface.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

using namespace decision;

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_decision(const Decision& decision) {
    std::cout << "  Decision ID: " << decision.decision_id << "\n";
    std::cout << "  Hardware: " << (decision.hardware_type == HardwareType::STINGRAY ? "Stingray" :
                                    decision.hardware_type == HardwareType::OCTEON ? "OCTEON" :
                                    "TrueNorth") << "\n";
    std::cout << "  Confidence: " << std::fixed << std::setprecision(2)
              << (decision.confidence * 100) << "%\n";
    std::cout << "  Latency: " << std::fixed << std::setprecision(3)
              << decision.latency_ms << "ms\n";

    // Print metadata
    if (!decision.metadata.empty()) {
        std::cout << "  Metadata:\n";
        for (const auto& [key, value] : decision.metadata) {
            std::cout << "    " << key << ": " << value << "\n";
        }
    }
}

void example_1_smart_network() {
    print_header("EXAMPLE 1: Smart Network Processing");

    // Initialize interface
    UnifiedDecisionInterface interface;

    auto octeon = std::make_shared<OcteonEngine>(
        std::unordered_map<std::string, std::string>{
            {"num_cores", "24"},
            {"packet_engines", "8"}
        }
    );
    auto stingray = std::make_shared<StingrayEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "32"}}
    );

    interface.register_engine(octeon, true);
    interface.register_engine(stingray);

    std::cout << "\nProcessing network packets through OCTEON DPU...\n";

    // Simulate different packet types
    std::vector<InputData> packets = {
        {
            .raw_data = std::vector<uint8_t>{'G', 'E', 'T', ' ', '/', 'a', 'p', 'i'},
            .metadata = {{"src_ip", "192.168.1.100"}, {"dst_port", "443"}},
            .data_format = "packet"
        },
        {
            .raw_data = std::vector<uint8_t>{'m', 'a', 'l', 'i', 'c', 'i', 'o', 'u', 's'},
            .metadata = {{"src_ip", "10.0.0.50"}, {"dst_port", "3306"}},
            .data_format = "packet"
        }
    };

    for (size_t i = 0; i < packets.size(); ++i) {
        std::cout << "\nPacket " << (i + 1) << ":\n";
        auto decision = interface.process(packets[i], HardwareType::OCTEON);
        print_decision(decision);

        // Escalate suspicious packets to ML analysis
        if (decision.get_result<std::string>() == "inspect") {
            std::cout << "\n  → Escalating to Stingray for deep ML analysis...\n";
            auto ml_decision = interface.process(packets[i], HardwareType::STINGRAY);
            std::cout << "  → ML Result: " << ml_decision.get_result<int>() << "\n";
        }
    }

    interface.shutdown_all();
}

void example_2_multimodal_processing() {
    print_header("EXAMPLE 2: Multi-Modal Sensor Fusion");

    UnifiedDecisionInterface interface;

    auto stingray = std::make_shared<StingrayEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "32"}}
    );
    auto truenorth = std::make_shared<TrueNorthEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "4096"}}
    );

    interface.register_engine(stingray, true);
    interface.register_engine(truenorth);

    std::cout << "\nProcessing multi-modal sensor data...\n";

    // Camera frame (traditional ML on ARM)
    std::cout << "\n[Camera Feed]\n";
    InputData camera_data{
        .raw_data = std::vector<float>(640 * 480 * 3, 0.5f),
        .metadata = {{"sensor", "front_camera"}, {"type", "rgb"}},
        .data_format = "tensor"
    };
    auto vision_decision = interface.process(camera_data, HardwareType::STINGRAY);
    print_decision(vision_decision);

    // Event sensor (neuromorphic processing)
    std::cout << "\n[Event Sensor]\n";
    InputData event_data{
        .raw_data = std::vector<float>(1024, 0.3f),
        .metadata = {{"sensor", "event_camera"}, {"type", "spikes"}},
        .data_format = "spike_train"
    };
    auto event_decision = interface.process(event_data, HardwareType::TRUENORTH);
    print_decision(event_decision);

    std::cout << "\n[Decision Fusion]\n";
    std::cout << "  Combined confidence: "
              << std::fixed << std::setprecision(2)
              << ((vision_decision.confidence + event_decision.confidence) / 2.0 * 100)
              << "%\n";

    interface.shutdown_all();
}

void example_3_batch_throughput() {
    print_header("EXAMPLE 3: High-Throughput Batch Processing");

    UnifiedDecisionInterface interface;

    auto stingray = std::make_shared<StingrayEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "32"}}
    );
    auto octeon = std::make_shared<OcteonEngine>(
        std::unordered_map<std::string, std::string>{
            {"num_cores", "24"},
            {"packet_engines", "8"}
        }
    );

    interface.register_engine(stingray, true);
    interface.register_engine(octeon);

    std::vector<int> batch_sizes = {10, 100, 1000};

    for (int batch_size : batch_sizes) {
        std::cout << "\n--- Batch Size: " << batch_size << " ---\n";

        // Create batch
        std::vector<InputData> batch;
        for (int i = 0; i < batch_size; ++i) {
            batch.push_back({
                .raw_data = std::vector<float>(224 * 224 * 3, 0.5f),
                .metadata = {{"batch_idx", std::to_string(i)}},
                .data_format = "tensor"
            });
        }

        // Measure throughput
        auto start = std::chrono::high_resolution_clock::now();
        auto decisions = interface.batch_process(batch, HardwareType::STINGRAY);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double throughput = batch_size / (elapsed_ms / 1000.0);

        std::cout << "Stingray (ARM): " << batch_size << " images in "
                  << std::fixed << std::setprecision(2) << elapsed_ms << "ms\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(0)
                  << throughput << " inferences/sec\n";
    }

    interface.shutdown_all();
}

void example_4_hardware_capabilities() {
    print_header("EXAMPLE 4: Hardware Capability Discovery");

    UnifiedDecisionInterface interface;

    auto stingray = std::make_shared<StingrayEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "32"}}
    );
    auto octeon = std::make_shared<OcteonEngine>(
        std::unordered_map<std::string, std::string>{
            {"num_cores", "24"},
            {"packet_engines", "8"}
        }
    );
    auto truenorth = std::make_shared<TrueNorthEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "4096"}}
    );

    interface.register_engine(stingray, true);
    interface.register_engine(octeon);
    interface.register_engine(truenorth);

    std::cout << "\nQuerying hardware capabilities...\n";

    for (const auto& [hw_type, engine] : interface.get_engines()) {
        std::string hw_name = (hw_type == HardwareType::STINGRAY ? "STINGRAY" :
                               hw_type == HardwareType::OCTEON ? "OCTEON 10" :
                               "TRUENORTH");

        std::cout << "\n" << hw_name << "\n";
        std::cout << std::string(50, '-') << "\n";

        auto caps = engine->get_capabilities();
        for (const auto& [key, value] : caps) {
            std::cout << "  " << std::left << std::setw(25) << key << ": " << value << "\n";
        }
    }

    interface.shutdown_all();
}

void example_5_automatic_routing() {
    print_header("EXAMPLE 5: Automatic Hardware Selection");

    UnifiedDecisionInterface interface;

    auto stingray = std::make_shared<StingrayEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "32"}}
    );
    auto octeon = std::make_shared<OcteonEngine>(
        std::unordered_map<std::string, std::string>{
            {"num_cores", "24"},
            {"packet_engines", "8"}
        }
    );
    auto truenorth = std::make_shared<TrueNorthEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "4096"}}
    );

    interface.register_engine(stingray, true);
    interface.register_engine(octeon);
    interface.register_engine(truenorth);

    std::cout << "\nAutomatically routing workloads to optimal hardware...\n";

    // Different input types
    std::vector<InputData> inputs = {
        {
            .raw_data = std::vector<float>(224 * 224 * 3, 0.5f),
            .metadata = {{"type", "image"}},
            .data_format = "tensor"
        },
        {
            .raw_data = std::vector<uint8_t>{'p', 'a', 'c', 'k', 'e', 't'},
            .metadata = {{"type", "network"}},
            .data_format = "packet"
        },
        {
            .raw_data = std::vector<float>(1024, 0.3f),
            .metadata = {{"type", "sensor"}},
            .data_format = "spike_train"
        }
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << "\nInput " << (i + 1) << " (" << inputs[i].data_format << "):\n";

        // Automatic hardware selection
        auto best_hw = interface.get_best_hardware(inputs[i]);
        auto decision = interface.process(inputs[i], best_hw);

        std::string hw_name = (decision.hardware_type == HardwareType::STINGRAY ? "Stingray" :
                               decision.hardware_type == HardwareType::OCTEON ? "OCTEON 10" :
                               "TrueNorth");

        std::cout << "  Routed to: " << hw_name << "\n";
        std::cout << "  Latency: " << std::fixed << std::setprecision(3)
                  << decision.latency_ms << "ms\n";
    }

    interface.shutdown_all();
}

void example_6_power_comparison() {
    print_header("EXAMPLE 6: Power Efficiency Analysis");

    UnifiedDecisionInterface interface;

    auto stingray = std::make_shared<StingrayEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "32"}}
    );
    auto truenorth = std::make_shared<TrueNorthEngine>(
        std::unordered_map<std::string, std::string>{{"num_cores", "4096"}}
    );

    interface.register_engine(stingray, true);
    interface.register_engine(truenorth);

    InputData input{
        .raw_data = std::vector<float>(1024, 0.5f),
        .metadata = {{"type", "pattern"}},
        .data_format = "tensor"
    };

    std::cout << "\nComparing power consumption for identical workload...\n";

    // ARM processing
    std::cout << "\nStingray (ARM):\n";
    auto arm_decision = interface.process(input, HardwareType::STINGRAY);
    double arm_power_watts = 80.0;  // Typical ARM server power
    double arm_energy_mj = arm_power_watts * 1000 * arm_decision.latency_ms / 1000.0;

    std::cout << "  Latency: " << std::fixed << std::setprecision(2)
              << arm_decision.latency_ms << "ms\n";
    std::cout << "  Power: ~" << arm_power_watts << "W\n";
    std::cout << "  Energy: " << std::fixed << std::setprecision(2)
              << arm_energy_mj << " mJ\n";

    // Neuromorphic processing
    std::cout << "\nTrueNorth (Neuromorphic):\n";
    auto neuro_decision = interface.process(input, HardwareType::TRUENORTH);
    double neuro_power_mw = std::stod(neuro_decision.metadata["power_mw"]);
    double neuro_energy_mj = neuro_power_mw * neuro_decision.latency_ms / 1000.0;

    std::cout << "  Latency: " << std::fixed << std::setprecision(2)
              << neuro_decision.latency_ms << "ms\n";
    std::cout << "  Power: " << std::fixed << std::setprecision(2)
              << neuro_power_mw << "mW\n";
    std::cout << "  Energy: " << std::fixed << std::setprecision(2)
              << neuro_energy_mj << " mJ\n";

    double power_savings = ((arm_power_watts * 1000) - neuro_power_mw) / (arm_power_watts * 1000);
    std::cout << "\n→ TrueNorth uses " << std::fixed << std::setprecision(1)
              << (power_savings * 100) << "% less power\n";

    interface.shutdown_all();
}

int main() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  UNIFIED DECISION INTERFACE - C++ EXAMPLES\n";
    std::cout << "  Heterogeneous Hardware Abstraction Demo\n";
    std::cout << std::string(70, '=') << "\n";

    try {
        example_1_smart_network();
        example_2_multimodal_processing();
        example_3_batch_throughput();
        example_4_hardware_capabilities();
        example_5_automatic_routing();
        example_6_power_comparison();

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "All examples completed successfully!\n";
        std::cout << std::string(70, '=') << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
