"""
Practical Examples: Using the Unified Decision Interface
Demonstrates real-world scenarios across different hardware
"""

from decision_interface import *
import numpy as np
import time


def example_1_smart_firewall():
    """
    Smart Firewall: Use OCTEON for packet filtering + Stingray for ML threat detection
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Smart Firewall with Hybrid Processing")
    print("="*70)

    interface = UnifiedDecisionInterface()
    interface.register_engine(OcteonEngine({"num_cores": 24, "packet_engines": 8}))
    interface.register_engine(StingrayEngine({"num_cores": 32}))

    # Scenario: Incoming network traffic
    packets = [
        InputData(
            raw_data=b"GET /api/users HTTP/1.1",
            metadata={"src_ip": "192.168.1.100", "dst_port": 443},
            data_format="packet"
        ),
        InputData(
            raw_data=b"malicious SQL injection attempt",
            metadata={"src_ip": "10.0.0.50", "dst_port": 3306},
            data_format="packet"
        ),
        InputData(
            raw_data=b"authorized admin session",
            metadata={"src_ip": "192.168.1.1", "dst_port": 22},
            data_format="packet"
        ),
    ]

    print("\nProcessing packets through OCTEON DPU...")
    for i, packet in enumerate(packets, 1):
        # Fast hardware-based classification
        decision = interface.process(packet, HardwareType.OCTEON)

        print(f"\nPacket {i}:")
        print(f"  Source: {packet.metadata['src_ip']}")
        print(f"  Action: {decision.result}")
        print(f"  Confidence: {decision.confidence:.2%}")
        print(f"  Latency: {decision.latency_ms:.3f}ms")

        # If suspicious, escalate to ML analysis on ARM
        if decision.result == "inspect":
            print(f"  → Escalating to Stingray for deep ML analysis...")
            ml_decision = interface.process(packet, HardwareType.STINGRAY)
            print(f"  → ML Result: {ml_decision.result}")

    interface.shutdown_all()


def example_2_autonomous_vehicle():
    """
    Autonomous Vehicle: Multi-sensor fusion across heterogeneous hardware
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Autonomous Vehicle Multi-Sensor Processing")
    print("="*70)

    interface = UnifiedDecisionInterface()
    interface.register_engine(StingrayEngine({"num_cores": 32}))
    interface.register_engine(TrueNorthEngine({"num_cores": 4096}))

    # Simulate sensors
    print("\nProcessing multi-modal sensor data...")

    # Camera feed → Stingray (CNN for object detection)
    camera_frame = InputData(
        raw_data=np.random.rand(640, 480, 3),
        metadata={"sensor": "front_camera", "timestamp": time.time()},
        data_format="tensor"
    )
    vision_decision = interface.process(camera_frame, HardwareType.STINGRAY)
    print(f"\n[Camera] Object detected: Class {vision_decision.result}")
    print(f"         Confidence: {vision_decision.confidence:.2%}")
    print(f"         Latency: {vision_decision.latency_ms:.2f}ms")

    # Event-based sensor → TrueNorth (neuromorphic processing)
    event_stream = InputData(
        raw_data=np.random.rand(1024),
        metadata={"sensor": "event_camera", "timestamp": time.time()},
        data_format="spike_train"
    )
    event_decision = interface.process(event_stream, HardwareType.TRUENORTH)
    print(f"\n[Event Sensor] Pattern: {event_decision.result}")
    print(f"               Confidence: {event_decision.confidence:.2%}")
    print(f"               Power: {event_decision.metadata['power_mw']}mW")
    print(f"               Latency: {event_decision.latency_ms:.2f}ms")

    # Decision fusion
    print("\n[Fusion] Combining decisions from multiple modalities...")
    print(f"         → Vision confidence: {vision_decision.confidence:.2%}")
    print(f"         → Event confidence: {event_decision.confidence:.2%}")
    print(f"         → Final decision: SAFE TO PROCEED")

    interface.shutdown_all()


def example_3_edge_ai_gateway():
    """
    Edge AI Gateway: Route different workloads to optimal hardware
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Edge AI Gateway with Intelligent Routing")
    print("="*70)

    interface = UnifiedDecisionInterface()
    interface.register_engine(StingrayEngine({"num_cores": 32}))
    interface.register_engine(OcteonEngine({"num_cores": 24, "packet_engines": 8}))
    interface.register_engine(TrueNorthEngine({"num_cores": 4096}))

    # Various input types
    inputs = [
        InputData(
            raw_data=np.random.rand(224, 224, 3),
            metadata={"type": "image", "source": "security_camera"},
            data_format="tensor"
        ),
        InputData(
            raw_data=b"network packet data",
            metadata={"type": "packet", "protocol": "tcp"},
            data_format="packet"
        ),
        InputData(
            raw_data=np.random.rand(512),
            metadata={"type": "sensor", "source": "vibration_monitor"},
            data_format="spike_train"
        ),
    ]

    print("\nAutomatically routing workloads to optimal hardware...")

    for i, input_data in enumerate(inputs, 1):
        # Automatically select best hardware
        best_hw = interface.get_best_hardware(input_data)
        decision = interface.process(input_data, best_hw)

        print(f"\nInput {i}: {input_data.metadata['type']}")
        print(f"  Routed to: {decision.hardware_type.value}")
        print(f"  Result: {decision.result}")
        print(f"  Latency: {decision.latency_ms:.3f}ms")
        print(f"  Metadata: {decision.metadata}")

    interface.shutdown_all()


def example_4_batch_processing():
    """
    Batch Processing: Maximize throughput with batched operations
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: High-Throughput Batch Processing")
    print("="*70)

    interface = UnifiedDecisionInterface()
    interface.register_engine(StingrayEngine({"num_cores": 32}))
    interface.register_engine(OcteonEngine({"num_cores": 24, "packet_engines": 8}))

    # Generate large batch of data
    batch_sizes = [10, 100, 1000]

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")

        # Image batch on Stingray
        image_batch = [
            InputData(
                raw_data=np.random.rand(224, 224, 3),
                metadata={"batch_idx": i},
                data_format="tensor"
            )
            for i in range(batch_size)
        ]

        start = time.time()
        decisions = interface.batch_process(image_batch, HardwareType.STINGRAY)
        stingray_time = time.time() - start

        print(f"Stingray (ARM): {batch_size} images in {stingray_time*1000:.2f}ms")
        print(f"  Throughput: {batch_size/stingray_time:.0f} inferences/sec")

        # Packet batch on OCTEON
        packet_batch = [
            InputData(
                raw_data=b"packet data",
                metadata={"batch_idx": i},
                data_format="packet"
            )
            for i in range(batch_size)
        ]

        start = time.time()
        decisions = interface.batch_process(packet_batch, HardwareType.OCTEON)
        octeon_time = time.time() - start

        print(f"OCTEON (DPU): {batch_size} packets in {octeon_time*1000:.2f}ms")
        print(f"  Throughput: {batch_size/octeon_time:.0f} packets/sec")

    interface.shutdown_all()


def example_5_hardware_capabilities():
    """
    Hardware Introspection: Query capabilities and constraints
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Hardware Capability Discovery")
    print("="*70)

    interface = UnifiedDecisionInterface()
    interface.register_engine(StingrayEngine({"num_cores": 32}))
    interface.register_engine(OcteonEngine({"num_cores": 24, "packet_engines": 8}))
    interface.register_engine(TrueNorthEngine({"num_cores": 4096}))

    print("\nQuerying hardware capabilities...\n")

    for hw_type, engine in interface.engines.items():
        caps = engine.get_capabilities()

        print(f"{hw_type.value.upper()}")
        print("-" * 50)
        for key, value in caps.items():
            print(f"  {key:25s}: {value}")
        print()

    interface.shutdown_all()


def example_6_fallback_strategy():
    """
    Fallback Strategy: Handle hardware failures gracefully
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Hardware Fallback and Resilience")
    print("="*70)

    interface = UnifiedDecisionInterface()
    interface.register_engine(StingrayEngine({"num_cores": 32}))
    interface.register_engine(OcteonEngine({"num_cores": 24}))

    # Try processing with fallback
    input_data = InputData(
        raw_data=np.random.rand(224, 224, 3),
        metadata={"type": "image"},
        data_format="tensor"
    )

    # Preferred order: Try TrueNorth first (not registered), fall back to Stingray
    print("\nAttempting processing with fallback strategy...")
    print("Preferred order: TrueNorth → OCTEON → Stingray")

    try:
        decision = interface.process_with_fallback(
            input_data,
            [HardwareType.TRUENORTH, HardwareType.OCTEON, HardwareType.STINGRAY]
        )
        print(f"\nSuccessfully processed on: {decision.hardware_type.value}")
        print(f"Result: {decision.result}")
        print(f"Latency: {decision.latency_ms:.2f}ms")
    except Exception as e:
        print(f"\nError: {e}")

    interface.shutdown_all()


def example_7_power_efficiency():
    """
    Power Efficiency: Compare energy consumption across hardware
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Power Efficiency Comparison")
    print("="*70)

    interface = UnifiedDecisionInterface()
    interface.register_engine(StingrayEngine({"num_cores": 32}))
    interface.register_engine(TrueNorthEngine({"num_cores": 4096}))

    input_data = InputData(
        raw_data=np.random.rand(1024),
        metadata={"type": "pattern"},
        data_format="tensor"
    )

    print("\nComparing power consumption for same workload...\n")

    # Process on Stingray
    decision_arm = interface.process(input_data, HardwareType.STINGRAY)
    estimated_power_arm = 80  # watts (typical ARM server)

    print(f"Stingray (ARM):")
    print(f"  Latency: {decision_arm.latency_ms:.2f}ms")
    print(f"  Power: ~{estimated_power_arm}W")
    print(f"  Energy: {estimated_power_arm * decision_arm.latency_ms / 1000:.2f} mJ")

    # Process on TrueNorth
    decision_neuro = interface.process(input_data, HardwareType.TRUENORTH)
    power_neuro = float(decision_neuro.metadata.get('power_mw', 70))

    print(f"\nTrueNorth (Neuromorphic):")
    print(f"  Latency: {decision_neuro.latency_ms:.2f}ms")
    print(f"  Power: {power_neuro}mW")
    print(f"  Energy: {power_neuro * decision_neuro.latency_ms / 1000:.2f} mJ")

    energy_savings = ((estimated_power_arm * 1000) - power_neuro) / (estimated_power_arm * 1000)
    print(f"\n→ TrueNorth uses {energy_savings:.1%} less power for this workload")

    interface.shutdown_all()


if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("  UNIFIED DECISION INTERFACE - PRACTICAL EXAMPLES")
    print("  Heterogeneous Hardware Abstraction Demo")
    print("="*70)

    # Run all examples
    example_1_smart_firewall()
    example_2_autonomous_vehicle()
    example_3_edge_ai_gateway()
    example_4_batch_processing()
    example_5_hardware_capabilities()
    example_6_fallback_strategy()
    example_7_power_efficiency()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")
