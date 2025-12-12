# Generic Decision Interface for Heterogeneous Hardware

A unified abstraction layer for making decisions across vastly different hardware architectures: general-purpose ARM processors, specialized DPUs, and neuromorphic chips.

## Supported Hardware

### 1. **Broadcom Stingray** (ARM Server Processor)
- **Architecture**: 32-core ARM Cortex-A72
- **Optimized for**: General-purpose compute, traditional ML inference
- **Key Features**:
  - ARM NEON SIMD optimizations
  - FP16 half-precision support
  - Multi-threaded batch processing
  - Compatible with ONNX, TensorFlow Lite, PyTorch

### 2. **Marvell OCTEON 10** (Data Processing Unit)
- **Architecture**: 24-core MIPS64-based DPU
- **Optimized for**: Network packet processing, real-time traffic decisions
- **Key Features**:
  - Hardware accelerators: crypto, compression, regex, deep packet inspection
  - 400 Mpps packet processing rate
  - Ultra-low latency classification
  - Inline security policy enforcement

### 3. **IBM TrueNorth** (Neuromorphic Chip)
- **Architecture**: 4,096 neurosynaptic cores (1M neurons, 256M synapses)
- **Optimized for**: Spiking neural networks, ultra-low-power pattern recognition
- **Key Features**:
  - Event-driven asynchronous processing
  - 0.1 mW per inference (70 mW total chip power)
  - Spike-based computing
  - Real-time sensory processing

## Interface Design

### Core Concepts

1. **Unified Input**: `InputData` structure accepts various formats (tensors, packets, spike trains)
2. **Unified Output**: `Decision` structure with hardware-agnostic results
3. **Hardware Abstraction**: `DecisionEngine` abstract base class
4. **Automatic Routing**: Select optimal hardware based on input characteristics

### Architecture

```
┌─────────────────────────────────────────┐
│    UnifiedDecisionInterface             │
│  (High-level orchestration)             │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       │ DecisionEngine │ (Abstract Base)
       └───────┬────────┘
               │
   ┌───────────┼───────────┐
   │           │           │
┌──▼────┐  ┌──▼─────┐  ┌──▼────────┐
│Stingray│  │OCTEON  │  │TrueNorth  │
│Engine  │  │Engine  │  │Engine     │
└────────┘  └────────┘  └───────────┘
     │           │             │
┌────▼────┐ ┌───▼─────┐ ┌─────▼──────┐
│ARM Cores│ │Packet   │ │Neurosynaptic│
│+ NEON   │ │Engines  │ │Cores        │
└─────────┘ └─────────┘ └────────────┘
```

## Usage

### Python Example

```python
from decision_interface import *
import numpy as np

# Initialize unified interface
interface = UnifiedDecisionInterface()

# Register hardware backends
interface.register_engine(StingrayEngine({"num_cores": 32}), set_default=True)
interface.register_engine(OcteonEngine({"num_cores": 24, "packet_engines": 8}))
interface.register_engine(TrueNorthEngine({"num_cores": 4096}))

# Example 1: Image classification on ARM
image = InputData(
    raw_data=np.random.rand(224, 224, 3),
    metadata={"type": "image"},
    data_format="tensor"
)
decision = interface.process(image, HardwareType.STINGRAY)
print(f"Class: {decision.result}, Confidence: {decision.confidence}")

# Example 2: Network packet on DPU
packet = InputData(
    raw_data=b"malicious payload",
    metadata={"src_ip": "192.168.1.100"},
    data_format="packet"
)
decision = interface.process(packet, HardwareType.OCTEON)
print(f"Action: {decision.result}")

# Example 3: Neuromorphic pattern recognition
spikes = InputData(
    raw_data=np.random.rand(1024),
    metadata={"type": "sensor"},
    data_format="spike_train"
)
decision = interface.process(spikes, HardwareType.TRUENORTH)
print(f"Result: {decision.result}, Power: {decision.metadata['power_mw']}mW")

# Automatic hardware selection
auto_decision = interface.process(
    packet,
    interface.get_best_hardware(packet)
)

# Batch processing
batch = [image] * 100
decisions = interface.batch_process(batch, HardwareType.STINGRAY)
```

### C++ Example

```cpp
#include "decision_interface.hpp"
#include <iostream>

using namespace decision;

int main() {
    // Initialize unified interface
    UnifiedDecisionInterface interface;

    // Register engines
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

    // Create input
    InputData input{
        .raw_data = std::vector<float>(224 * 224 * 3, 0.5f),
        .metadata = {{"type", "image"}},
        .data_format = "tensor"
    };

    // Process
    auto decision = interface.process(input, HardwareType::STINGRAY);

    std::cout << "Decision: " << decision.get_result<int>()
              << ", Confidence: " << decision.confidence
              << ", Latency: " << decision.latency_ms << "ms\n";

    // Cleanup
    interface.shutdown_all();

    return 0;
}
```

## Performance Characteristics

| Hardware | Throughput | Latency | Power | Best For |
|----------|------------|---------|-------|----------|
| **Stingray** | 100-1000 inferences/sec | 1-10ms | 50-120W | General ML, CNNs, transformers |
| **OCTEON 10** | 400M packets/sec | <1μs | 50-90W | Packet classification, DPI, security |
| **TrueNorth** | 1000 patterns/sec | 1-5ms | 70mW | Sensor fusion, edge AI, always-on |

## Hardware Selection Strategy

The interface automatically routes workloads based on input characteristics:

```python
def get_best_hardware(input_data: InputData) -> HardwareType:
    if input_data.data_format == "packet":
        return HardwareType.OCTEON       # DPU excels at packet processing
    elif input_data.data_format in ["spike_train", "event"]:
        return HardwareType.TRUENORTH    # Neuromorphic for event-driven
    else:
        return HardwareType.STINGRAY     # ARM for general compute
```

## Use Cases

### 1. Edge AI Gateway
```python
# Multi-modal processing on heterogeneous hardware
video_stream → STINGRAY (object detection)
network_traffic → OCTEON (threat detection)
sensor_events → TRUENORTH (anomaly detection)
```

### 2. Smart NIC with AI
```python
# Inline processing at line rate
incoming_packets → OCTEON (L2-L7 classification)
    ↓ suspicious packets
    → STINGRAY (deep ML analysis)
    → TRUENORTH (behavioral pattern matching)
```

### 3. Autonomous Systems
```python
# Real-time multi-sensor fusion
camera_frames → STINGRAY (CNN-based vision)
lidar_points → STINGRAY (3D object detection)
radar_events → TRUENORTH (event-based processing)
    ↓ all decisions
    → Decision Fusion → Control Output
```

## Real Hardware Integration

### Stingray Integration
```python
# Use actual ARM Compute Library
from arm_compute import cl  # OpenCL backend
from onnxruntime import InferenceSession

class StingrayEngine(DecisionEngine):
    def initialize(self):
        # Load ONNX model with ARM optimizations
        self.session = InferenceSession(
            "model.onnx",
            providers=['CPUExecutionProvider'],
            provider_options=[{'arena_extend_strategy': 'kSameAsRequested'}]
        )
```

### OCTEON Integration
```python
# Use OCTEON SDK
import octeon_sdk

class OcteonEngine(DecisionEngine):
    def initialize(self):
        # Initialize packet processing units
        self.ppu = octeon_sdk.PacketProcessingUnit()
        self.ppu.load_rules("firewall_rules.yml")
        self.ppu.enable_hardware_offload(["dpi", "regex", "crypto"])
```

### TrueNorth Integration
```python
# Use Corelet Programming Environment
from truenorth import CoreletRuntime, SNNModel

class TrueNorthEngine(DecisionEngine):
    def initialize(self):
        # Load spiking neural network
        self.runtime = CoreletRuntime()
        self.snn = SNNModel.from_file("model.corelet")
        self.runtime.deploy(self.snn)
```

## Extensibility

### Adding New Hardware

```python
class CustomAcceleratorEngine(DecisionEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hardware_type = HardwareType.CUSTOM_ACCEL

    def initialize(self) -> bool:
        # Initialize your hardware
        return True

    def process(self, input_data: InputData) -> Decision:
        # Implement processing logic
        pass

    def batch_process(self, inputs: List[InputData]) -> List[Decision]:
        # Implement batch processing
        pass

    def shutdown(self) -> None:
        # Cleanup
        pass
```

### Custom Decision Types

```python
@dataclass
class DetailedDecision(Decision):
    """Extended decision with additional metadata"""
    attention_map: np.ndarray
    intermediate_features: List[np.ndarray]
    execution_trace: List[str]
```

## Testing

```bash
# Run the Python example
python decision_interface.py

# Expected output:
# [Stingray] Initializing 32 ARM cores
# [OCTEON 10] Initializing 24 cores, 8 packet engines
# [TrueNorth] Initializing 4096 neurosynaptic cores
#
# Test 1: Image classification on Stingray
# Decision: 3, Confidence: 0.45, Latency: 0.23ms
#
# Test 2: Network packet processing on OCTEON 10
# Decision: drop, Confidence: 0.95, Latency: 0.08ms
#
# Test 3: Neuromorphic pattern recognition on TrueNorth
# Decision: 7, Confidence: 0.32, Latency: 1.45ms, Power: 0.05mW
```

## Design Principles

1. **Hardware Agnostic**: Application code doesn't need to know hardware details
2. **Zero Copy**: Minimize data movement between CPU and accelerators
3. **Async Ready**: Support for asynchronous processing (futures/promises)
4. **Batching**: Maximize throughput via batch processing
5. **Fallback**: Graceful degradation if hardware unavailable
6. **Observability**: Rich metadata for monitoring and debugging

## Future Enhancements

- [ ] Async/await API for non-blocking processing
- [ ] Multi-engine ensemble decisions (majority voting)
- [ ] Dynamic workload balancing across hardware
- [ ] Hardware capability negotiation
- [ ] Zero-copy shared memory between engines
- [ ] Distributed processing across multiple nodes
- [ ] Real-time performance telemetry
- [ ] Auto-tuning for optimal hardware selection

## License

MIT License - feel free to adapt for your hardware platforms.

## References

- [Broadcom Stingray Product Brief](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches/stingray)
- [Marvell OCTEON 10 DPU](https://www.marvell.com/products/data-processing-units.html)
- [IBM TrueNorth Research](https://research.ibm.com/articles/truenorth)
- [ARM Compute Library](https://github.com/ARM-software/ComputeLibrary)
- [Neuromorphic Computing](https://en.wikipedia.org/wiki/Neuromorphic_engineering)
