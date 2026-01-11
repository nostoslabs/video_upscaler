# Device Management

## Title
Enhanced multi-device support with automatic selection and fallback

## Description

The current device detection logic is minimal - it checks for CUDA, then MPS, then falls back to CPU. For production use, we need more sophisticated device management that handles:
- Multi-GPU systems with device selection
- Memory management and limits
- Automatic fallback when resources are exhausted
- Device capability detection (FP16, TF32 support)
- Cross-platform compatibility
- Device pools for parallel processing

### Why This Matters
Proper device management ensures the application runs optimally across different hardware configurations and fails gracefully when resources are constrained.

## Acceptance Criteria

1. **Device Detection and Selection**
   - [ ] Automatic detection of all available devices
   - [ ] Device capability reporting (compute capability, memory, features)
   - [ ] User can specify device preference
   - [ ] Automatic fallback to next best device if primary unavailable

2. **Multi-GPU Support**
   - [ ] Detect and enumerate all available GPUs
   - [ ] Support for device selection by index
   - [ ] Device pool for distributing work
   - [ ] Load balancing across multiple GPUs

3. **Memory Management**
   - [ ] Query available device memory
   - [ ] Monitor memory usage during processing
   - [ ] Automatic memory cleanup on OOM
   - [ ] Configurable memory limits
   - [ ] Warning when approaching memory limits

4. **Feature Detection**
   - [ ] Detect FP16/BF16 support
   - [ ] Detect TF32 support
   - [ ] Detect fast CUDA kernels
   - [ ] Platform-specific optimizations

5. **Error Handling**
   - [ ] Graceful handling of device errors
   - [ ] Automatic retry on different device
   - [ ] Clear error messages with suggestions
   - [ ] Device diagnostic command

6. **Testing**
   - [ ] Tests for each device type
   - [ ] Mock device for testing edge cases
   - [ ] Memory limit testing
   - [ ] Multi-GPU testing (where available)

## Technical Details

### Device Manager Implementation

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Supported device types."""
    CUDA = 'cuda'
    MPS = 'mps'
    CPU = 'cpu'

@dataclass
class DeviceCapabilities:
    """Device capability information."""
    device: torch.device
    device_type: DeviceType
    name: str
    compute_capability: Optional[tuple[int, int]] = None  # (major, minor) for CUDA
    total_memory: int = 0  # bytes
    available_memory: int = 0  # bytes
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_tf32: bool = False
    cuda_cores: Optional[int] = None
    
    def __repr__(self) -> str:
        mem_gb = self.total_memory / (1024**3) if self.total_memory > 0 else 0
        return (
            f"Device({self.name}, type={self.device_type.value}, "
            f"memory={mem_gb:.1f}GB, fp16={self.supports_fp16})"
        )

class DeviceManager:
    """Manages device detection, selection, and resource allocation."""
    
    def __init__(self):
        self._devices: List[DeviceCapabilities] = []
        self._current_device: Optional[torch.device] = None
        self._memory_limit: Optional[int] = None
        self._detect_devices()
    
    def _detect_devices(self) -> None:
        """Detect all available devices and their capabilities."""
        self._devices = []
        
        # Detect CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                props = torch.cuda.get_device_properties(i)
                
                caps = DeviceCapabilities(
                    device=device,
                    device_type=DeviceType.CUDA,
                    name=props.name,
                    compute_capability=(props.major, props.minor),
                    total_memory=props.total_memory,
                    available_memory=self._get_cuda_available_memory(i),
                    supports_fp16=True,  # All modern CUDA devices support FP16
                    supports_bf16=props.major >= 8,  # Ampere and newer
                    supports_tf32=props.major >= 8,  # Ampere and newer
                    cuda_cores=props.multi_processor_count
                )
                self._devices.append(caps)
                logger.info(f"Detected CUDA device {i}: {caps}")
        
        # Detect MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            
            # MPS memory management is different
            caps = DeviceCapabilities(
                device=device,
                device_type=DeviceType.MPS,
                name="Apple Silicon GPU",
                supports_fp16=True,
                supports_bf16=False,  # Not yet supported on MPS
                supports_tf32=False
            )
            self._devices.append(caps)
            logger.info(f"Detected MPS device: {caps}")
        
        # CPU is always available
        device = torch.device('cpu')
        
        # Estimate available system memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_memory = mem.total
            available_memory = mem.available
        except ImportError:
            total_memory = 0
            available_memory = 0
        
        caps = DeviceCapabilities(
            device=device,
            device_type=DeviceType.CPU,
            name="CPU",
            total_memory=total_memory,
            available_memory=available_memory,
            supports_fp16=False,  # CPU typically doesn't benefit from FP16
            supports_bf16=False,
            supports_tf32=False
        )
        self._devices.append(caps)
        logger.info(f"Detected CPU: {caps}")
    
    def _get_cuda_available_memory(self, device_id: int) -> int:
        """Get available CUDA memory for a device."""
        try:
            torch.cuda.set_device(device_id)
            return torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)
        except Exception as e:
            logger.warning(f"Could not get CUDA memory for device {device_id}: {e}")
            return 0
    
    def get_devices(self) -> List[DeviceCapabilities]:
        """Get all detected devices."""
        return self._devices.copy()
    
    def get_device_by_type(self, device_type: DeviceType) -> List[DeviceCapabilities]:
        """Get all devices of a specific type."""
        return [d for d in self._devices if d.device_type == device_type]
    
    def select_device(
        self,
        preference: str = 'auto',
        device_index: Optional[int] = None,
        min_memory_gb: float = 0
    ) -> torch.device:
        """Select the best device based on preferences.
        
        Args:
            preference: Device preference ('auto', 'cuda', 'mps', 'cpu')
            device_index: Specific device index (for CUDA multi-GPU)
            min_memory_gb: Minimum required memory in GB
            
        Returns:
            Selected torch device
            
        Raises:
            RuntimeError: If no suitable device found
        """
        min_memory_bytes = int(min_memory_gb * 1024**3)
        
        # Handle specific device index
        if device_index is not None:
            cuda_devices = self.get_device_by_type(DeviceType.CUDA)
            if device_index < len(cuda_devices):
                device = cuda_devices[device_index]
                if device.available_memory >= min_memory_bytes:
                    self._current_device = device.device
                    logger.info(f"Selected device: {device}")
                    return self._current_device
                else:
                    raise RuntimeError(
                        f"Device {device_index} does not have enough memory. "
                        f"Required: {min_memory_gb}GB, Available: {device.available_memory / 1024**3:.1f}GB"
                    )
            else:
                raise RuntimeError(f"CUDA device {device_index} not found")
        
        # Auto selection priority: CUDA > MPS > CPU
        if preference == 'auto':
            priority = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]
        elif preference == 'cuda':
            priority = [DeviceType.CUDA, DeviceType.CPU]
        elif preference == 'mps':
            priority = [DeviceType.MPS, DeviceType.CPU]
        elif preference == 'cpu':
            priority = [DeviceType.CPU]
        else:
            raise ValueError(f"Unknown device preference: {preference}")
        
        # Find best device matching priority and memory requirements
        for device_type in priority:
            devices = self.get_device_by_type(device_type)
            
            # Sort by available memory (descending)
            devices.sort(key=lambda d: d.available_memory, reverse=True)
            
            for device in devices:
                if device.available_memory >= min_memory_bytes or device_type == DeviceType.CPU:
                    self._current_device = device.device
                    logger.info(f"Selected device: {device}")
                    return self._current_device
        
        raise RuntimeError(
            f"No suitable device found with {min_memory_gb}GB memory. "
            f"Available devices: {self._devices}"
        )
    
    def get_current_device(self) -> Optional[torch.device]:
        """Get the currently selected device."""
        return self._current_device
    
    def get_device_info(self, device: Optional[torch.device] = None) -> Optional[DeviceCapabilities]:
        """Get capabilities for a specific device."""
        if device is None:
            device = self._current_device
        
        if device is None:
            return None
        
        for caps in self._devices:
            if caps.device == device:
                return caps
        
        return None
    
    def set_memory_limit(self, limit_gb: float) -> None:
        """Set memory usage limit.
        
        Args:
            limit_gb: Memory limit in GB
        """
        self._memory_limit = int(limit_gb * 1024**3)
        logger.info(f"Set memory limit to {limit_gb}GB")
    
    def check_memory_available(self, required_bytes: int) -> bool:
        """Check if enough memory is available.
        
        Args:
            required_bytes: Required memory in bytes
            
        Returns:
            True if enough memory available
        """
        if self._current_device is None:
            return False
        
        info = self.get_device_info(self._current_device)
        if info is None:
            return False
        
        available = info.available_memory
        if self._memory_limit:
            available = min(available, self._memory_limit)
        
        return available >= required_bytes
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage.
        
        Returns:
            Dict with allocated, reserved, total memory
        """
        if self._current_device is None:
            return {'allocated': 0, 'reserved': 0, 'total': 0}
        
        if self._current_device.type == 'cuda':
            device_id = self._current_device.index if self._current_device.index is not None else 0
            return {
                'allocated': torch.cuda.memory_allocated(device_id),
                'reserved': torch.cuda.memory_reserved(device_id),
                'total': torch.cuda.get_device_properties(device_id).total_memory
            }
        else:
            # For CPU/MPS, we can't easily track PyTorch memory
            info = self.get_device_info(self._current_device)
            return {
                'allocated': 0,
                'reserved': 0,
                'total': info.total_memory if info else 0
            }
    
    @contextmanager
    def device_context(self, device: Optional[torch.device] = None):
        """Context manager for temporarily using a specific device.
        
        Args:
            device: Device to use (None = current device)
        """
        if device is None:
            device = self._current_device
        
        if device is None:
            raise RuntimeError("No device selected")
        
        old_device = self._current_device
        
        try:
            if device.type == 'cuda':
                torch.cuda.set_device(device)
            self._current_device = device
            yield device
        finally:
            self._current_device = old_device
            if old_device and old_device.type == 'cuda':
                torch.cuda.set_device(old_device)
    
    def optimize_for_device(self, model: torch.nn.Module) -> None:
        """Apply device-specific optimizations to a model.
        
        Args:
            model: PyTorch model to optimize
        """
        info = self.get_device_info(self._current_device)
        if info is None:
            return
        
        if info.device_type == DeviceType.CUDA:
            # Enable TF32 if supported
            if info.supports_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 for faster computation")
            
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN autotuner")
            
            # Use channels_last memory format if beneficial
            # model = model.to(memory_format=torch.channels_last)
        
        elif info.device_type == DeviceType.MPS:
            # MPS-specific optimizations
            logger.info("Applied MPS optimizations")
        
        elif info.device_type == DeviceType.CPU:
            # CPU-specific optimizations
            torch.set_num_threads(torch.get_num_threads())
            logger.info(f"Using {torch.get_num_threads()} CPU threads")
    
    def clear_cache(self) -> None:
        """Clear device memory cache."""
        if self._current_device and self._current_device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
    
    def diagnose(self) -> str:
        """Generate diagnostic information.
        
        Returns:
            Diagnostic report string
        """
        lines = ["Device Manager Diagnostic Report", "=" * 50]
        
        lines.append(f"\nDetected {len(self._devices)} device(s):")
        for i, device in enumerate(self._devices):
            lines.append(f"\n{i+1}. {device}")
            if device.device_type == DeviceType.CUDA:
                lines.append(f"   Compute Capability: {device.compute_capability}")
                lines.append(f"   Memory: {device.total_memory / 1024**3:.1f}GB total, "
                           f"{device.available_memory / 1024**3:.1f}GB available")
        
        lines.append(f"\nCurrent device: {self._current_device}")
        
        if self._current_device:
            usage = self.get_memory_usage()
            lines.append(f"Memory usage: {usage['allocated'] / 1024**3:.1f}GB allocated, "
                        f"{usage['reserved'] / 1024**3:.1f}GB reserved, "
                        f"{usage['total'] / 1024**3:.1f}GB total")
        
        return "\n".join(lines)

# Global device manager instance
_device_manager: Optional[DeviceManager] = None

def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager
```

### Usage Examples

```python
# Basic usage
manager = get_device_manager()

# Auto-select best device
device = manager.select_device(preference='auto')

# Select specific CUDA device
device = manager.select_device(preference='cuda', device_index=1)

# Select device with minimum memory requirement
device = manager.select_device(preference='auto', min_memory_gb=8.0)

# Get device info
info = manager.get_device_info(device)
print(f"Using {info.name} with {info.total_memory / 1024**3:.1f}GB memory")

# Check if enough memory for operation
if manager.check_memory_available(2 * 1024**3):  # 2GB
    # Proceed with operation
    pass

# Use device context
with manager.device_context(device):
    # Operations on this device
    pass

# Optimize model for device
model = MyModel()
manager.optimize_for_device(model)

# Clear cache when needed
manager.clear_cache()

# Diagnostic info
print(manager.diagnose())
```

### CLI Integration

```python
parser.add_argument('--device', default='auto', help='Device preference (auto, cuda, mps, cpu)')
parser.add_argument('--device-index', type=int, help='Specific GPU index')
parser.add_argument('--memory-limit', type=float, help='Memory limit in GB')
parser.add_argument('--diagnose-devices', action='store_true', help='Show device info and exit')

args = parser.parse_args()

manager = get_device_manager()

if args.diagnose_devices:
    print(manager.diagnose())
    sys.exit(0)

if args.memory_limit:
    manager.set_memory_limit(args.memory_limit)

device = manager.select_device(
    preference=args.device,
    device_index=args.device_index
)
```

## Dependencies

**Before starting this issue:**
- None (independent task)

**Blocks:**
- All model and pipeline tasks benefit from better device management

## Estimated Effort

**Size:** Medium (M)

**Breakdown:**
- Design: 4 hours
- Implementation: 12 hours
- Testing: 4 hours
- Documentation: 4 hours

**Total:** ~24 hours (3 days)

## Priority

**P0** - Foundation for reliable multi-device support

## Labels

- `epic:core-architecture`
- `type:enhancement`
- `area:device`
- `priority:p0`
- `size:medium`

## Implementation Notes

### Testing
- Test on CUDA, MPS, and CPU systems
- Test multi-GPU scenarios
- Test memory limit enforcement
- Test fallback behavior
- Mock devices for CI/CD

### Error Handling
- Handle CUDA out-of-memory gracefully
- Provide actionable error messages
- Auto-retry on different device when possible

### Performance
- Cache device detection results
- Minimize device switching overhead
- Efficient memory tracking
