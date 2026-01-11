# Model Abstraction Layer

## Title
Create unified model abstraction layer for super-resolution models

## Description

The current implementation is tightly coupled to the sberbank-ai Real-ESRGAN implementation. To support multiple super-resolution models (Real-ESRGAN, EDVR, VSR-Transformer, STAR, etc.), we need to create an abstraction layer that provides a unified interface for all models.

This abstraction layer will:
- Define a common interface that all SR models must implement
- Handle model initialization, weight loading, and inference
- Provide consistent input/output formats across models
- Support model-specific configuration and parameters
- Enable easy addition of new models without changing core code

### Why This Matters
Without this abstraction, adding each new model would require modifying the video processing pipeline and main application logic. This creates maintenance burden and makes the codebase fragile. A well-designed abstraction layer is the foundation for extensibility.

## Acceptance Criteria

1. **Base Model Interface Defined**
   - [ ] Abstract base class `BaseModel` or protocol created
   - [ ] Required methods: `__init__()`, `load_weights()`, `predict()`, `get_scale()`, `get_device()`
   - [ ] Optional methods: `supports_batch()`, `supports_temporal()`, `get_metadata()`

2. **Model Configuration**
   - [ ] Each model can specify its requirements (min input size, channel format, normalization, etc.)
   - [ ] Models can declare supported scales (2x, 4x, 8x)
   - [ ] Models can specify memory requirements
   - [ ] Configuration validation implemented

3. **Input/Output Standardization**
   - [ ] All models accept PIL Image or numpy array
   - [ ] All models return PIL Image or numpy array in consistent format
   - [ ] Color space handling (RGB/BGR) unified
   - [ ] Tensor shape conventions documented and enforced

4. **Error Handling**
   - [ ] Common exceptions defined: `ModelNotFoundError`, `WeightsNotFoundError`, `InferenceError`
   - [ ] Graceful handling of unsupported operations
   - [ ] Clear error messages with suggested fixes

5. **Testing**
   - [ ] Unit tests for base interface
   - [ ] Mock model implementation for testing
   - [ ] Integration test with at least one real model

6. **Documentation**
   - [ ] API documentation for base interface
   - [ ] Guide for implementing new models
   - [ ] Examples of model implementations

## Technical Details

### Proposed Interface Structure

```python
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, List
from PIL import Image
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    """Metadata about a super-resolution model."""
    name: str
    version: str
    scales: List[int]  # Supported upscaling factors
    description: str
    author: str
    paper_url: Optional[str] = None
    implementation_url: Optional[str] = None
    supports_temporal: bool = False  # Multi-frame processing
    supports_batch: bool = True
    min_input_size: tuple[int, int] = (32, 32)
    max_input_size: Optional[tuple[int, int]] = None
    memory_gb_estimate: float = 2.0  # Rough estimate

class ModelConfig:
    """Configuration for model inference."""
    def __init__(
        self,
        device: torch.device,
        scale: int = 4,
        tile_size: Optional[int] = None,
        tile_padding: int = 10,
        half_precision: bool = False,
        **kwargs
    ):
        self.device = device
        self.scale = scale
        self.tile_size = tile_size
        self.tile_padding = tile_padding
        self.half_precision = half_precision
        self.extra_params = kwargs

class BaseModel(ABC):
    """Abstract base class for super-resolution models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model with configuration.
        
        Args:
            config: Model configuration including device and parameters
        """
        self.config = config
        self._model = None
        self._metadata = None
    
    @abstractmethod
    def load_weights(
        self, 
        weights_path: Union[str, Path],
        download: bool = False
    ) -> None:
        """Load model weights from file or download.
        
        Args:
            weights_path: Path to weights file
            download: If True, download weights if not found locally
            
        Raises:
            WeightsNotFoundError: If weights not found and download=False
            ModelLoadError: If weights cannot be loaded
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        image: Union[Image.Image, np.ndarray],
        **kwargs
    ) -> Union[Image.Image, np.ndarray]:
        """Upscale a single image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            **kwargs: Model-specific parameters
            
        Returns:
            Upscaled image in same format as input
            
        Raises:
            InferenceError: If inference fails
        """
        pass
    
    def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        **kwargs
    ) -> List[Union[Image.Image, np.ndarray]]:
        """Upscale a batch of images (default: sequential).
        
        Models that support true batch processing should override this.
        
        Args:
            images: List of input images
            **kwargs: Model-specific parameters
            
        Returns:
            List of upscaled images
        """
        return [self.predict(img, **kwargs) for img in images]
    
    def predict_temporal(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        **kwargs
    ) -> List[Union[Image.Image, np.ndarray]]:
        """Upscale frames with temporal consistency.
        
        Default implementation falls back to frame-by-frame.
        Models with temporal support should override this.
        
        Args:
            frames: List of consecutive frames
            **kwargs: Model-specific parameters
            
        Returns:
            List of upscaled frames with temporal consistency
        """
        if not self.supports_temporal():
            return self.predict_batch(frames, **kwargs)
        raise NotImplementedError("Temporal processing not implemented")
    
    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata.
        
        Returns:
            ModelMetadata object with model information
        """
        pass
    
    def get_scale(self) -> int:
        """Get the upscaling factor.
        
        Returns:
            Upscaling factor (e.g., 2, 4, 8)
        """
        return self.config.scale
    
    def get_device(self) -> torch.device:
        """Get the device model is running on.
        
        Returns:
            PyTorch device
        """
        return self.config.device
    
    def supports_batch(self) -> bool:
        """Check if model supports efficient batch processing.
        
        Returns:
            True if batch processing is optimized
        """
        metadata = self.get_metadata()
        return metadata.supports_batch
    
    def supports_temporal(self) -> bool:
        """Check if model supports temporal consistency.
        
        Returns:
            True if model can process multiple frames with consistency
        """
        metadata = self.get_metadata()
        return metadata.supports_temporal
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Convert image to model input format.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if image.shape[-1] == 4:  # RGBA
            image = image[..., :3]
        
        # Convert to float [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = torch.from_numpy(image).unsqueeze(0)
        
        # Move to device
        image = image.to(self.config.device)
        
        # Half precision if requested
        if self.config.half_precision:
            image = image.half()
        
        return image
    
    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output to image.
        
        Args:
            tensor: Model output tensor
            
        Returns:
            Image as numpy array [H, W, C] in uint8 format
        """
        # Remove batch dimension
        tensor = tensor.squeeze(0)
        
        # Move to CPU
        tensor = tensor.cpu()
        
        # Convert to float32 if half precision
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        
        # CHW to HWC
        image = tensor.numpy().transpose(1, 2, 0)
        
        # Clip to [0, 1]
        image = np.clip(image, 0, 1)
        
        # Convert to uint8
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def __repr__(self) -> str:
        metadata = self.get_metadata()
        return f"{metadata.name} (scale={self.config.scale}, device={self.config.device})"

class ModelFactory:
    """Factory for creating model instances."""
    
    _registry: Dict[str, type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type[BaseModel]):
        """Register a model class.
        
        Args:
            name: Model identifier (e.g., 'realesrgan', 'edvr')
            model_class: Model class implementing BaseModel
        """
        cls._registry[name] = model_class
    
    @classmethod
    def create(cls, name: str, config: ModelConfig) -> BaseModel:
        """Create a model instance.
        
        Args:
            name: Model identifier
            config: Model configuration
            
        Returns:
            Model instance
            
        Raises:
            ModelNotFoundError: If model not registered
        """
        if name not in cls._registry:
            raise ModelNotFoundError(
                f"Model '{name}' not found. Available models: {list(cls._registry.keys())}"
            )
        return cls._registry[name](config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models.
        
        Returns:
            List of model names
        """
        return list(cls._registry.keys())

# Custom Exceptions
class ModelError(Exception):
    """Base exception for model errors."""
    pass

class ModelNotFoundError(ModelError):
    """Model not found in registry."""
    pass

class WeightsNotFoundError(ModelError):
    """Model weights not found."""
    pass

class ModelLoadError(ModelError):
    """Error loading model weights."""
    pass

class InferenceError(ModelError):
    """Error during model inference."""
    pass
```

### Example Implementation

```python
# Example: Wrapping Real-ESRGAN with the new interface

from realesrgan import RealESRGAN as RealESRGANImpl

class RealESRGANModel(BaseModel):
    """Real-ESRGAN model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._impl = None
    
    def load_weights(self, weights_path: Union[str, Path], download: bool = False) -> None:
        try:
            self._impl = RealESRGANImpl(self.config.device, self.config.scale)
            self._impl.load_weights(weights_path, download=download)
        except Exception as e:
            raise ModelLoadError(f"Failed to load Real-ESRGAN weights: {e}")
    
    def predict(self, image: Union[Image.Image, np.ndarray], **kwargs) -> Union[Image.Image, np.ndarray]:
        if self._impl is None:
            raise ModelLoadError("Model weights not loaded. Call load_weights() first.")
        
        try:
            return_pil = isinstance(image, Image.Image)
            if return_pil:
                result = self._impl.predict(image)
            else:
                pil_img = Image.fromarray(image)
                result = self._impl.predict(pil_img)
                result = np.array(result)
            return result
        except Exception as e:
            raise InferenceError(f"Real-ESRGAN inference failed: {e}")
    
    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="Real-ESRGAN",
            version="0.3.0",
            scales=[2, 4],
            description="Real-ESRGAN for practical image restoration",
            author="xinntao",
            paper_url="https://arxiv.org/abs/2107.10833",
            implementation_url="https://github.com/xinntao/Real-ESRGAN",
            supports_temporal=False,
            supports_batch=True,
            memory_gb_estimate=2.5
        )

# Register the model
ModelFactory.register('realesrgan', RealESRGANModel)
```

### File Structure

```
src/video_upscaler/
├── models/
│   ├── __init__.py
│   ├── base.py              # BaseModel, ModelConfig, ModelMetadata
│   ├── factory.py           # ModelFactory
│   ├── exceptions.py        # Custom exceptions
│   ├── realesrgan.py        # Real-ESRGAN wrapper
│   └── utils.py             # Shared utilities
└── ...
```

### Migration Strategy

1. Create new `models` package with base classes
2. Implement Real-ESRGAN wrapper using new interface
3. Update main.py to use ModelFactory instead of direct import
4. Keep old code path temporarily with deprecation warning
5. Remove old implementation once new path is stable

## Dependencies

**Before starting this issue:**
- None (this is the foundation)

**Blocks:**
- Epic 02, Issue 01: Real-ESRGAN Upgrade
- Epic 02, Issue 02: EDVR Integration
- Epic 02, Issue 03: VSR-Transformer Integration
- Epic 02, Issue 04: STAR Model Integration
- Epic 02, Issue 05: Model Registry

## Estimated Effort

**Size:** Large (L)

**Breakdown:**
- Design and review: 8 hours
- Implementation: 16 hours
- Testing: 8 hours
- Documentation: 4 hours
- Code review and iteration: 4 hours

**Total:** ~40 hours (~1 week)

## Priority

**P0** - Critical path for all future model integrations

## Labels

- `epic:core-architecture`
- `type:enhancement`
- `area:models`
- `priority:p0`
- `size:large`
- `good-first-issue` (well-defined scope)

## Related Issues

- Enables: All Epic 02 issues (Model Integration)
- Related: Epic 01, Issue 02 (Video Processing Pipeline)

## Implementation Notes

### Testing Strategy
1. Create comprehensive unit tests for BaseModel interface
2. Create a MockModel for testing without real model dependencies
3. Test error handling paths
4. Test with Real-ESRGAN as integration test

### Review Checklist
- [ ] Interface is intuitive and well-documented
- [ ] Error messages are helpful
- [ ] Code follows project style guide
- [ ] All tests pass
- [ ] Documentation is complete

### Future Considerations
- Model versioning and compatibility
- Model quantization support (int8, mixed precision)
- ONNX export capability
- Model ensembling support
