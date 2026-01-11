# Real-ESRGAN Upgrade

## Title
Migrate from sberbank-ai to xinntao Real-ESRGAN with basicsr

## Description

Replace the outdated sberbank-ai/Real-ESRGAN fork with the actively maintained xinntao/Real-ESRGAN implementation. The xinntao version includes:
- Better maintenance and bug fixes
- Support for more models (RealESRGAN, RealESRNet, RealESRGAN_x4plus_anime)
- Integration with BasicSR framework
- Better video support
- Improved performance

This migration is essential as the sberbank-ai fork is no longer maintained and lacks recent improvements.

## Acceptance Criteria

1. **Dependency Update**
   - [ ] Remove sberbank-ai/Real-ESRGAN git dependency
   - [ ] Add basicsr and realesrgan packages
   - [ ] Update requirements and lock file
   - [ ] Verify no conflicts with other dependencies

2. **Model Wrapper Implementation**
   - [ ] Implement RealESRGANModel using BaseModel interface
   - [ ] Support for RealESRGAN_x4plus
   - [ ] Support for RealESRGAN_x2plus
   - [ ] Support for RealESRGANer (xinntao's wrapper class)
   - [ ] Automatic weight download from GitHub releases

3. **Backward Compatibility**
   - [ ] Support for existing weight files
   - [ ] Migration path documented
   - [ ] Deprecation warnings for old API

4. **Testing**
   - [ ] Unit tests for model wrapper
   - [ ] Integration tests with actual weights
   - [ ] Comparison tests (output quality similar or better)
   - [ ] Performance benchmarks

5. **Documentation**
   - [ ] Updated README with new model info
   - [ ] Migration guide for existing users
   - [ ] Model comparison and recommendations

## Technical Details

### Implementation

```python
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from video_upscaler.models.base import BaseModel, ModelConfig, ModelMetadata
import torch
from PIL import Image
import numpy as np
from typing import Union
import requests
from pathlib import Path

class RealESRGANModel(BaseModel):
    """Real-ESRGAN model using xinntao implementation."""
    
    MODELS = {
        'RealESRGAN_x4plus': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'scale': 4,
            'arch': 'RRDBNet',
            'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32}
        },
        'RealESRGAN_x2plus': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'scale': 2,
            'arch': 'RRDBNet',
            'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32}
        },
        'RealESRGAN_x4plus_anime': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            'scale': 4,
            'arch': 'RRDBNet',
            'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 6, 'num_grow_ch': 32}
        },
        'RealESRNet_x4plus': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
            'scale': 4,
            'arch': 'RRDBNet',
            'params': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32}
        },
    }
    
    def __init__(self, config: ModelConfig, model_name: str = 'RealESRGAN_x4plus'):
        super().__init__(config)
        self.model_name = model_name
        self._upsampler = None
        
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        
        self.model_info = self.MODELS[model_name]
    
    def load_weights(self, weights_path: Union[str, Path], download: bool = False) -> None:
        weights_path = Path(weights_path)
        
        # Download if needed
        if not weights_path.exists() and download:
            self._download_weights(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        # Create model architecture
        if self.model_info['arch'] == 'RRDBNet':
            model = RRDBNet(**self.model_info['params'])
        else:
            raise ValueError(f"Unknown architecture: {self.model_info['arch']}")
        
        # Create upsampler
        self._upsampler = RealESRGANer(
            scale=self.config.scale,
            model_path=str(weights_path),
            model=model,
            tile=self.config.tile_size or 0,
            tile_pad=self.config.tile_padding,
            pre_pad=0,
            half=self.config.half_precision,
            device=self.config.device
        )
    
    def _download_weights(self, weights_path: Path) -> None:
        """Download model weights from GitHub."""
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        url = self.model_info['url']
        print(f"Downloading {self.model_name} from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to {weights_path}")
    
    def predict(self, image: Union[Image.Image, np.ndarray], **kwargs) -> Union[Image.Image, np.ndarray]:
        if self._upsampler is None:
            raise RuntimeError("Model not loaded. Call load_weights() first.")
        
        return_pil = isinstance(image, Image.Image)
        
        # Convert to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Upscale
        output, _ = self._upsampler.enhance(image, outscale=self.config.scale)
        
        # Convert back to PIL if needed
        if return_pil:
            output = Image.fromarray(output)
        
        return output
    
    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=f"Real-ESRGAN ({self.model_name})",
            version="0.3.0",
            scales=[self.model_info['scale']],
            description="Real-ESRGAN for practical image/video restoration",
            author="xinntao",
            paper_url="https://arxiv.org/abs/2107.10833",
            implementation_url="https://github.com/xinntao/Real-ESRGAN",
            supports_temporal=False,
            supports_batch=False,  # RealESRGANer doesn't have batch support
            memory_gb_estimate=2.5
        )

# Register model
from video_upscaler.models import ModelFactory
ModelFactory.register('realesrgan', RealESRGANModel)
```

### Migration Guide

**For users:**
```bash
# Old way
poetry run upscale video.mp4

# New way (same command, but better model under the hood)
poetry run upscale video.mp4

# New: Select specific model variant
poetry run upscale video.mp4 --model realesrgan --model-variant RealESRGAN_x4plus_anime
```

**For developers:**
```python
# Old way
from RealESRGAN import RealESRGAN
upscaler = RealESRGAN(device, scale=4)
upscaler.load_weights(weights_path, download=True)
result = upscaler.predict(image)

# New way
from video_upscaler.models import ModelFactory, ModelConfig
config = ModelConfig(device=device, scale=4)
model = ModelFactory.create('realesrgan', config)
model.load_weights(weights_path, download=True)
result = model.predict(image)
```

## Dependencies

**Before starting:**
- Epic 01, Issue 01: Model Abstraction Layer

**Blocks:**
- Epic 02, Issue 05: Model Registry (needs at least one working model)

## Estimated Effort

**Size:** Medium (M)
**Time:** ~20 hours (2.5 days)

## Priority

**P0** - Critical for maintaining functionality

## Labels

- `epic:model-integration`
- `type:enhancement`
- `area:models`
- `priority:p0`
- `size:medium`
