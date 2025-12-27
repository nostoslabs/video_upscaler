# Model Registry

## Title
Implement centralized model registry and discovery system

## Description

Create a model registry that:
- Catalogs all available models
- Provides model discovery and metadata
- Manages model downloads and caching
- Enables easy model selection
- Supports custom user models

## Acceptance Criteria

1. **Registry Implementation**
   - [ ] ModelRegistry class
   - [ ] Model registration and lookup
   - [ ] Model metadata storage
   - [ ] Version management

2. **Model Discovery**
   - [ ] List available models
   - [ ] Filter by capabilities (scale, temporal support, etc.)
   - [ ] Model recommendations based on use case

3. **Weight Management**
   - [ ] Centralized weight download
   - [ ] Weight caching and versioning
   - [ ] Integrity verification (checksums)
   - [ ] Weight storage management

4. **CLI Integration**
   - [ ] `upscale list-models` command
   - [ ] `upscale download-model` command
   - [ ] Model selection in main command

5. **Testing**
   - [ ] Registry tests
   - [ ] Download and caching tests
   - [ ] Model discovery tests

## Technical Details

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from pathlib import Path
import hashlib
import requests

@dataclass
class ModelEntry:
    """Registry entry for a model."""
    name: str
    model_class: type
    default_weights_url: str
    default_weights_hash: str
    description: str
    scales: List[int]
    recommended_for: List[str]  # ['anime', 'photo', 'video', etc.]

class ModelRegistry:
    """Central registry for all models."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / '.video_upscaler' / 'models'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelEntry] = {}
    
    def register(self, entry: ModelEntry):
        """Register a model."""
        self._models[entry.name] = entry
    
    def get(self, name: str) -> Optional[ModelEntry]:
        """Get model entry by name."""
        return self._models.get(name)
    
    def list_models(self, filter_scale: int = None, filter_temporal: bool = None):
        """List available models with optional filtering."""
        models = list(self._models.values())
        
        if filter_scale:
            models = [m for m in models if filter_scale in m.scales]
        
        return models
    
    def download_weights(self, model_name: str, force: bool = False) -> Path:
        """Download model weights."""
        entry = self.get(model_name)
        if not entry:
            raise ValueError(f"Model not found: {model_name}")
        
        # Check cache
        cache_path = self.cache_dir / f"{model_name}.pth"
        
        if cache_path.exists() and not force:
            # Verify hash
            if self._verify_hash(cache_path, entry.default_weights_hash):
                return cache_path
        
        # Download
        print(f"Downloading {model_name}...")
        response = requests.get(entry.default_weights_url, stream=True)
        response.raise_for_status()
        
        with open(cache_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify
        if not self._verify_hash(cache_path, entry.default_weights_hash):
            cache_path.unlink()
            raise RuntimeError("Downloaded weights failed hash verification")
        
        return cache_path
    
    def _verify_hash(self, path: Path, expected_hash: str) -> bool:
        """Verify file hash."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_hash

# Global registry
_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    return _registry
```

## Dependencies

- Epic 02, Issue 01: Real-ESRGAN Upgrade (first model to register)

## Estimated Effort

**Size:** Medium (M)
**Time:** ~24 hours (3 days)

## Priority

**P0** - Needed for clean model management

## Labels

- `epic:model-integration`
- `type:enhancement`
- `area:models`
- `priority:p0`
- `size:medium`
