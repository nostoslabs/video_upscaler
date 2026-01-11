# Configuration System

## Title
Implement flexible configuration management system

## Description

Currently, the application only accepts command-line arguments with hardcoded defaults. To make the tool more flexible and user-friendly, we need a comprehensive configuration system that supports:
- Configuration files (YAML/TOML/JSON)
- Environment variables
- Command-line arguments (highest priority)
- Sensible defaults
- Configuration validation
- Preset profiles for common scenarios

This will make it easier to:
- Reproduce results with saved configurations
- Share configurations between users
- Maintain different profiles for different use cases
- Configure advanced options without cluttering the CLI

### Why This Matters
As the application grows in complexity with more models, options, and features, managing configuration through CLI arguments alone becomes unwieldy. A proper configuration system improves usability and maintainability.

## Acceptance Criteria

1. **Configuration File Support**
   - [ ] Support YAML, TOML, and JSON formats
   - [ ] Config file location customizable via CLI or environment variable
   - [ ] Default config file locations searched (~/.video_upscaler/config.yaml, ./video_upscaler.yaml)
   - [ ] Sample configuration files provided

2. **Configuration Schema**
   - [ ] Well-defined configuration schema with validation
   - [ ] All options documented with types and defaults
   - [ ] Nested configuration for different components (model, pipeline, output)
   - [ ] Schema validation on load with clear error messages

3. **Priority System**
   - [ ] Configuration sources prioritized: CLI > Environment > Config File > Defaults
   - [ ] Partial overrides work correctly (e.g., only override codec, keep other settings)
   - [ ] Clear indication of where each setting comes from (for debugging)

4. **Presets and Profiles**
   - [ ] Built-in presets (fast, balanced, quality, experimental)
   - [ ] User can define custom presets in config file
   - [ ] CLI flag to select preset (--preset quality)
   - [ ] Presets can inherit from each other

5. **Environment Variable Support**
   - [ ] All major settings can be set via environment variables
   - [ ] Clear naming convention (VIDEO_UPSCALER_MODEL, VIDEO_UPSCALER_DEVICE)
   - [ ] Environment variable documentation

6. **Validation and Help**
   - [ ] Configuration validation on load
   - [ ] Schema export command (generate JSON schema for editors)
   - [ ] Config file validation command
   - [ ] Show effective configuration command

## Technical Details

### Proposed Configuration Structure

```yaml
# video_upscaler.yaml - Example configuration file

# Model configuration
model:
  name: realesrgan  # Options: realesrgan, edvr, vsr-transformer, star
  scale: 4          # Upscaling factor: 2, 4, 8
  weights: ~/.video_upscaler/weights/RealESRGAN_x4.pth
  download_weights: true
  
  # Model-specific parameters
  params:
    tile_size: 512  # Process image in tiles (0 = no tiling)
    tile_padding: 10
    half_precision: false

# Device configuration
device:
  type: auto      # Options: auto, cuda, mps, cpu
  cuda_device: 0  # For multi-GPU systems
  allow_tf32: true

# Pipeline configuration
pipeline:
  batch_size: 1     # Number of frames to process together
  num_workers: 4    # CPU workers for data loading
  prefetch: 2       # Number of batches to prefetch

# Video input/output configuration
video:
  # Input settings
  input:
    start_time: null     # Start processing from timestamp (seconds)
    end_time: null       # End processing at timestamp (seconds)
    start_frame: null    # Start from frame number
    end_frame: null      # End at frame number
    fps: null           # Override input FPS
  
  # Output settings
  output:
    codec: libx264       # Options: libx264, libx265, vp9, prores, etc.
    pixel_format: yuv420p
    crf: 18             # Constant Rate Factor (0-51, lower = better)
    preset: medium      # Speed preset: ultrafast, fast, medium, slow, veryslow
    bitrate: null       # Explicit bitrate (overrides CRF)
    two_pass: false     # Two-pass encoding for better quality
    
    # Audio handling
    audio:
      copy: true        # Copy audio stream without re-encoding
      codec: aac        # Audio codec if re-encoding
      bitrate: 192k     # Audio bitrate
    
    # Metadata
    metadata:
      copy: true                    # Copy input metadata
      title: null
      comment: "Upscaled with video_upscaler"

# Processing options
processing:
  temporal_consistency: false  # Enable inter-frame processing
  temporal_window: 5          # Number of frames for temporal processing
  denoise: false              # Apply denoising
  sharpen: 0.0                # Sharpening strength (0-1)

# Performance and resource limits
performance:
  max_memory_gb: null   # Maximum memory to use (null = no limit)
  gpu_memory_fraction: 0.9  # Fraction of GPU memory to use
  cache_dir: ~/.video_upscaler/cache
  
# Logging and output
logging:
  level: INFO          # DEBUG, INFO, WARNING, ERROR
  format: colored      # Options: colored, plain, json
  file: null          # Log file path (null = stdout only)
  
# Progress tracking
progress:
  show_progress: true
  progress_bar: true
  eta: true
  save_checkpoint: false  # Save progress for resumption
  checkpoint_interval: 100  # Frames between checkpoints

# Quality metrics
quality:
  calculate_metrics: false  # Calculate PSNR, SSIM, etc.
  reference_video: null     # Reference video for comparison
  metrics:
    - psnr
    - ssim
    - vmaf
```

### Python Implementation

```python
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import yaml
import toml
import json
from enum import Enum

class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = 'yaml'
    TOML = 'toml'
    JSON = 'json'

@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = 'realesrgan'
    scale: int = 4
    weights: Optional[Path] = None
    download_weights: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceConfig:
    """Device configuration."""
    type: str = 'auto'  # auto, cuda, mps, cpu
    cuda_device: int = 0
    allow_tf32: bool = True

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    batch_size: int = 1
    num_workers: int = 4
    prefetch: int = 2

@dataclass
class VideoInputConfig:
    """Video input configuration."""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    fps: Optional[float] = None

@dataclass
class AudioConfig:
    """Audio configuration."""
    copy: bool = True
    codec: str = 'aac'
    bitrate: str = '192k'

@dataclass
class MetadataConfig:
    """Metadata configuration."""
    copy: bool = True
    title: Optional[str] = None
    comment: str = "Upscaled with video_upscaler"

@dataclass
class VideoOutputConfig:
    """Video output configuration."""
    codec: str = 'libx264'
    pixel_format: str = 'yuv420p'
    crf: int = 18
    preset: str = 'medium'
    bitrate: Optional[str] = None
    two_pass: bool = False
    audio: AudioConfig = field(default_factory=AudioConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

@dataclass
class VideoConfig:
    """Video configuration."""
    input: VideoInputConfig = field(default_factory=VideoInputConfig)
    output: VideoOutputConfig = field(default_factory=VideoOutputConfig)

@dataclass
class ProcessingConfig:
    """Processing configuration."""
    temporal_consistency: bool = False
    temporal_window: int = 5
    denoise: bool = False
    sharpen: float = 0.0

@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_memory_gb: Optional[float] = None
    gpu_memory_fraction: float = 0.9
    cache_dir: Path = Path.home() / '.video_upscaler' / 'cache'

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = 'INFO'
    format: str = 'colored'
    file: Optional[Path] = None

@dataclass
class ProgressConfig:
    """Progress tracking configuration."""
    show_progress: bool = True
    progress_bar: bool = True
    eta: bool = True
    save_checkpoint: bool = False
    checkpoint_interval: int = 100

@dataclass
class QualityConfig:
    """Quality metrics configuration."""
    calculate_metrics: bool = False
    reference_video: Optional[Path] = None
    metrics: List[str] = field(default_factory=lambda: ['psnr', 'ssim'])

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    @classmethod
    def from_file(cls, path: Path) -> 'Config':
        """Load configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Config instance
            
        Raises:
            ValueError: If file format not supported or parsing fails
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Determine format from extension
        ext = path.suffix.lower()
        if ext in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        elif ext == '.toml':
            with open(path) as f:
                data = toml.load(f)
        elif ext == '.json':
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Config instance
        """
        # Recursively create nested dataclass instances
        def create_nested(cls_type, data):
            if data is None:
                return cls_type()
            
            # Handle nested dataclasses
            kwargs = {}
            for field_name, field_type in cls_type.__annotations__.items():
                if field_name in data:
                    value = data[field_name]
                    
                    # Check if field type is a dataclass
                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[field_name] = create_nested(field_type, value)
                    else:
                        kwargs[field_name] = value
            
            return cls_type(**kwargs)
        
        return create_nested(cls, data)
    
    @classmethod
    def from_env(cls, prefix: str = 'VIDEO_UPSCALER') -> Dict[str, Any]:
        """Extract configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Dictionary with configuration overrides
        """
        config = {}
        prefix = f"{prefix}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert env var name to config path
                # VIDEO_UPSCALER_MODEL_NAME -> model.name
                path = key[len(prefix):].lower().split('_')
                
                # Build nested dict
                current = config
                for part in path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Parse value
                current[path[-1]] = cls._parse_env_value(value)
        
        return config
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
        
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_file(self, path: Path, format: Optional[ConfigFormat] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Output file path
            format: File format (auto-detected from extension if None)
        """
        if format is None:
            ext = path.suffix.lower()
            if ext in ['.yaml', '.yml']:
                format = ConfigFormat.YAML
            elif ext == '.toml':
                format = ConfigFormat.TOML
            elif ext == '.json':
                format = ConfigFormat.JSON
            else:
                raise ValueError(f"Cannot determine format from extension: {ext}")
        
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if format == ConfigFormat.YAML:
                yaml.dump(data, f, default_flow_style=False)
            elif format == ConfigFormat.TOML:
                toml.dump(data, f)
            elif format == ConfigFormat.JSON:
                json.dump(data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate model
        if self.model.scale not in [2, 4, 8]:
            errors.append(f"Invalid scale: {self.model.scale}. Must be 2, 4, or 8.")
        
        # Validate device
        if self.device.type not in ['auto', 'cuda', 'mps', 'cpu']:
            errors.append(f"Invalid device type: {self.device.type}")
        
        # Validate CRF
        if not 0 <= self.video.output.crf <= 51:
            errors.append(f"Invalid CRF: {self.video.output.crf}. Must be 0-51.")
        
        # Validate sharpen
        if not 0 <= self.processing.sharpen <= 1:
            errors.append(f"Invalid sharpen: {self.processing.sharpen}. Must be 0-1.")
        
        return errors

class ConfigManager:
    """Manages configuration loading and merging."""
    
    DEFAULT_CONFIG_LOCATIONS = [
        Path.home() / '.video_upscaler' / 'config.yaml',
        Path('./video_upscaler.yaml'),
        Path('./video_upscaler.toml'),
        Path('./video_upscaler.json'),
    ]
    
    PRESETS = {
        'fast': {
            'video': {'output': {'preset': 'ultrafast', 'crf': 23}},
            'pipeline': {'batch_size': 8},
        },
        'balanced': {
            'video': {'output': {'preset': 'medium', 'crf': 20}},
            'pipeline': {'batch_size': 4},
        },
        'quality': {
            'video': {'output': {'preset': 'slow', 'crf': 16}},
            'pipeline': {'batch_size': 1},
        },
        'experimental': {
            'model': {'params': {'tile_size': 256}},
            'processing': {'temporal_consistency': True},
            'video': {'output': {'preset': 'veryslow', 'crf': 14}},
        }
    }
    
    @classmethod
    def load(
        cls,
        config_file: Optional[Path] = None,
        preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Config:
        """Load configuration from multiple sources.
        
        Args:
            config_file: Explicit config file path
            preset: Preset name to use
            overrides: CLI/programmatic overrides
            
        Returns:
            Merged configuration
        """
        # Start with defaults
        config_dict = {}
        
        # Apply preset if specified
        if preset:
            if preset not in cls.PRESETS:
                raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")
            config_dict = cls._deep_merge(config_dict, cls.PRESETS[preset])
        
        # Load from file
        if config_file:
            if config_file.exists():
                file_config = cls._load_file(config_file)
                config_dict = cls._deep_merge(config_dict, file_config)
        else:
            # Try default locations
            for path in cls.DEFAULT_CONFIG_LOCATIONS:
                if path.exists():
                    file_config = cls._load_file(path)
                    config_dict = cls._deep_merge(config_dict, file_config)
                    break
        
        # Apply environment variables
        env_config = Config.from_env()
        config_dict = cls._deep_merge(config_dict, env_config)
        
        # Apply overrides
        if overrides:
            config_dict = cls._deep_merge(config_dict, overrides)
        
        # Create config object
        config = Config.from_dict(config_dict)
        
        # Validate
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return config
    
    @staticmethod
    def _load_file(path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        ext = path.suffix.lower()
        with open(path) as f:
            if ext in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif ext == '.toml':
                return toml.load(f)
            elif ext == '.json':
                return json.load(f)
        return {}
    
    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
```

### CLI Integration

```python
import argparse
from pathlib import Path

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with config support."""
    parser = argparse.ArgumentParser(
        description="Video Upscaler - Upscale videos using super-resolution models"
    )
    
    # Config file
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to configuration file'
    )
    
    # Preset
    parser.add_argument(
        '--preset', '-p',
        choices=['fast', 'balanced', 'quality', 'experimental'],
        help='Use a preset configuration'
    )
    
    # Essential arguments
    parser.add_argument('input', type=Path, help='Input video file')
    parser.add_argument('output', type=Path, help='Output video file')
    
    # Model options
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--scale', type=int, help='Upscaling factor')
    parser.add_argument('--weights', type=Path, help='Model weights path')
    
    # Device options
    parser.add_argument('--device', help='Device to use (auto, cuda, mps, cpu)')
    
    # Output options
    parser.add_argument('--codec', help='Output codec')
    parser.add_argument('--crf', type=int, help='Constant rate factor (quality)')
    parser.add_argument('--preset', dest='encode_preset', help='Encoding preset')
    
    # Show effective config
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show effective configuration and exit'
    )
    
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Build overrides from CLI args
    overrides = {}
    if args.model:
        overrides.setdefault('model', {})['name'] = args.model
    if args.scale:
        overrides.setdefault('model', {})['scale'] = args.scale
    # ... more overrides
    
    # Load configuration
    config = ConfigManager.load(
        config_file=args.config,
        preset=args.preset,
        overrides=overrides
    )
    
    if args.show_config:
        print(yaml.dump(config.to_dict()))
        return
    
    # Use configuration...
```

## Dependencies

**Before starting this issue:**
- None (independent task)

**Blocks:**
- All other issues benefit from configuration system

## Estimated Effort

**Size:** Medium (M)

**Breakdown:**
- Design: 4 hours
- Implementation: 12 hours
- Testing: 4 hours
- Documentation: 4 hours

**Total:** ~24 hours (3 days)

## Priority

**P0** - Needed early for ease of development and testing

## Labels

- `epic:core-architecture`
- `type:enhancement`
- `area:config`
- `priority:p0`
- `size:medium`

## Implementation Notes

### Testing
- Test each config source independently
- Test priority ordering
- Test validation
- Test preset system
- Test with invalid configs

### Documentation
- Provide complete example config files
- Document all options with types and defaults
- Create configuration guide
- Add JSON schema for editor support
