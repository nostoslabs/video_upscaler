# EDVR Integration

## Title
Integrate EDVR for temporal-aware video super-resolution

## Description

EDVR (Enhanced Deformable Video Restoration) is a multi-frame video SR model that uses deformable convolutions and temporal alignment to leverage information from neighboring frames. Unlike frame-by-frame methods, EDVR provides better temporal consistency and detail preservation.

Key features:
- Multi-frame temporal processing
- Deformable convolution for alignment
- Better handling of motion and occlusions
- Reduced flickering artifacts

## Acceptance Criteria

1. **Model Implementation**
   - [ ] EDVRModel class implementing BaseModel
   - [ ] Support for EDVR_M (medium) and EDVR_L (large) variants
   - [ ] Temporal window configuration (5-7 frames)
   - [ ] Proper frame buffering and alignment

2. **Temporal Processing**
   - [ ] Override predict_temporal() method
   - [ ] Efficient frame buffer management
   - [ ] Handle video edges (start/end frames)
   - [ ] Memory-efficient implementation

3. **Weight Management**
   - [ ] Download from BasicSR or official sources
   - [ ] Support for pretrained weights
   - [ ] Model caching

4. **Testing**
   - [ ] Unit tests for single-frame fallback
   - [ ] Integration tests with frame sequences
   - [ ] Quality comparison with frame-by-frame
   - [ ] Temporal consistency metrics

## Technical Details

```python
from basicsr.archs.edvr_arch import EDVR
from video_upscaler.models.base import BaseModel, ModelMetadata
from collections import deque
import torch

class EDVRModel(BaseModel):
    """EDVR model for video super-resolution."""
    
    def __init__(self, config: ModelConfig, num_frames: int = 5):
        super().__init__(config)
        self.num_frames = num_frames
        self._model = None
        self._frame_buffer = deque(maxlen=num_frames)
    
    def load_weights(self, weights_path, download=False):
        # Load EDVR architecture
        self._model = EDVR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=128,
            num_frame=self.num_frames,
            deformable_groups=8,
            num_extract_block=5,
            num_reconstruct_block=40,
            center_frame_idx=self.num_frames // 2,
            hr_in=False,
            with_predeblur=False,
            with_tsa=True
        )
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=self.config.device)
        self._model.load_state_dict(checkpoint['params'])
        self._model.to(self.config.device)
        self._model.eval()
    
    def predict_temporal(self, frames: List, **kwargs):
        """Process frames with temporal consistency."""
        results = []
        
        for i, frame in enumerate(frames):
            # Build temporal window
            window_frames = self._get_temporal_window(frames, i)
            
            # Process with EDVR
            result = self._process_window(window_frames)
            results.append(result)
        
        return results
    
    def _get_temporal_window(self, frames, center_idx):
        """Get frames for temporal window around center frame."""
        half_window = self.num_frames // 2
        start = max(0, center_idx - half_window)
        end = min(len(frames), center_idx + half_window + 1)
        
        window = frames[start:end]
        
        # Pad if at edges
        while len(window) < self.num_frames:
            if start == 0:
                window.insert(0, window[0])  # Repeat first frame
            else:
                window.append(window[-1])  # Repeat last frame
        
        return window
    
    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="EDVR",
            version="1.0.0",
            scales=[4],
            description="Enhanced Deformable Video Restoration",
            author="xinntao",
            paper_url="https://arxiv.org/abs/1905.02716",
            implementation_url="https://github.com/xinntao/EDVR",
            supports_temporal=True,
            supports_batch=True,
            memory_gb_estimate=6.0  # Requires more memory for multi-frame
        )
```

## Dependencies

- Epic 01, Issue 01: Model Abstraction Layer
- Epic 01, Issue 02: Video Processing Pipeline (for temporal support)

## Estimated Effort

**Size:** Large (L)
**Time:** ~32 hours (4 days)

## Priority

**P1** - Important for video-specific quality improvements

## Labels

- `epic:model-integration`
- `type:enhancement`
- `area:models`
- `priority:p1`
- `size:large`
