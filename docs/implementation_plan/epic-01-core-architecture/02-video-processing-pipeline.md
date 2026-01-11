# Video Processing Pipeline

## Title
Refactor video processing into modular, reusable pipeline components

## Description

The current video processing logic in `main.py` is monolithic and tightly couples video I/O with frame processing. To support advanced features like audio handling, codec selection, temporal consistency, and batch processing, we need to refactor this into a modular pipeline architecture.

This refactoring will:
- Separate video input/output from frame processing
- Create reusable components for each stage of processing
- Support multiple video backends (PyAV, ffmpeg-python, decord)
- Enable streaming processing for memory efficiency
- Provide hooks for custom processing stages

### Why This Matters
A modular pipeline makes it easy to add new features without modifying core logic. It also enables testing individual components in isolation and supports different processing strategies (streaming vs. batch, single-pass vs. multi-pass).

## Acceptance Criteria

1. **Pipeline Architecture Defined**
   - [ ] Pipeline stages clearly separated: Read → Process → Write
   - [ ] Each stage is independently testable
   - [ ] Stages can be composed and reordered
   - [ ] Pipeline supports both streaming and batch modes

2. **Video Reader Component**
   - [ ] Abstract VideoReader interface created
   - [ ] PyAVReader implementation
   - [ ] Support for frame extraction with metadata (timestamp, frame number)
   - [ ] Efficient seeking and random access
   - [ ] Stream selection (video/audio/subtitle tracks)

3. **Video Writer Component**
   - [ ] Abstract VideoWriter interface created
   - [ ] PyAVWriter implementation
   - [ ] Support for codec/quality configuration
   - [ ] Audio track writing
   - [ ] Metadata preservation

4. **Frame Processor Component**
   - [ ] Abstract FrameProcessor interface
   - [ ] Integration with model abstraction layer
   - [ ] Support for pre/post-processing hooks
   - [ ] Batch processing support

5. **Pipeline Orchestrator**
   - [ ] Main Pipeline class that coordinates all stages
   - [ ] Progress tracking and callbacks
   - [ ] Error handling and recovery
   - [ ] Resource management (memory, GPU)

6. **Testing**
   - [ ] Unit tests for each component
   - [ ] Integration tests for full pipeline
   - [ ] Test with various video formats and codecs
   - [ ] Performance benchmarks

## Technical Details

### Proposed Architecture

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from pathlib import Path
import av
import numpy as np
from PIL import Image

@dataclass
class VideoMetadata:
    """Metadata about a video file."""
    width: int
    height: int
    fps: float
    duration: float  # seconds
    frame_count: int
    codec: str
    pixel_format: str
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    file_size: int = 0
    
    def __repr__(self) -> str:
        return (
            f"VideoMetadata({self.width}x{self.height}, "
            f"{self.fps:.2f}fps, {self.frame_count} frames, "
            f"codec={self.codec}, audio={self.has_audio})"
        )

@dataclass
class Frame:
    """A video frame with metadata."""
    image: np.ndarray  # HWC format, RGB, uint8
    index: int
    timestamp: float  # seconds
    pts: int  # presentation timestamp
    
    def to_pil(self) -> Image.Image:
        """Convert to PIL Image."""
        return Image.fromarray(self.image)
    
    @classmethod
    def from_pil(cls, image: Image.Image, index: int, timestamp: float, pts: int):
        """Create from PIL Image."""
        return cls(
            image=np.array(image),
            index=index,
            timestamp=timestamp,
            pts=pts
        )

class VideoReader(ABC):
    """Abstract interface for reading video files."""
    
    def __init__(self, path: Path):
        self.path = path
        self._metadata: Optional[VideoMetadata] = None
    
    @abstractmethod
    def open(self) -> None:
        """Open the video file."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the video file and release resources."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> VideoMetadata:
        """Get video metadata."""
        pass
    
    @abstractmethod
    def read_frames(self) -> Iterator[Frame]:
        """Iterate over all frames sequentially."""
        pass
    
    @abstractmethod
    def read_frame(self, index: int) -> Frame:
        """Read a specific frame by index."""
        pass
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __len__(self) -> int:
        """Get number of frames."""
        return self.get_metadata().frame_count

class PyAVReader(VideoReader):
    """PyAV-based video reader."""
    
    def __init__(self, path: Path):
        super().__init__(path)
        self._container = None
        self._video_stream = None
    
    def open(self) -> None:
        self._container = av.open(str(self.path))
        self._video_stream = self._container.streams.video[0]
        
        # Extract metadata
        self._metadata = VideoMetadata(
            width=self._video_stream.width,
            height=self._video_stream.height,
            fps=float(self._video_stream.average_rate),
            duration=float(self._video_stream.duration * self._video_stream.time_base),
            frame_count=self._video_stream.frames,
            codec=self._video_stream.codec_context.name,
            pixel_format=self._video_stream.codec_context.pix_fmt,
            has_audio=len(self._container.streams.audio) > 0,
            audio_codec=self._container.streams.audio[0].codec_context.name if len(self._container.streams.audio) > 0 else None,
            audio_sample_rate=self._container.streams.audio[0].sample_rate if len(self._container.streams.audio) > 0 else None,
            file_size=self.path.stat().st_size
        )
    
    def close(self) -> None:
        if self._container:
            self._container.close()
            self._container = None
            self._video_stream = None
    
    def get_metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("Video not opened. Call open() first.")
        return self._metadata
    
    def read_frames(self) -> Iterator[Frame]:
        if self._container is None:
            raise RuntimeError("Video not opened. Call open() first.")
        
        for frame_idx, av_frame in enumerate(self._container.decode(self._video_stream)):
            yield Frame(
                image=av_frame.to_ndarray(format='rgb24'),
                index=frame_idx,
                timestamp=float(av_frame.pts * self._video_stream.time_base),
                pts=av_frame.pts
            )
    
    def read_frame(self, index: int) -> Frame:
        # Seek to frame (simplified - production code needs better seeking)
        if self._container is None:
            raise RuntimeError("Video not opened. Call open() first.")
        
        # Reset and seek
        self._container.seek(0)
        for i, frame in enumerate(self.read_frames()):
            if i == index:
                return frame
        raise ValueError(f"Frame {index} not found")

class VideoWriter(ABC):
    """Abstract interface for writing video files."""
    
    def __init__(self, path: Path, metadata: VideoMetadata, codec: str = 'libx264'):
        self.path = path
        self.metadata = metadata
        self.codec = codec
    
    @abstractmethod
    def open(self) -> None:
        """Open the video file for writing."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the video file and finalize."""
        pass
    
    @abstractmethod
    def write_frame(self, frame: Frame) -> None:
        """Write a single frame."""
        pass
    
    @abstractmethod
    def write_frames(self, frames: Iterator[Frame]) -> None:
        """Write multiple frames."""
        pass
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class PyAVWriter(VideoWriter):
    """PyAV-based video writer."""
    
    def __init__(
        self,
        path: Path,
        metadata: VideoMetadata,
        codec: str = 'libx264',
        pixel_format: str = 'yuv420p',
        crf: int = 18,
        preset: str = 'medium'
    ):
        super().__init__(path, metadata, codec)
        self.pixel_format = pixel_format
        self.crf = crf
        self.preset = preset
        self._container = None
        self._video_stream = None
    
    def open(self) -> None:
        self._container = av.open(str(self.path), 'w')
        
        # Create video stream
        self._video_stream = self._container.add_stream(
            self.codec,
            rate=self.metadata.fps
        )
        self._video_stream.width = self.metadata.width
        self._video_stream.height = self.metadata.height
        self._video_stream.pix_fmt = self.pixel_format
        
        # Set codec options
        self._video_stream.options = {
            'crf': str(self.crf),
            'preset': self.preset
        }
    
    def close(self) -> None:
        if self._container:
            # Flush encoder
            for packet in self._video_stream.encode():
                self._container.mux(packet)
            
            self._container.close()
            self._container = None
            self._video_stream = None
    
    def write_frame(self, frame: Frame) -> None:
        if self._container is None:
            raise RuntimeError("Video not opened. Call open() first.")
        
        # Convert numpy array to PyAV frame
        av_frame = av.VideoFrame.from_ndarray(frame.image, format='rgb24')
        av_frame.pts = frame.pts
        
        # Encode and mux
        for packet in self._video_stream.encode(av_frame):
            self._container.mux(packet)
    
    def write_frames(self, frames: Iterator[Frame]) -> None:
        for frame in frames:
            self.write_frame(frame)

class FrameProcessor(ABC):
    """Abstract interface for processing frames."""
    
    @abstractmethod
    def process(self, frame: Frame) -> Frame:
        """Process a single frame."""
        pass
    
    def process_batch(self, frames: List[Frame]) -> List[Frame]:
        """Process a batch of frames (default: sequential)."""
        return [self.process(frame) for frame in frames]

class ModelFrameProcessor(FrameProcessor):
    """Frame processor that uses a super-resolution model."""
    
    def __init__(self, model: 'BaseModel'):
        self.model = model
    
    def process(self, frame: Frame) -> Frame:
        # Upscale the image
        upscaled_image = self.model.predict(frame.image)
        
        # Return new frame with same metadata
        return Frame(
            image=np.array(upscaled_image) if isinstance(upscaled_image, Image.Image) else upscaled_image,
            index=frame.index,
            timestamp=frame.timestamp,
            pts=frame.pts
        )
    
    def process_batch(self, frames: List[Frame]) -> List[Frame]:
        if self.model.supports_batch():
            # Use model's batch processing
            images = [frame.image for frame in frames]
            upscaled_images = self.model.predict_batch(images)
            
            return [
                Frame(
                    image=np.array(img) if isinstance(img, Image.Image) else img,
                    index=frame.index,
                    timestamp=frame.timestamp,
                    pts=frame.pts
                )
                for img, frame in zip(upscaled_images, frames)
            ]
        else:
            return super().process_batch(frames)

class VideoPipeline:
    """Main pipeline that orchestrates video processing."""
    
    def __init__(
        self,
        reader: VideoReader,
        processor: FrameProcessor,
        writer: VideoWriter,
        batch_size: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        self.reader = reader
        self.processor = processor
        self.writer = writer
        self.batch_size = batch_size
        self.progress_callback = progress_callback
    
    def run(self) -> None:
        """Execute the pipeline."""
        with self.reader, self.writer:
            metadata = self.reader.get_metadata()
            total_frames = metadata.frame_count
            
            if self.batch_size == 1:
                # Streaming mode - process frame by frame
                for frame_idx, frame in enumerate(self.reader.read_frames()):
                    processed_frame = self.processor.process(frame)
                    self.writer.write_frame(processed_frame)
                    
                    if self.progress_callback:
                        self.progress_callback(frame_idx + 1, total_frames)
            else:
                # Batch mode - process multiple frames at once
                batch = []
                frame_idx = 0
                
                for frame in self.reader.read_frames():
                    batch.append(frame)
                    
                    if len(batch) >= self.batch_size:
                        processed_frames = self.processor.process_batch(batch)
                        for pf in processed_frames:
                            self.writer.write_frame(pf)
                        
                        frame_idx += len(batch)
                        if self.progress_callback:
                            self.progress_callback(frame_idx, total_frames)
                        
                        batch = []
                
                # Process remaining frames
                if batch:
                    processed_frames = self.processor.process_batch(batch)
                    for pf in processed_frames:
                        self.writer.write_frame(pf)
                    
                    frame_idx += len(batch)
                    if self.progress_callback:
                        self.progress_callback(frame_idx, total_frames)
    
    @classmethod
    def create(
        cls,
        input_path: Path,
        output_path: Path,
        model: 'BaseModel',
        codec: str = 'libx264',
        batch_size: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> 'VideoPipeline':
        """Factory method to create a pipeline."""
        # Create reader
        reader = PyAVReader(input_path)
        reader.open()
        input_metadata = reader.get_metadata()
        reader.close()
        
        # Create output metadata (scaled dimensions)
        scale = model.get_scale()
        output_metadata = VideoMetadata(
            width=input_metadata.width * scale,
            height=input_metadata.height * scale,
            fps=input_metadata.fps,
            duration=input_metadata.duration,
            frame_count=input_metadata.frame_count,
            codec=codec,
            pixel_format='yuv420p',
            has_audio=input_metadata.has_audio,
            audio_codec=input_metadata.audio_codec,
            audio_sample_rate=input_metadata.audio_sample_rate
        )
        
        # Create components
        reader = PyAVReader(input_path)
        processor = ModelFrameProcessor(model)
        writer = PyAVWriter(output_path, output_metadata, codec=codec)
        
        return cls(reader, processor, writer, batch_size, progress_callback)
```

### File Structure

```
src/video_upscaler/
├── pipeline/
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── reader.py            # VideoReader implementations
│   ├── writer.py            # VideoWriter implementations
│   ├── processor.py         # FrameProcessor implementations
│   ├── pipeline.py          # VideoPipeline
│   └── types.py             # VideoMetadata, Frame dataclasses
└── ...
```

### Usage Example

```python
from pathlib import Path
from video_upscaler.models import ModelFactory, ModelConfig
from video_upscaler.pipeline import VideoPipeline
from tqdm import tqdm

# Create model
config = ModelConfig(device=torch.device('cuda'), scale=4)
model = ModelFactory.create('realesrgan', config)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

# Create progress bar
pbar = tqdm(total=100, desc='Processing')

def progress_callback(current, total):
    pbar.total = total
    pbar.n = current
    pbar.refresh()

# Create and run pipeline
pipeline = VideoPipeline.create(
    input_path=Path('input.mp4'),
    output_path=Path('output.mp4'),
    model=model,
    codec='libx264',
    batch_size=4,
    progress_callback=progress_callback
)

pipeline.run()
pbar.close()
```

## Dependencies

**Before starting this issue:**
- Epic 01, Issue 01: Model Abstraction Layer (for ModelFrameProcessor integration)

**Blocks:**
- Epic 03, Issue 01: Audio Stream Handling
- Epic 03, Issue 02: Codec Selection
- Epic 03, Issue 03: Temporal Consistency
- Epic 03, Issue 04: Batch Frame Processing

## Estimated Effort

**Size:** Large (L)

**Breakdown:**
- Design and review: 8 hours
- Implementation: 20 hours
- Testing: 8 hours
- Documentation: 4 hours
- Integration with existing code: 4 hours

**Total:** ~44 hours (~1 week)

## Priority

**P0** - Critical for modular architecture

## Labels

- `epic:core-architecture`
- `type:refactor`
- `area:pipeline`
- `priority:p0`
- `size:large`

## Implementation Notes

### Migration Strategy
1. Implement new pipeline components alongside existing code
2. Add feature flag to use new pipeline
3. Test thoroughly with various videos
4. Switch to new pipeline as default
5. Remove old implementation

### Testing Strategy
- Test each component independently
- Test with various video formats (MP4, AVI, MOV, MKV)
- Test with different codecs
- Verify frame accuracy (output matches expected)
- Performance benchmarks

### Future Enhancements
- Support for other backends (ffmpeg-python, decord)
- Memory-mapped file support for large videos
- GPU-accelerated video decoding
- Support for image sequences as input
