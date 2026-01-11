# Codec Selection

## Title
Support multiple output codecs and quality settings

## Description

Replace hardcoded MPEG4 with flexible codec selection:
- H.264/H.265/VP9/AV1 support
- CRF-based quality control
- Preset selection (speed vs quality)
- Two-pass encoding option
- Hardware encoding support (NVENC, QSV, VideoToolbox)

## Acceptance Criteria

1. **Codec Support**
   - [ ] H.264 (libx264)
   - [ ] H.265 (libx265)
   - [ ] VP9
   - [ ] AV1 (libaom-av1, SVT-AV1)
   - [ ] ProRes (for professional workflows)

2. **Quality Control**
   - [ ] CRF-based quality (0-51)
   - [ ] Explicit bitrate control
   - [ ] Preset selection
   - [ ] Two-pass encoding

3. **Hardware Encoding**
   - [ ] NVENC (NVIDIA)
   - [ ] QuickSync (Intel)
   - [ ] VideoToolbox (Apple)
   - [ ] Auto-detection and fallback

4. **Configuration**
   - [ ] CLI options for codec selection
   - [ ] Config file support
   - [ ] Codec presets (fast, balanced, quality)

## Dependencies

- Epic 01, Issue 02: Video Processing Pipeline
- Epic 01, Issue 03: Configuration System

## Estimated Effort

**Size:** Medium (M)  
**Time:** ~20 hours (2.5 days)

## Priority

**P1** - Important for output quality control

## Labels

- `epic:video-pipeline`
- `type:enhancement`
- `priority:p1`
- `size:medium`
