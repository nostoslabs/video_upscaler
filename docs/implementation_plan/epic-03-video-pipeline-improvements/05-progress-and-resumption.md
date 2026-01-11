# Progress and Resumption

## Title
Add checkpoint-based progress saving and job resumption

## Description

Enable long video processing to be interrupted and resumed:
- Checkpoint saving (every N frames)
- Resume from checkpoint
- Progress state persistence
- Crash recovery

## Acceptance Criteria

1. **Checkpoint System**
   - [ ] Save checkpoints periodically
   - [ ] Checkpoint format (processed frames, metadata)
   - [ ] Configurable checkpoint interval

2. **Resumption**
   - [ ] Detect existing checkpoints
   - [ ] Resume from last checkpoint
   - [ ] Validate checkpoint integrity
   - [ ] Merge checkpoint output with final video

3. **Progress Tracking**
   - [ ] Persistent progress state
   - [ ] ETA calculation
   - [ ] Statistics (frames/sec, time remaining)

4. **Error Handling**
   - [ ] Graceful shutdown on interrupt
   - [ ] Checkpoint cleanup on completion
   - [ ] Recovery from corrupted checkpoints

## Dependencies

- Epic 01, Issue 02: Video Processing Pipeline

## Estimated Effort

**Size:** Medium (M)  
**Time:** ~20 hours (2.5 days)

## Priority

**P2** - Nice to have for long videos

## Labels

- `epic:video-pipeline`
- `type:enhancement`
- `priority:p2`
- `size:medium`
