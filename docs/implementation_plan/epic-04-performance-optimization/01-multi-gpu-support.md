# Multi-GPU Support

## Title
Distribute workload across multiple GPUs

## Description

Implement multi-GPU parallelization:
- Distribute frames across GPUs
- Load balancing
- Near-linear speedup with multiple GPUs
- Support for different GPU models

## Acceptance Criteria

1. [ ] Detect and enumerate all GPUs
2. [ ] Distribute frame batches across GPUs
3. [ ] Load balancing algorithm
4. [ ] Result aggregation
5. [ ] Performance benchmarks (2GPU, 4GPU)

## Dependencies

- Epic 01, Issue 04: Device Management
- Epic 03, Issue 04: Batch Frame Processing

## Estimated Effort

**Size:** Large (L)  
**Time:** ~28 hours

## Priority

**P2**

## Labels

- `epic:performance`
- `type:enhancement`
- `priority:p2`
- `size:large`
