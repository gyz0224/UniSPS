# Task Plan: Supervised Evaluation Metrics

## Goal
Add NIQE to supervised low-light test evaluation and enable LPIPS for every paired/supervised dehazing evaluation preset, with tests and documentation kept consistent.

## Current Phase
Complete

## Phases

### Phase 1: Requirements & Discovery
- [x] Locate low-light and dehazing metric entry points
- [x] Identify supervised dataset/preset routing
- [x] Preserve existing unrelated worktree changes
- **Status:** complete

### Phase 2: Design & Regression Tests
- [x] Define one reusable NIQE evaluator and dependency/error behavior
- [x] Add regression coverage for low-light NIQE output and all paired dehazing LPIPS presets
- **Status:** complete

### Phase 3: Implementation
- [x] Add NIQE to supervised low-light directory evaluation, CLI output, and JSON
- [x] Enable LPIPS whenever a dehazing evaluation has a clean reference
- [x] Update user-facing metric documentation
- **Status:** complete

### Phase 4: Verification
- [x] Run focused unit tests
- [x] Run broader relevant test suite and compile checks
- [x] Review diff for unrelated changes
- **Status:** complete

### Phase 5: Delivery
- [x] Summarize behavior and verification
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Treat supervised dehazing as every preset with a non-null clean-reference directory, including a reference supplied for a real preset | `measure_dehaze.py` already uses reference presence to distinguish full-reference from FADE evaluation |
| Keep existing `eval_lowlight.py` zero-sanitization diff untouched | It predates this request and belongs to existing user work |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| Initial combined source dump was truncated | Re-read targeted line ranges and use focused searches |
| First planning-file patch used delete/add operations on the same paths and failed validation | Switch to targeted update patches; no project source was affected |
| Focused tests under system `python` could not import project dependencies (`torch`, `cv2`) | Use the repository's existing ICLR conda environment for test execution |
