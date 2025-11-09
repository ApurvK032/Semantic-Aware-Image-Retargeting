# Semantic-Aware Image Retargeting - Project Proposal

## Executive Summary

This project improves intelligent image retargeting by combining classical seam carving with modern semantic understanding. Rather than relying solely on gradient-based energy functions, we integrate Segment Anything 2 (for object-level understanding) and Depth Anything V2 (for spatial reasoning) to create a robust, semantic-aware energy function that better preserves important content and reduces visual artifacts.

## Problem Statement

Traditional seam carving has fundamental limitations:

1. **Gradient-Only Energy**: Treats all regions equally, often cutting through important objects
2. **Object Distortion**: Lacks semantic understanding of what's important
3. **Artifact Creation**: Causes unnatural warping and discontinuities
4. **No Foreground-Background Distinction**: Can't differentiate between salient and background regions
5. **Limited Context**: No understanding of object relationships or spatial hierarchy

Real-world challenges:
- Portraits get facial distortions
- Multiple objects conflict in importance ranking
- Lack of global scene understanding
- No depth or spatial reasoning

## Project Objectives

1. **Semantic Integration**: Leverage SAM 2 for fine-grained object understanding
2. **Spatial Awareness**: Use Depth Anything V2 for depth-informed decisions
3. **Robust Energy**: Develop multi-modal energy function combining gradients, semantics, and depth
4. **Quality Improvement**: Achieve >85% object preservation vs. ~62% baseline
5. **Artifact Reduction**: Reduce visual artifacts by >70% compared to traditional seam carving
6. **Comprehensive Evaluation**: Validate on RetargetMe benchmark (80 diverse images)

## Technical Approach

### Phase 1: Model Integration & Setup
- Integrate SAM 2 for semantic segmentation
- Integrate Depth Anything V2 for depth estimation
- Set up PyTorch inference pipelines
- Establish baseline comparisons with traditional seam carving
- **Timeline**: Week 1

### Phase 2: Energy Function Design
- Analyze gradient-based baseline energy
- Design semantic penalty scores (object boundaries)
- Compute depth discontinuity maps
- Implement edge preservation mechanisms
- Develop weighting scheme for multi-modal fusion
- **Timeline**: Week 2

**Energy Function:**
```
E(x,y) = w_grad × ∇I(x,y) 
       + w_sem × Semantic_Penalty(x,y)
       + w_depth × Depth_Discontinuity(x,y)
       + w_edge × Edge_Score(x,y)

where w_grad + w_sem + w_depth + w_edge = 1
```

### Phase 3: Seam Carving Implementation
- Implement semantic-aware seam carving algorithm
- Dynamic programming with energy minimization
- Seam removal with artifact mitigation
- Iterative resizing until target dimensions reached
- **Timeline**: Week 2-3

**Algorithm:**
```
1. Compute semantic-aware energy map
2. Find minimum-cost vertical/horizontal seams (DP)
3. Remove seams with local blending
4. Update affected regions
5. Repeat until target size reached
```

### Phase 4: Optimization & Acceleration
- Profile inference time per component
- Implement batch processing
- Consider GPU optimization (TorchScript, ONNX)
- Optional: Mobile deployment considerations
- **Timeline**: Week 3

### Phase 5: Comprehensive Evaluation
- Quantitative metrics on RetargetMe dataset (80 images)
- Object preservation score (semantic overlap)
- Artifact detection and quantification
- Perceptual quality metrics (LPIPS, SSIM)
- **Timeline**: Week 3-4

### Phase 6: Ablation & Analysis
- Study impact of each energy component
- Weight sensitivity analysis
- Comparison with baseline and other methods
- Failure case analysis
- **Timeline**: Week 4

### Phase 7: Documentation & Presentation
- Comprehensive documentation
- Qualitative/quantitative results analysis
- User study (if time permits)
- Final presentation materials
- **Timeline**: Week 4

## System Architecture

```
INPUT IMAGE
    ↓
┌─────────────────────────────────────┐
│  PARALLEL MODEL INFERENCE           │
├─────────────────────────────────────┤
│  SAM 2 Segmentation  │ Depth Estimation
│  (Object Masks)      │ (Depth Map)
└─────────────────────────────────────┘
    ↓                   ↓
Semantic Masks      Depth Map
    ↓                   ↓
    └─────────────────┬────────────────┘
                ↓
    ┌─────────────────────────────┐
    │ ENERGY COMPUTATION          │
    │ • Gradient Map              │
    │ • Semantic Penalties        │
    │ • Depth Discontinuities     │
    │ • Edge Scores               │
    │ • Weighted Combination      │
    └─────────────────────────────┘
                ↓
        Energy Map E(x,y)
                ↓
    ┌─────────────────────────────┐
    │ SEAM CARVING ENGINE         │
    │ • DP seam finding           │
    │ • Seam removal              │
    │ • Artifact mitigation       │
    └─────────────────────────────┘
                ↓
        RETARGETED IMAGE
```

## Expected Results

### Quantitative Targets

**Object Preservation:**
- Target: >85% (vs. baseline 62%)
- Metric: F1-score of object overlap before/after

**Artifact Reduction:**
- Target: >70% fewer artifacts
- Metric: Automatic detection + manual inspection

**Processing Speed:**
- Target: <500ms per image (1024×768)
- Components: SAM 2 (80ms) + Depth (70ms) + Seam carving (200-300ms)

**RetargetMe Dataset:**
- 80 diverse images (portraits, landscapes, multi-object)
- Comprehensive evaluation across categories
- Statistical significance testing

### Qualitative Results

- **Portraits**: Better facial feature preservation
- **Landscapes**: Maintains horizon and focal points
- **Multi-object**: Intelligent prioritization
- **Complex scenes**: Fewer visual discontinuities
- **Overall**: Superior visual quality vs. baseline

## Evaluation Methodology

### Quantitative Metrics

1. **Semantic Object Preservation**
   - Overlap of object masks before/after
   - Per-class accuracy
   - Overall F1-score

2. **Artifact Quantification**
   - Edge continuity score
   - Perceptual discontinuity metrics
   - Manual artifact labeling

3. **Perceptual Quality**
   - LPIPS (learned perceptual metric)
   - SSIM (structural similarity)
   - FID (Fréchet Inception Distance)

### Qualitative Evaluation

1. **User Study** (if time permits)
   - Participants rate image quality (1-5 scale)
   - Compare semantic-aware vs. baseline
   - Collect artifact feedback

2. **Visual Inspection**
   - Before/after comparisons
   - Artifact identification
   - Edge preservation analysis

### Dataset

**RetargetMe Benchmark:**
- 80 high-quality test images
- Categories: portraits, landscapes, objects, scenes
- Multiple aspect ratio targets
- Standard evaluation protocol

## Deliverables

| Week | Deliverable | Details |
|------|-------------|---------|
| 1 | Model integration & baselines | SAM 2, Depth setup, benchmark |
| 2 | Energy function design | Multi-modal energy, weighting |
| 2-3 | Seam carving implementation | Full algorithm with optimization |
| 3 | Optimization & profiling | GPU acceleration, speed benchmarks |
| 3-4 | Quantitative evaluation | RetargetMe results, metrics |
| 4 | Ablation study | Component analysis, weights |
| 4 | Documentation & presentation | Code, results, write-up |

## Key Innovation Points

1. **Multi-Modal Energy**: First combination of gradient, semantic, depth, and edge cues
2. **Zero-Shot Segmentation**: Leverage SAM 2 without fine-tuning
3. **Depth-Aware Decisions**: Protect foreground using modern depth estimation
4. **Comprehensive Evaluation**: Full analysis on diverse benchmark dataset

## Success Criteria

✅ Object preservation >85% on RetargetMe  
✅ Artifact count reduced by >70%  
✅ Processing time <500ms per image  
✅ Qualitative results show clear improvement  
✅ Reproducible code with documentation  
✅ Comprehensive evaluation report  

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Model inference too slow | Implement batch processing, GPU optimization |
| Energy function ineffective | Ablation study, iterative tuning |
| Artifact still present | Combine with post-processing (e.g., blending) |
| Limited improvement over baseline | Explore alternative semantic models, depth variants |
| Evaluation challenges | Use established metrics, user studies |

## References

1. Avidan & Shamir (2007): Seam Carving for Content-Aware Image Resizing
2. Kirillov et al. (2024): Segment Anything 2
3. Yang et al. (2024): Depth Anything V2
4. Ye et al. (2024): PruneRepaint (NeurIPS 2024)
5. Rubinstein et al. (2010): RetargetMe Benchmark

## Timeline

**Total Duration**: 4 weeks (active development)

```
Week 1: Setup & Integration
Week 2: Energy Design & Algorithm
Week 3: Optimization & Evaluation
Week 4: Ablation & Final Polish
```

---

**Project Status**: Planning Phase  
**Last Updated**: November 2025  
**Team**: Apurv Kushwaha, Ayaan Mohammed, Srinivas Kantha Reddy
