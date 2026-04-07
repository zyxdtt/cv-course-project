# Image Perception Enhancement System

A computer vision system for enhancing images degraded by adverse visual conditions, including haze, low-light, and uneven illumination. The system supports both **static image processing** and **real-time camera enhancement**.

## Features

- **Three Degradation Types**
  - Haze removal using Dark Channel Prior
  - Low-light enhancement with adaptive gamma correction and CLAHE
  - Uneven illumination correction

- **Two Operating Modes**
  - Static image enhancement (offline single image)
  - Real-time video enhancement (live camera stream)

- **Interactive Controls**
  - Save enhanced frames (`s` key)
  - Quit real-time mode (`q` key)

## System Architecture

The system adopts a modular design with separate processing pipelines for static images and real-time video, sharing core enhancement algorithms.

## Algorithm Overview

### Haze Removal - Dark Channel Prior

Based on the atmospheric scattering model:
```
I(x) = J(x)·t(x) + A·(1 - t(x))
```

**Steps:**
1. Compute dark channel
2. Estimate atmospheric light A
3. Estimate transmission map
4. Refine using guided filter
5. Recover haze-free image

### Low-light Enhancement

Combines adaptive gamma correction with CLAHE:

- **Adaptive Gamma:** `γ = 0.5 + (1 - V̄) × 0.5`
- **CLAHE:** Contrast Limited Adaptive Histogram Equalization with 8×8 tiles
- **Saturation Compensation:** Prevents color desaturation after brightening

### Uneven Illumination Correction

1. Convert RGB → Lab color space
2. Apply CLAHE to L channel
3. Global brightness gain: `L_new = L_enhanced × strength`
4. Convert back to RGB

## Function Interfaces

### Dehaze
```cpp
Mat dehazeImage(const Mat& src, 
                int winSize = 15, 
                double omega = 0.85, 
                double t0 = 0.15);
```

### Low-light Enhancement
```cpp
Mat lowlightEnhance(const Mat& src, 
                    double claheClipLimit = 3.5, 
                    double satCompensation = 1.15, 
                    bool denoise = false);
```

### Uneven Illumination Correction
```cpp
Mat unevenIlluminationEnhance(const Mat& src, 
                              double strength = 1.5);
```

## Parameter Guidelines

### Dehaze Parameters

| Parameter | Meaning | Range | Default |
|-----------|---------|-------|---------|
| winSize | Dark channel window | 11-31 (odd) | 15 |
| omega | Dehazing strength | 0.70-0.95 | 0.85 |
| t0 | Transmission lower bound | 0.10-0.25 | 0.15 |

**Recommended by haze level:**
- Light haze: `omega=0.80, t0=0.12`
- Moderate haze: `omega=0.85, t0=0.15`
- Heavy haze: `omega=0.92, t0=0.18`
- Night hazy scene: `omega=0.80, t0=0.22`

### Low-light Parameters

| Parameter | Meaning | Range | Default |
|-----------|---------|-------|---------|
| claheClipLimit | CLAHE clip limit | 2.0-5.0 | 3.5 |
| satCompensation | Saturation boost | 1.0-1.3 | 1.15 |
| denoise | Bilateral filter | true/false | false |

**Recommended by brightness level:**
- Slightly dark: `clip=2.5, sat=1.1, denoise=false`
- Moderately dark: `clip=3.5, sat=1.15, denoise=false`
- Very dark: `clip=4.5, sat=1.2, denoise=true`
- Noisy dark image: `clip=3.0, sat=1.1, denoise=true`

### Uneven Illumination Parameters

| Parameter | Meaning | Range | Default |
|-----------|---------|-------|---------|
| strength | Brightness gain | 1.2-2.0 | 1.5 |

**Recommended by severity:**
- Mild unevenness: `strength=1.3`
- Moderate unevenness: `strength=1.5`
- Severe unevenness: `strength=1.8`

## Usage

### Static Image Processing

1. Run the program and select static image mode
2. Enter the image file path
3. Choose enhancement type (1: Dehaze, 2: Low-light, 3: Uneven)
4. View original vs enhanced images
5. Result is automatically saved as `result.jpg`

### Real-time Camera Enhancement

1. Run the program and select real-time mode
2. Choose enhancement type (1/2/3)
3. Live camera feed opens with side-by-side comparison
4. Controls:
   - Press `q` to quit
   - Press `s` to save current enhanced frame

## Performance Expectations

| Algorithm | Expected Speed |
|-----------|----------------|
| Dehaze (with guided filter) | ~100-200 ms/frame |
| Low-light enhancement | ~20-30 ms/frame |
| Uneven illumination | ~15-25 ms/frame |

**Note:** Real-time processing target is <33ms per frame (30fps). For dehaze, consider reducing resolution for better real-time performance.

## Expected Outcomes

- Successful implementation of three image enhancement algorithms
- Static image processing meets subjective quality requirements
- Real-time camera enhancement achieves 15-30fps
- Optimal parameters determined for each scenario through experiments

## Open Issues

- Guided filter in dehaze is computationally heavy; real-time optimization needed
- Optimal parameters require extensive experimental validation
- Mixed degradation (e.g., haze + low-light) handling strategy to be determined

## Dependencies

- OpenCV (core, imgproc, highgui, videoio)
- C++11 or later


这个 README 涵盖了系统功能、算法原理、函数接口、参数配置、使用方法和性能预期，可以直接放在您的项目仓库中。需要调整或补充任何内容请告诉我。
