# MISO — Memory-Integrated Search and Optimization
### CNN Inference Scheduling for the TI MSP430F5529 (8 KB SRAM)

> **Associated paper:** Aaditya Deokar, *"Memory Optimization for TinyML and IoT Devices"*, BITS Pilani Dubai Campus, Design Project 1, 2026.
> Built on and improving: Banerjee et al., *"Memory-aware Efficient Deep Learning Mechanism for IoT Devices"*, IEEE ASAP 2021.

---

## Overview

MISO is a hardware-aware CNN scheduling framework that solves a fundamental problem in TinyML: how to run a convolutional neural network on a microcontroller with only **8 KB of working memory**.

On the TI MSP430F5529, the primary bottleneck is not arithmetic — it is data movement. Every byte transferred between Flash storage and SRAM costs measurable energy. MISO minimises that cost through a five-stage deterministic pipeline:

```
Raw input
  → Normalization          (map sensor data to int8 [-128, 127])
  → Macro-Tiling           (outer C_out strip decomposition, O(log C_out))
      └─ Inner Tiling      (exhaustive (T_c, T_r, T_f) search)
           └─ Ping-Pong    (DMA double-buffer scheduling)
  → Execution plan         (energy, latency, SRAM breakdown per layer)
```

### Key results across 13 benchmark layers (MNIST, HAR, TrafficSign)

| Metric | Banerjee Hybrid (fp32) | MISO (int8) | Improvement |
|--------|------------------------|-------------|-------------|
| Energy | 6,660 mJ | 535 mJ | **12.4×** lower |
| Latency | 549,143 ms | 27,424 ms | **20.0×** lower |
| SRAM feasibility | OaaT fails for all Conv layers | All 13 layers fit in 8 KB | **100%** feasible |

---

## Hardware Target

| Parameter | Value |
|-----------|-------|
| MCU | TI MSP430F5529 |
| SRAM | 8,192 bytes (8 KB) |
| Flash | 131,072 bytes (128 KB) |
| CPU | 8 MHz, no hardware FPU |
| DMA | None (MSP430); modelled for STM32/nRF52840 |
| Active power | 3.3 V × 3.5 mA = 11.55 mW |

---

## Notebook Contents — `Tiling.ipynb`

The notebook is structured as a single executable pipeline with 6 cells:

| Cell | Type | Content |
|------|------|---------|
| 0 | Code | Complete MISO framework: all 19 sections from `LayerInput` through `macro_tile_layer`, `ping_pong`, normalization, and Banerjee comparison engine |
| 1 | Markdown | Methodology for int8 comparison: converting Banerjee strategies to int8 for fair evaluation |
| 2 | Code | `_banerjee_compute_int8()` — Banerjee strategies at int8 + bar charts (energy, latency, SRAM on log scale) |
| 3 | Code | Pivot table comparing MISO vs Banerjee Hybrid at int8; exports `int8_comparison_results.xlsx` |
| 4 | Markdown | MISO algorithm description — formal pseudocode for the Orchestration Engine |
| 5 | Code | `_banerjee_compute_fp32()` — fp32 baseline comparison + charts |

### What Cell 0 implements (19 sections)

```
S1   Hardware constants       E_R, E_W, E_S, T_R, T_W, T_S, SRAM_BYTES
S2   LayerInput               layer descriptor (8 fields → auto-computes H_out, MACs, weights)
S3   TilingPlan               tiling work order produced by tile_layer()
S4   PingPongResult           DMA execution plan produced by ping_pong()
S5   LayerExecution           combined per-layer result (tiling + ping-pong + normalization)
S6   compute_buffers()        SRAM byte sizes with overhang formula
S7   estimate_cost()          4-term energy/latency formula
S8   tile_candidates()        divisors ∪ powers-of-2 per axis
S9   tile_layer()             exhaustive (T_c, T_r, T_f) tiling search
S10  ping_pong()              DMA double-buffer scheduler
S11  _run_strip()             per-strip pipeline helper
S12  table_ii_layers()        13 Table II layers from the Banerjee paper
S13  print_*()                summary, normalization, DMA explanation, Banerjee comparison
S15  NormParams / compute_norm_params()   3-mode input normalization
S16  macro_tile_layer()       outer C_out strip decomposition (binary search)
S17  run_layer() / run_all()  full pipeline entry points
S18  main()                   orchestration: run all → summary → comparison
S19  BanerjeeModelResult / print_banerjee_comparison()   fp32 reference engine
```

---

## Installation

### Run on Google Colab (recommended)

```
1. Upload Tiling.ipynb to Google Colab
2. Run Cell 0 first — it defines all classes and functions
3. Run subsequent cells in order
```

No additional pip installs are required for the core framework. Cell 2 and Cell 5 require:

```python
# Already available in Colab
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
```

Cell 3 uses `google.colab.files` for Excel export — this only works inside Colab.

### Run locally

```bash
git clone <this-repo>
cd <this-repo>
pip install matplotlib seaborn pandas numpy openpyxl
jupyter notebook Tiling.ipynb
```

> **Note:** Remove or comment out the `from google.colab import files` line and `files.download(...)` call in Cell 3 when running locally.

---

## Usage

### Run the full MISO pipeline

```python
# Define a layer
layer = LayerInput(
    name="MyConv", layer_type="Conv",
    C_in=1, H_in=32, W_in=32, C_out=6,
    K=5, stride=1, precision="int8"
)

# Run tiling + ping-pong + normalization
result = run_layer(layer, sram_budget=8192, objective="energy",
                   has_dma=False, norm_mode="uint8",
                   input_min=0.0, input_max=255.0)

print(f"Peak SRAM : {result.peak_sram} B")
print(f"Energy    : {result.total_energy_uj:.2f} µJ")
print(f"Latency   : {result.total_latency_ms:.4f} ms")
print(f"n_tiles   : {result.strips[0][0].n_tiles_total}")
```

### Run all 13 Table II benchmark layers

```python
layers  = table_ii_layers(precision="int8")
results = run_all(layers, sram_budget=8192, objective="energy",
                  has_dma=False, norm_mode="uint8",
                  input_min=0.0, input_max=255.0)

print_summary(results)
print_banerjee_comparison(results)
```

### Enable DMA double-buffering (for STM32 / nRF52840)

```python
# Change at the top of Cell 0:
HAS_DMA = True

# Or pass per-call:
result = run_layer(layer, has_dma=True, ...)
```

### Change normalization mode

```python
# For 12-bit ADC sensor data [0, 4095]
result = run_layer(layer, norm_mode="minmax", input_min=0.0, input_max=4095.0)

# For statistical sensor with known distribution
result = run_layer(layer, norm_mode="zscore",
                   input_min=0.0, input_max=100.0,
                   # pass mean/std via macro_tile_layer directly:
                   )
result = macro_tile_layer(layer, norm_mode="zscore", mean=50.0, std=15.0)
```

---

## Hardware Cost Model

Three physically distinct constants (corrects Banerjee's single-variable ambiguity):

| Constant | Value | Unit | Meaning |
|----------|-------|------|---------|
| `E_R` | 68.55 | nJ/byte | Flash → SRAM read energy |
| `E_W` | 393.75 | nJ/byte | SRAM → Flash write energy (**5.74× E_R**) |
| `E_S` | 38.232 | nJ/byte | SRAM internal access energy |
| `T_R` | 5.77 | µs/byte | Flash read latency |
| `T_W` | 8.75 | µs/byte | Flash write latency |
| `T_S` | 3.54 | µs/byte | SRAM access latency |

**Per-tile energy formula:**
```
E_tile = E_R×(buf_in + buf_wt) + E_W×buf_out + E_S×buf_out + E_comp×MACs_per_tile
E_total = E_tile × n_tiles
```

Because `E_W = 5.74 × E_R`, minimising write-backs (choosing larger tiles) is the primary optimisation target.

---

## Tiling Search

For each layer, MISO searches all legal `(T_c, T_r, T_f)` combinations:

```
rows_needed = T_r × stride + (K − 1)    ← overhang formula
cols_needed = T_f × stride + (K − 1)

buf_in  = T_c × rows_needed × cols_needed × B
buf_wt  = C_out × T_c × K² × B
buf_out = C_out × T_r × T_f × B

if buf_in + buf_wt + buf_out > 8192: SKIP   ← hard feasibility gate
```

The paper's strategies are recovered as special cases:
- **OaaT**: `T_c=C_in, T_r=H_out, T_f=W_out` → `n_tiles=1`
- **MM**: `T_c=C_in, T_r=1, T_f=1` → maximum `n_tiles`
- **MISO**: finds the minimum-energy point between these extremes

---

## Benchmark Layers (Table II, Banerjee et al.)

| Layer | Tc | Tr | Tf | Peak SRAM | PP? | n_tiles | Energy (µJ) | Latency (ms) |
|-------|----|----|----|-----------|-----|---------|-------------|--------------|
| MNIST-Conv1 | 1 | 28 | 28 | 5,878 B | ✓ | 1 | 2,791.66 | 123.39 |
| MNIST-Conv2 | 6 | 10 | 10 | 5,176 B | ✓ | 1 | 2,322.31 | 160.30 |
| MNIST-FC1 | 50 | 1 | 1 | 6,170 B | ✓ | 8 | 4,009.72 | 315.07 |
| MNIST-FC2 | 120 | 1 | 1 | 1,330 B | ✓ | 1 | 101.74 | 8.34 |
| HAR-Conv1 | 4 | 10 | 10 | 7,184 B | ✓ | 4 | 14,013.78 | 889.26 |
| HAR-Conv2 | 8 | 4 | 4 | 5,920 B | ✓ | 4 | 4,815.00 | 310.80 |
| HAR-FC1 | 24 | 1 | 1 | 6,424 B | ✓ | 12 | 6,826.62 | 501.69 |
| HAR-FC2 | 16 | 1 | 1 | 4,368 B | ✓ | 16 | 6,657.91 | 462.73 |
| TS-Conv1 | 3 | 6 | 12 | 7,830 B | ✗* | 18 | 61,278.33 | 3,665.72 |
| TS-Conv2 | 4 | 2 | 7 | 7,644 B | ✓ | 182 | 310,674.27 | 17,134.21 |
| TS-Conv3 | 2 | 3 | 3 | 6,800 B | ✓ | 75 | 113,831.21 | 5,561.70 |
| TS-FC1 | 25 | 1 | 1 | 7,825 B | ✓ | 10 | 6,887.46 | 508.56 |
| TS-FC2 | 150 | 1 | 1 | 6,643 B | ✓ | 2 | 1,016.51 | 83.67 |

*TS-Conv1: ping-pong SRAM = 8,310 B > 8,192 B → runs in single-buffer mode.

---

## Normalization Modes

| Mode | Formula | Use case | SRAM cost |
|------|---------|----------|-----------|
| `uint8` | `int8 = raw − 128` | Image pixels [0, 255] | 256 B LUT |
| `minmax` | `int8 = clip(round(raw × scale) + zero_point, −128, 127)` | ADC sensors, known range | 4 B |
| `zscore` | `int8 = clip(round((raw − mean) / std × 64), −128, 127)` | Statistical sensors | 8 B |

---

## Identified Shortcomings in Banerjee et al.

| # | Shortcoming | Fix in MISO |
|---|-------------|-------------|
| 1 | fp32 MNIST weights = 207 KB > 128 KB Flash | int8 reduces to 1,001 bytes |
| 2 | `E_sram` conflates E_R (68.55) and E_S (38.232 nJ/B) | Three named constants |
| 3 | Flash erase (302,400 nJ/4KB) omitted from write energy | Included in E_W |
| 4 | OaaT requires 23,512 B SRAM for MNIST-Conv1 | Hard feasibility gate |
| 5 | Conv + FC only (no DW, PW, GAP, pooling) | Extended operator coverage |
| 6 | fp32 only — 4× memory waste | int8 throughout |
| 7 | Three fixed strategies only | Exhaustive (T_c, T_r, T_f) search |
| 8 | No code generation | Firmware-oriented execution plan |

---

## Project Structure

```
.
├── Tiling.ipynb              # Main notebook — complete MISO implementation
├── README.md                 # This file
└── (outputs from Cell 3)
    └── int8_comparison_results.xlsx   # Auto-generated comparison table
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{deokar2026miso,
  author    = {Aaditya Deokar},
  title     = {Memory Optimization for TinyML and IoT Devices},
  year      = {2026},
  institution = {BITS Pilani, Dubai Campus},
  note      = {Design Project 1}
}
```

Original paper being improved:

```bibtex
@inproceedings{banerjee2021memory,
  author    = {Banerjee, Jishnu and Islam, Sahidul and Wei, Wei and
               Pan, Chen and Zhu, Dakai and Xie, Mimi},
  title     = {Memory-aware Efficient Deep Learning Mechanism for IoT Devices},
  booktitle = {IEEE 32nd International Conference on Application-specific
               Systems, Architectures and Processors (ASAP)},
  pages     = {187--194},
  year      = {2021}
}
```

---


