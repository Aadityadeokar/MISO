# MISO

# MISO: Memory-aware Intermittent System Optimizer for CNNs on MCUs

This notebook provides a comprehensive framework for analyzing and optimizing the energy, latency, and SRAM usage of Convolutional Neural Networks (CNNs) deployed on resource-constrained microcontrollers (MCUs), specifically targeting the MSP430F5529.

## Overview

The core of this project is a multi-level tiling and scheduling algorithm designed to overcome hardware limitations (e.g., 8KB SRAM) by:

*   **Optimal Inner Tiling:** Exhaustively searching for the best tile dimensions (channels, rows, columns) to minimize energy or latency while ensuring SRAM feasibility.
*   **Macro-Tiling:** Splitting large layers into smaller, manageable "strips" when the entire layer's weights exceed SRAM capacity.
*   **DMA Scheduling (Ping-Pong):** Modeling the benefits of Direct Memory Access (DMA) for overlapping Flash data transfers with CPU computation, reducing effective latency.
*   **Input Normalization:** Supporting various modes (`uint8`, `minmax`, `zscore`) to map raw sensor data to `int8` computation ranges, crucial for efficient fixed-point arithmetic.

## Key Features

*   **Hardware Cost Models:** Incorporates detailed energy and latency constants for Flash and SRAM access, as well as CPU arithmetic (MAC operations).
*   **Layer Abstraction:** A `LayerInput` dataclass simplifies layer definition (Conv, FC, input/output dimensions, kernel size, stride, precision).
*   **Feasibility Guarantee:** Ensures that all generated execution plans respect the MCU's SRAM budget through intelligent tiling strategies.
*   **Banerjee et al. Comparison:** Includes an implementation of reference scheduling strategies from Banerjee et al. (IEEE ASAP 2021) for both `fp32` and `int8` precisions, allowing for direct comparison and quantification of MISO's improvements.
*   **Detailed Reporting:** Generates comprehensive summary tables, normalization parameter breakdowns, and visual explanations of DMA double-buffering.

## How to Use

1.  **Define Layers:** Modify the `table_ii_layers()` function or define your own `LayerInput` objects to describe your CNN architecture.
2.  **Configure Hardware:** Adjust constants like `SRAM_BYTES`, `FLASH_BYTES`, and `HAS_DMA` in Section 1 to match your target MCU.
3.  **Run the Notebook:** Execute all cells to run the MISO pipeline for all defined layers. The output will include:
    *   A pipeline summary with optimal tile dimensions, SRAM usage, and total energy/latency for each layer.
    *   Normalization parameter tables and examples.
    *   A side-by-side comparison of MISO against various Banerjee et al. strategies, highlighting energy, latency, and SRAM improvements.
    *   Visual explanations of DMA scheduling.

## Outputs

The notebook outputs several tables and plots, including:

*   **Pipeline Summary:** Per-layer breakdown of optimization results.
*   **Normalization Table:** Details on input scaling for int8 conversion.
*   **Banerjee Comparison:** Quantitative comparison of MISO with established methods.
*   **Visualization:** Bar plots showing energy, latency, and SRAM for different strategies.

This tool is invaluable for designers of low-power, intermittently-powered embedded systems that rely on CNN inference.
