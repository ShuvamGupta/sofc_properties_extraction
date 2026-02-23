# SOFC Anode Microstructure Analysis Toolkit

This repository provides a Python-based framework for quantitative analysis of segmented FIB/SEM microstructures of Solid Oxide Fuel Cell (SOFC) anodes.

The code processes segmented datasets containing Nickel (Ni), YSZ, and pore phases, and extracts grain-resolved geometric descriptors together with rigorous Triple Phase Boundary (TPB) statistics.

---

## Overview

The electrochemical performance of SOFC anodes is strongly governed by microstructural characteristics such as TPB density, nickel grain connectivity, and phase interfaces. 

This toolkit enables automated, scalable, and reproducible extraction of:

- Phase grain geometry
- Total TPB length
- TPB length shared by each individual Phase grain
- Phase interface statistics

The implementation supports both 2D and 3D segmented datasets.

---

## Methodology

### Grain Identification

Nickel grains are identified using connected component labeling applied to the segmented Ni phase. Each connected region is assigned a unique grain ID, enabling grain-resolved geometric measurements.

The I/O handling and geometry-related helper functions are derived and adapted from the mspacman repository:
https://github.com/ShuvamGupta/mspacman

### TPB Detection

Triple Phase Boundary (TPB) length is computed using a voxel-edge-based enumeration approach.

All voxel edges along the x, y, and z directions are examined. An edge is classified as a TPB edge when the local voxel configuration contains all three phases: Ni, YSZ, and pore.

The total TPB length is obtained by summing all detected TPB edges and scaling by the voxel size.

This discrete edge-based formulation is consistent with standard FIB/SEM tomographic TPB quantification approaches used in SOFC microstructure analysis.

### Grain-Resolved TPB Assignment

After total TPB edges are identified, grain-level TPB statistics are computed.

For each TPB edge, the contribution is assigned to every distinct non-zero Ni grain among the incident voxels. No artificial geometric splitting of TPB segments is performed.

Thus, the TPB length shared by each grain is calculated on top of the global TPB extraction, ensuring physically meaningful “grain surface length in contact with TPB” statistics.

---

## Input Requirements

The code expects:

- Segmented 2D or 3D image data
- Integer phase labels (e.g. Ni, YSZ, Pore)
- Known voxel size (in micrometers)

Supported formats may include NumPy arrays and TIFF stacks (depending on your implementation).

---

## Output

The analysis generates structured tabular outputs containing:

- Grain geometry
- TPB edge counts per direction
- TPB length per grain
- Total TPB length of the volume
- Voxelized TPB image as tiff

Results are exported as CSV files for further analysis.

---

## Installation

Install directly from GitHub:

pip install git+https://github.com/ShuvamGupta/sofc_properties_extraction.git

---

## Example Usage

A complete working example is provided in:

example_usage.py

---

## Applications

This toolkit can be used for:

- Quantitative SOFC microstructure characterization
- Degradation and aging studies
- Structure–property correlation
- Microstructure-informed electrochemical modeling
- Data-driven microstructural optimization

---

## Reproducibility

All calculations are deterministic and based on voxel-level enumeration. Results depend only on the provided segmentation and voxel size.

---

## Author

Shuvam Gupta  
Forshungszentrum Jülich

---

## License

Specify your license here (e.g., MIT License).
