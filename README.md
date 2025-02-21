# Flow Partition with Porous Structure in Open Channels

This repository has the solution scripts and simulation cases for the paper "Flow Partition in Two-Dimensional Open Channels with Porous Structures". The problem setup is shown in the following figure. The key question addressed in the paper is how to compute $\alpha$ as a function of Froude number $Fr$, channel opening ratio $\beta$, and drag coefficient $C_d$.

<p align="center">
  <img src="images/scheme_diagram_side.png" width="400" alt="Flow over porous structure">
  <br>
  <em>Figure 1: Schematic diagram of flow partition due to a porous structure in an open channel.</em>
</p>

## Prerequisites

### Clone the repository

```bash
git clone https://github.com/zhaoyu-li/flow_partition_porous_structure.git
```

### Create a conda environment

```bash
conda env create -f environment.yml
```

### Install pyHMT2D


- Python 3.8+
- NumPy
- SciPy
- Matplotlib


## Simulation Cases

### Case 1: Flow over a porous structure




