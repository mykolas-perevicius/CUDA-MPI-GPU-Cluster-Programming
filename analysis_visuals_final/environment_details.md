
### 1. Execution Environment Details

The primary performance data reported in this analysis was generated on the following system configuration:

*   **Operating System (Dev/Primary Collection):** NixOS 23.11 ( अधिकांशतः), with final runs validated on Fedora 37 for consistency with target.
*   **CPU:** Intel Core i7-7700HQ @ 2.80GHz (Laptop)
*   **GPU:** NVIDIA Quadro M1200 Mobile (4GB GDDR5, Compute Capability 5.0 - Maxwell)
*   **RAM:** 32 GB DDR4
*   **CUDA Toolkit Version:** 12.2 (primary for dev), with tests against 12.4 compatibility. V3/V4 Makefiles target sm_50 due to M1200.
*   **MPI Implementation:** Open MPI 4.1.x series (via NixOS packages), and system Open MPI on Fedora cluster for target.
*   **Host Compiler (GCC):** Version 11.x to 12.x (depending on NixOS channel / Fedora version).
*   **NVCC:** Bundled with CUDA Toolkit 12.2 / 12.4.
*   **Key Python Libraries:** pandas (for data handling), duckdb (for data warehousing), matplotlib & seaborn (for plotting).

**Note on Data Collection:**
*   Initial data collection and most development iterations occurred on the NixOS laptop described.
*   Log files from a Slurm-based Fedora cluster (specific CPU/GPU models vary by node) were also ingested to broaden the dataset, though the laptop data forms the primary basis for the detailed version-to-version performance progression analysis due to consistent hardware.
*   The `total_time_s` metric captures end-to-end wall-clock time for the core computational part of the AlexNet forward pass, as reported by the respective `main` programs.
