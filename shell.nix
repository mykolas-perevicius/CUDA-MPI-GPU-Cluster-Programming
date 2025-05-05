{ pkgs ? import <nixpkgs> {} }:

let
  # Use CUDA 12.x package set from nixpkgs
  # Ensure your NixOS NVIDIA driver (from nvidia-smi) supports this!
  cudaVersion = pkgs.cudaPackages_12;

  # Use OpenMPI (matches PC setup)
  mpiImpl = pkgs.openmpi;

in
pkgs.mkShell {
  name = "alexnet-full-project-dev";

  # Packages needed for building and running ALL versions (V1-V4)
  buildInputs = [
    # Core scripting/build tools
    pkgs.bash                # For running the .sh script
    pkgs.gnumake             # For 'make' in all subdirs
    pkgs.coreutils           # Provides basic commands like cd, mkdir, etc.
    pkgs.gawk                # Used by the script for parsing
    pkgs.gnused              # Used by the script for parsing
    pkgs.gnugrep             # Used by the script for parsing

    # C/C++ compiler (for V1, and host compiler for V3/V4)
    pkgs.gcc                 # Matches PC environment reasonably

    # MPI (for V2, V4)
    mpiImpl                  # OpenMPI (mpicc, mpicxx, mpirun)

    # CUDA (for V3, V4)
    cudaVersion.cudatoolkit  # NVCC and CUDA runtime libs (CUDA 12.x)

    # Optional: Only needed if you want to run 'make lint' MANUALLY inside subdirs
    # pkgs.bear
    # pkgs.clang-tools
  ];

  # Set environment variables potentially needed by Makefiles or runtime
  # (Nix often handles this, but being explicit can help)
  shellHook = ''
    # Ensure MPI compilers are preferred if needed (V2 Makefile might just use 'g++')
    # The V4 Makefile specifically uses 'mpicxx' via nvcc's -ccbin, so this helps V2 primarily.
    export CXX=${mpiImpl}/bin/mpicxx
    export CC=${mpiImpl}/bin/mpicc

    echo "--- Nix Shell for Full AlexNet Project (V1-V4) ---"
    echo "Using CUDA Toolkit: $(nvcc --version | grep "release")"
    echo "Using MPI: $(mpicc --version | head -n 1)"
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! IMPORTANT: Before running the script, ensure GPU_ARCH_FLAGS in:"
    echo "!!!   - final_project/v3_cuda_only/Makefile"
    echo "!!!   - final_project/v4_mpi_cuda/Makefile"
    echo "!!! MATCH YOUR LAPTOP's GPU Compute Capability (e.g., sm_86, sm_89)!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
  '';
}