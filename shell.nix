# shell.nix (with pinned nixpkgs for reproducibility and GCC compatibility)
{ pkgs ? let
    # Pin to a known stable branch of nixpkgs (nixos-23.11)
    # This ensures a consistent GCC version (GCC 13 in nixos-23.11) compatible with CUDA 12.x
    internal_nixpkgs_branch = "nixos-23.11";

    # SHA256 hash obtained from:
    # nix-prefetch-url --unpack https://github.com/NixOS/nixpkgs/archive/refs/heads/nixos-23.11.tar.gz
    # User provided: 1f5d2g1p6nfwycpmrnnmc2xmcszp804adp16knjvdkj8nz36y1fg
    internal_nixpkgs_sha256 = "1f5d2g1p6nfwycpmrnnmc2xmcszp804adp16knjvdkj8nz36y1fg";

  in import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/refs/heads/${internal_nixpkgs_branch}.tar.gz";
    sha256 = internal_nixpkgs_sha256;
  }) {} }:

let
  nixpkgs_pin_info = {
    branch = "nixos-23.11";
  };

  cudaVersion = pkgs.cudaPackages_12;
  mpiImpl = pkgs.openmpi;
  hostCompiler = pkgs.gcc; # GCC 13.x from nixos-23.11
in
pkgs.mkShell {
  name = "alexnet-full-project-dev-pinned";

  buildInputs = [
    pkgs.bash pkgs.gnumake pkgs.coreutils pkgs.gawk pkgs.gnused pkgs.gnugrep
    hostCompiler
    mpiImpl
    cudaVersion.cudatoolkit
    # pkgs.bear # Optional
    # pkgs.clang-tools # Optional
  ];

  shellHook = ''
    export CXX=${mpiImpl}/bin/mpicxx
    export CC=${mpiImpl}/bin/mpicc

    echo "--- Nix Shell for Full AlexNet Project (PINNED NIXPKGS: ${nixpkgs_pin_info.branch}) ---"
    if command -v nvcc &> /dev/null; then
      echo "Using CUDA Toolkit: $(nvcc --version 2>/dev/null | grep "release" || echo "NVCC found but version info failed")"
    else
      echo "NVCC not found in path."
    fi
    if command -v mpicc &> /dev/null; then
      echo "Using MPI: $(mpicc --version 2>/dev/null | head -n 1 || echo "MPICC found but version info failed")"
    else
      echo "MPICC not found in path."
    fi
    if command -v gcc &> /dev/null; then
      echo "Using Host GCC: $(gcc --version | head -n 1)"
    else
      echo "GCC not found in path."
    fi
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! IMPORTANT: GPU_ARCH_FLAGS in Makefiles must match LAPTOP's GPU Capability!"
    echo "!!! (e.g., for Quadro M1200 in Dell 5520, use: -gencode arch=compute_50,code=sm_50)"
    echo "!!! Testing scripts attempt to detect this, but verify Makefile usage."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
  '';
}