# ───────────────────────────────────────────────────────────────
# AlexNet V3 – CUDA‑only build (FINAL - STRICTLY EXPLICIT)
# ───────────────────────────────────────────────────────────────

# Define GPU_ARCH_FLAGS:
GPU_ARCH_FLAGS ?= \
                  -gencode arch=compute_50,code=sm_50 \
                  -gencode arch=compute_50,code=compute_50

# Compiler and flags
NVCC          := nvcc
NVCCFLAGS     := -std=c++11 -O3 \
                 -Xcompiler="-Wall -Wextra" \
                 -cudart=shared \
                 --cudadevrt=none \
                 $(GPU_ARCH_FLAGS)

INCLUDES      := -I./include # Assuming include is in the same dir as Makefile
LDFLAGS       := -lm

# --- Explicit Source and Object File Definitions ---
# All .o files will be placed in the 'src' directory.

SRC_DIR       := src
OBJ_DIR       := $(SRC_DIR)

# Explicitly define each object file
ALEXNET_CUDA_OBJ := $(OBJ_DIR)/alexnet_cuda.o
LAYERS_CUDA_OBJ  := $(OBJ_DIR)/layers_cuda.o
MAIN_CUDA_OBJ    := $(OBJ_DIR)/main_cuda.o

# List of all object files
OBJS          := $(ALEXNET_CUDA_OBJ) $(LAYERS_CUDA_OBJ) $(MAIN_CUDA_OBJ)

TARGET        := template

.PHONY: all clean default

# Default target
default: all

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo '--- Linking V3 Target $(TARGET) from $^ ---'
	@echo "NVCC Command: $(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)"
	@$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# --- Explicit Rules for Each Object File ---
# No pattern rules. Each object is built by its own specific rule.

$(ALEXNET_CUDA_OBJ): $(SRC_DIR)/alexnet_cuda.cu $(wildcard include/*.hpp) Makefile
	@echo 'Compiling (CUDA): $(SRC_DIR)/alexnet_cuda.cu to $(ALEXNET_CUDA_OBJ)'
	@mkdir -p $(OBJ_DIR)
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $(SRC_DIR)/alexnet_cuda.cu -o $(ALEXNET_CUDA_OBJ)

$(LAYERS_CUDA_OBJ): $(SRC_DIR)/layers_cuda.cu $(wildcard include/*.hpp) Makefile
	@echo 'Compiling (CUDA): $(SRC_DIR)/layers_cuda.cu to $(LAYERS_CUDA_OBJ)'
	@mkdir -p $(OBJ_DIR)
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $(SRC_DIR)/layers_cuda.cu -o $(LAYERS_CUDA_OBJ)

$(MAIN_CUDA_OBJ): $(SRC_DIR)/main_cuda.cpp $(wildcard include/*.hpp) Makefile
	@echo 'Compiling (C++ via NVCC): $(SRC_DIR)/main_cuda.cpp to $(MAIN_CUDA_OBJ)'
	@mkdir -p $(OBJ_DIR)
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $(SRC_DIR)/main_cuda.cpp -o $(MAIN_CUDA_OBJ)

clean:
	@echo 'Cleaning V3...'
	@rm -f $(OBJ_DIR)/*.o $(TARGET)