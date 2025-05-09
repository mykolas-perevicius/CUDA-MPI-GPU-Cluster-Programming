# ───────────────────────────────────────────────────────────────
# AlexNet V3 – CUDA‑only build
# Tested with CUDA 11‑14 (nvcc >= 11.0)
# ───────────────────────────────────────────────────────────────

# Allow overriding from script, otherwise default to a common set including sm_86
ifndef HOST_CUDA_ARCH_FLAGS
  # Default for better portability, ensure sm_86 for your RTX 3090
  # You can add more architectures if you plan to run on diverse hardware
  # compute_X0 means PTX for X.0 and forwards compatibility
  EFFECTIVE_CUDA_ARCH := \
                        -gencode arch=compute_75,code=sm_75 \
                        -gencode arch=compute_86,code=sm_86 \
                        -gencode arch=compute_86,code=compute_86
else
  EFFECTIVE_CUDA_ARCH := $(HOST_CUDA_ARCH_FLAGS)
endif

NVCCFLAGS     := -std=c++11 -O3 \
                 -Xcompiler="-Wall -Wextra" \
                 -cudart=shared \
                 --cudadevrt=none        

INCLUDES      := -I./include
LDFLAGS       := -lm                      # cudart is linked automatically via -cudart=shared

SRC_DIR       := src
OBJ_DIR       := $(SRC_DIR)
SRCS          := $(SRC_DIR)/alexnet_cuda.cu \
                 $(SRC_DIR)/layers_cuda.cu \
                 $(SRC_DIR)/main_cuda.cpp
OBJS          := $(SRCS:.cu=.o)
OBJS          := $(OBJS:.cpp=.o)

TARGET        := template

# default rule
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo '--- Linking V3 Target ---'
	@echo 'NVCC Command: nvcc $(NVCCFLAGS) $(EFFECTIVE_CUDA_ARCH) -o $@ $^ $(LDFLAGS)'
	@nvcc $(NVCCFLAGS) $(EFFECTIVE_CUDA_ARCH) -o $@ $^ $(LDFLAGS)

# pattern rules
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo 'Compiling (CUDA): $<'
	@nvcc $(NVCCFLAGS) $(EFFECTIVE_CUDA_ARCH) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo 'Compiling (C++): $<'
	@nvcc $(NVCCFLAGS) $(EFFECTIVE_CUDA_ARCH) $(INCLUDES) -c $< -o $@

# housekeeping
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean
