# ───────────────────────────────────────────────────────────────
# AlexNet V3 – CUDA‑only build
# Tested with CUDA 11‑14 (nvcc >= 11.0)
# ───────────────────────────────────────────────────────────────

CUDA_ARCH     := -gencode arch=compute_50,code=sm_50   
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
	@echo 'NVCC Command: nvcc $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^ $(LDFLAGS)'
	@nvcc $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^ $(LDFLAGS)

# pattern rules
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo 'Compiling (CUDA): $<'
	@nvcc $(NVCCFLAGS) $(CUDA_ARCH) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo 'Compiling (C++): $<'
	@nvcc $(NVCCFLAGS) $(CUDA_ARCH) $(INCLUDES) -c $< -o $@

# housekeeping
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean
