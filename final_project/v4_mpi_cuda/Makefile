# ────────────────────────────────────────────────────────────────
# Version‑4  (MPI + CUDA)  Makefile
# Tested on: CUDA 12.x  •  Open MPI 5.x  •  NixOS 24.05
# Fixes:
#   • filters out all “‑Wl,*” flags nvcc rejects
#   • disables dev‑runtime, uses shared cudart
# ────────────────────────────────────────────────────────────────

DIAG_FLAGS := -fno-diagnostics-show-caret -fno-diagnostics-show-option -fmessage-length=0

NVCC    := nvcc
CXX_MPI := mpicxx

MPI_CFLAGS  := $(shell $(CXX_MPI) --showme:compile 2>/dev/null)
MPI_LDFLAGS := $(filter-out -Wl%,$(shell $(CXX_MPI) --showme:link 2>/dev/null))

# Allow overriding from script, otherwise default
ifndef HOST_CUDA_ARCH_FLAGS
  EFFECTIVE_GPU_ARCH_FLAGS := \
                        -gencode arch=compute_75,code=sm_75 \
                        -gencode arch=compute_86,code=sm_86 \
                        -gencode arch=compute_86,code=compute_86
else
  EFFECTIVE_GPU_ARCH_FLAGS := $(HOST_CUDA_ARCH_FLAGS)
endif

NVCC_FLAGS := -std=c++17 -O3 \
              -ccbin=$(CXX_MPI) \
              -Xcompiler="-Wall -Wextra $(DIAG_FLAGS) $(MPI_CFLAGS)" \
              $(EFFECTIVE_CUDA_ARCH) \
              -cudart=shared \
              --cudadevrt=none \
              -Xcudafe "--diag_suppress=186"

INCLUDES := -I./include
TARGET   := template

SRCS_CPP := src/main_mpi_cuda.cpp
SRCS_CU  := src/alexnet_mpi_cuda.cu src/layers_mpi_cuda.cu
OBJS_CPP := $(SRCS_CPP:.cpp=.o)
OBJS_CU  := $(SRCS_CU:.cu=.o)
OBJS     := $(OBJS_CPP) $(OBJS_CU)

LDFLAGS := $(MPI_LDFLAGS) -lm

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDFLAGS)

$(OBJS_CU): %.o: %.cu $(wildcard include/*.hpp) Makefile
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

$(OBJS_CPP): %.o: %.cpp $(wildcard include/*.hpp) Makefile		
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c  $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
