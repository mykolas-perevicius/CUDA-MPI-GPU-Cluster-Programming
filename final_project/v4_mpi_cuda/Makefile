# ────────────────────────────────────────────────────────────────
# Version‑4  (MPI + CUDA)  Makefile
# ────────────────────────────────────────────────────────────────

DIAG_FLAGS := -fno-diagnostics-show-caret -fno-diagnostics-show-option -fmessage-length=0

NVCC    := nvcc
CXX_MPI := mpicxx

# Get MPI CFLAGS for the host compiler part
MPI_COMPILE_FLAGS_RAW := $(shell $(CXX_MPI) --showme:compile 2>/dev/null)
# Filter out potentially problematic -Wl flags from compile flags too, if any
MPI_COMPILE_FLAGS := $(filter-out -Wl%,$(MPI_COMPILE_FLAGS_RAW))

# Get MPI LDFLAGS, focusing on -L and -l, try to avoid problematic ones for nvcc
MPI_LINK_SHOWME := $(shell $(CXX_MPI) --showme:link 2>/dev/null)
# Extract -L paths
MPI_L_PATHS := $(filter -L%,$(MPI_LINK_SHOWME))
# Extract -l library names
MPI_LIBS := $(filter -l%,$(MPI_LINK_SHOWME))
# Remove any remaining -Wl flags from the general flags if we take others
MPI_OTHER_LINK_FLAGS := $(filter-out -L% -l% -Wl%,$(MPI_LINK_SHOWME))


# Define GPU_ARCH_FLAGS:
GPU_ARCH_FLAGS ?= \
                  -gencode arch=compute_50,code=sm_50 \
                  -gencode arch=compute_50,code=compute_50

NVCC_BASE_FLAGS := -std=c++17 -O3 \
                   -ccbin=$(CXX_MPI) \
                   -Xcompiler="-Wall -Wextra $(DIAG_FLAGS)" \
                   -cudart=shared \
                   --cudadevrt=none \
                   -Xcudafe "--diag_suppress=186"

# Apply GPU flags to base flags
FULL_NVCC_COMPILE_FLAGS := $(NVCC_BASE_FLAGS) $(GPU_ARCH_FLAGS) \
                           -Xcompiler="$(MPI_COMPILE_FLAGS)" # Add MPI CFLAGS for host part

# For linking, we need the GPU arch flags for device linking consistency
FULL_NVCC_LINK_FLAGS := $(NVCC_BASE_FLAGS) $(GPU_ARCH_FLAGS)


INCLUDES := -I./include
TARGET   := template

SRCS_CPP := $(wildcard src/*.cpp)
SRCS_CU  := $(wildcard src/*.cu)
OBJS_CPP := $(patsubst src/%.cpp,src/%.o,$(SRCS_CPP))
OBJS_CU  := $(patsubst src/%.cu,src/%.o,$(SRCS_CU))
OBJS     := $(OBJS_CPP) $(OBJS_CU)

# Construct LDFLAGS carefully
LDFLAGS := $(MPI_L_PATHS) $(MPI_LIBS) $(MPI_OTHER_LINK_FLAGS) -lm

.PHONY: all clean
all: $(TARGET)

# Rule for .cu files (device code needs -dc for separate compilation)
src/%.o: src/%.cu $(wildcard include/*.hpp) Makefile
	@echo "Compiling .cu file: $<"
	@echo "Command: $(NVCC) $(FULL_NVCC_COMPILE_FLAGS) $(INCLUDES) -dc $< -o $@"
	@$(NVCC) $(FULL_NVCC_COMPILE_FLAGS) $(INCLUDES) -dc $< -o $@

# Rule for .cpp files (host code, compiled by nvcc using mpicxx as backend)
src/%.o: src/%.cpp $(wildcard include/*.hpp) Makefile		
	@echo "Compiling .cpp file: $<"
	@echo "Command: $(NVCC) $(FULL_NVCC_COMPILE_FLAGS) $(INCLUDES) -c $< -o $@"
	@$(NVCC) $(FULL_NVCC_COMPILE_FLAGS) $(INCLUDES) -c $< -o $@

# Linking all objects
$(TARGET): $(OBJS)
	@echo "Linking target: $@"
	@echo "Command: $(NVCC) $(FULL_NVCC_LINK_FLAGS) $^ -o $@ $(LDFLAGS)"
	@$(NVCC) $(FULL_NVCC_LINK_FLAGS) $^ -o $@ $(LDFLAGS)
  
clean:
	@echo 'Cleaning V4...'
	@rm -f src/*.o $(TARGET)