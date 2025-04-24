# Makefile for V4: MPI + CUDA (Host Staging)

# Use nvcc as the primary compiler, but tell it to use mpicxx for host code compilation/linking
NVCC       := nvcc
CXX_MPI    := mpicxx
# Ensure mpicxx is found if not in default PATH, e.g., export PATH=/path/to/openmpi/bin:$PATH
# Or provide full path: CXX_MPI := /path/to/openmpi/bin/mpicxx

NVCC_FLAGS := -std=c++11 -O3 -Xcompiler="-Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type" \
              -gencode arch=compute_75,code=compute_75 \
              -gencode arch=compute_75,code=sm_75 \
              -ccbin=$(CXX_MPI)

TARGET    := template
INCLUDES  := -I./include

# --- List Source Files ---
# .cpp files are compiled by nvcc using the C++ compiler specified by -ccbin
SRCS_CPP  := src/main_mpi_cuda.cpp
# .cu files are compiled by nvcc's CUDA compiler
SRCS_CU   := src/alexnet_mpi_cuda.cu src/layers_mpi_cuda.cu

OBJS      := $(SRCS_CPP:.cpp=.o) $(SRCS_CU:.cu=.o)

# --- Libraries ---
# Get MPI linker flags dynamically from mpicxx. Handle potential errors.
MPI_LDFLAGS := $(shell $(CXX_MPI) --showme:link 2>/dev/null)
ifeq ($(MPI_LDFLAGS),)
    $(error "Failed to get MPI linker flags from $(CXX_MPI). Is Open MPI configured correctly?")
endif
# Add CUDA runtime library
CUDA_LDFLAGS := -lcudart -lm

LDFLAGS   := $(MPI_LDFLAGS) $(CUDA_LDFLAGS)

.PHONY: all clean

all: $(TARGET)

# --- Compile Rules ---
# Rule for compiling .cu files (CUDA code)
# Use -dc for relocatable device code needed for separate compilation
%.o: %.cu $(wildcard include/*.hpp) Makefile
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Rule for compiling .cpp files (Host code, using nvcc which invokes mpicxx via -ccbin)
# No -dc needed here.
%.o: %.cpp $(wildcard include/*.hpp) Makefile
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# --- Link Rule ---
# Link all object files using nvcc (which uses mpicxx for final linking via -ccbin)
# Note: Linking .o files generated with -dc requires nvcc for device linking step.
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS)