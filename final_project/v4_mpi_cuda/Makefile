# final_project/v4_mpi_cuda/Makefile
# Build + lint for Version-4 (MPI + CUDA)
# Combined: .clang-tidy for checks, compile_commands.json for build flags

# --- Diagnostics compact to one line ----------------------------------------
DIAG_FLAGS    := -fno-diagnostics-show-caret -fno-diagnostics-show-option -fmessage-length=0

# --- Tools ------------------------------------------------------------------
NVCC          := nvcc
CXX_MPI       := mpicxx
CLANG_TIDY    := clang-tidy
# Use 'bear' (Build EAR) to generate compile_commands.json
# Install via package manager, e.g., sudo apt install bear
BEAR          := bear

# --- MPI flags --------------------------------------------------------------
# Fetch compile/link flags from MPI wrapper. bear will capture these during build.
MPI_CFLAGS_LIST := $(shell $(CXX_MPI) --showme:compile 2>/dev/null)
MPI_CFLAGS      := $(MPI_CFLAGS_LIST)
MPI_LDFLAGS   := $(shell $(CXX_MPI) --showme:link    2>/dev/null)
ifeq ($(strip $(MPI_CFLAGS)),)
  $(warning "Could not determine MPI compile flags from $(CXX_MPI) --showme:compile")
endif
ifeq ($(strip $(MPI_LDFLAGS)),)
  $(warning "Could not determine MPI link flags from $(CXX_MPI) --showme:link")
endif

# --- Locate CUDA include dir (Good practice, but bear should capture it) ----
NVCC_PATH     := $(shell which nvcc)
ifeq ($(NVCC_PATH),)
  $(error "nvcc not found in PATH. Ensure CUDA toolkit is installed and configured.")
endif
CUDA_BASE     := $(shell dirname $(shell dirname $(NVCC_PATH)))
CUDA_INC      := $(CUDA_BASE)/include

# --- NVCC compile flags -----------------------------------------------------
# These flags are used by the actual build process monitored by bear.
NVCC_FLAGS    := -std=c++17 -O3 \
                  -ccbin=$(CXX_MPI) \
                  -Xcompiler="-Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type -Wno-type-limits $(DIAG_FLAGS) $(MPI_CFLAGS)" \
                  -gencode arch=compute_75,code=compute_75 \
                  -gencode arch=compute_75,code=sm_75 \
                  -Xcudafe "--diag_suppress=186"

CXXFLAGS     += $(DIAG_FLAGS) $(MPI_CFLAGS)

INCLUDES      := -I./include
TARGET        := template

# --- Sources & objects ------------------------------------------------------
SRCS_CPP      := src/main_mpi_cuda.cpp
SRCS_CU       := src/alexnet_mpi_cuda.cu src/layers_mpi_cuda.cu
OBJS          := $(SRCS_CPP:.cpp=.o) $(SRCS_CU:.cu=.o)
# Source files to be linted (headers are checked via includes)
LINT_SOURCES  := $(SRCS_CPP) $(SRCS_CU)

# --- Link -------------------------------------------------------------------
CUDA_LDFLAGS  := -lcudart
LDFLAGS       := $(MPI_LDFLAGS) $(CUDA_LDFLAGS)

# --- Phony targets ----------------------------------------------------------
.PHONY: all clean lint compile_commands build_only

# Default target: Build the executable
all: $(TARGET)

# Target to build the executable without involving bear
# Useful if bear is not installed or for faster rebuilds when DB isn't needed.
build_only: $(TARGET)

# Target to generate compile_commands.json using bear
# This involves cleaning and then rebuilding the project under bear's watch.
compile_commands: clean
	@if command -v $(BEAR) >/dev/null 2>&1; then \
	  echo "Generating compile_commands.json using bear by building target '$(TARGET)'..."; \
	  $(BEAR) -- make build_only; \
	  echo "compile_commands.json generated."; \
	else \
	  echo ""; \
	  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"; \
	  echo "!!! Warning: 'bear' command not found."; \
	  echo "!!! Cannot generate compile_commands.json automatically."; \
	  echo "!!! Install 'bear' (e.g., 'sudo apt install bear') for accurate linting."; \
	  echo "!!! Makefile will proceed without compile_commands.json."; \
	  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"; \
	  echo ""; \
	fi # <<<====== ADDED THE MISSING 'fi' HERE

# Standard build target definition (used by 'all', 'build_only', and 'bear')
$(TARGET): $(OBJS)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Device objects (.cu → .o) - Separate compilation step
%.o: %.cu $(wildcard include/*.hpp) Makefile
	@echo "Compiling (device) $<..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Host objects (.cpp → .o) - Compiled via nvcc
%.o: %.cpp $(wildcard include/*.hpp) Makefile
	@echo "Compiling (host) $<..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c  $< -o $@

# Lint target using compile_commands.json (if available) for build flags
# and .clang-tidy (implicitly found) for checks configuration.
# We add explicit flags via -extra-arg to help clang-tidy interpret
# the CUDA/MPI environment even when using the compilation database.
lint: compile_commands # Ensure compile_commands.json is attempted first
	@if command -v $(CLANG_TIDY) >/dev/null 2>&1; then \
	  if [ -f compile_commands.json ]; then \
	    echo "Running clang-tidy (using compile_commands.json, .clang-tidy, + explicit args)..."; \
	    # Prepare MPI flags for clang-tidy command line, even with -p .
	    CLANG_TIDY_MPI_ARGS=$$(echo "$(MPI_CFLAGS_LIST)" | sed 's/[^ ]*/-extra-arg-before=&/g'); \
	    $(CLANG_TYDY) \
	      -p . \
	      $(LINT_SOURCES) \
	      -- \
	      -extra-arg-before=--cuda-path=$(CUDA_BASE) \
	      -extra-arg-before=--cuda-gpu-arch=sm_75 \
	      $${CLANG_TIDY_MPI_ARGS}; \
	  else \
	    # Fallback if compile_commands.json couldn't be generated
	    echo "Warning: compile_commands.json not found. Linting may be inaccurate."; \
	    echo "Attempting lint with manually specified flags (may be incomplete)..."; \
	    CLANG_TIDY_MPI_ARGS_FALLBACK=$$(echo "$(MPI_CFLAGS_LIST)" | sed 's/[^ ]*/-extra-arg=&/g'); \
	    $(CLANG_TIDY) \
	      $(LINT_SOURCES) $(wildcard include/*.hpp) \
	      -- \
	      -std=c++17 \
	      -I./include \
	      -I$(CUDA_INC) \
	      --cuda-gpu-arch=sm_75 \
	      $${CLANG_TIDY_MPI_ARGS_FALLBACK}; \
	  fi \
	else \
	  echo "clang-tidy not found; skipping lint."; \
	fi

clean:
	@echo "Cleaning project files..."
	rm -f $(TARGET) $(OBJS) compile_commands.json bear-* # Remove bear's log file too
