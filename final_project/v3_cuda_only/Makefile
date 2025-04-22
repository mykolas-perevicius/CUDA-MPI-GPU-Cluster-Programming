# Makefile for CUDA-Only Version (V3) - Use nvcc
# --- MANUALLY EDIT/COMPLETE THIS ---
# Access shared data/docs using relative paths like '../data/' from src files.

NVCC = nvcc
# --- Adjust CUDA architecture flags as needed ---
SMS ?= 75 80 86 90
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

NVCCFLAGS = -O3 -std=c++11 $(GENCODE_FLAGS)
LDFLAGS = -lcudart
INCLUDES = -I./include

# --- !!! UPDATE THIS LIST based on final .cu/.cpp files after modification !!! ---
SRCS_CU = src/alexnet_cuda.cu src/layers.cu # Example names - EDIT ME
SRCS_CPP = src/main_cuda.cpp # Example name - EDIT ME
OBJS_CU = $(SRCS_CU:.cu=.o)
OBJS_CPP = $(SRCS_CPP:.cpp=.o)
OBJS = $(OBJS_CU) $(OBJS_CPP)

TARGET = template

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
