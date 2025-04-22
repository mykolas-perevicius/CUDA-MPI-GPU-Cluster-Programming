# Updated Makefile for final_project (builds "template" and "conv_test" executables)
MPICC    = mpicxx
NVCC     = nvcc

CXXFLAGS = -O2 -std=c++17
NVCCFLAGS = -O2 -std=c++14

INCFLAGS = -Iinclude

# Sources for the full project.
SOURCES  = src/main.cpp src/alexnet_hybrid.cu
TARGET   = template

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(INCFLAGS) -ccbin=$(MPICC) $(SOURCES) -o $(TARGET)

# New target: conv_test builds the standalone convolution test.
conv_test: src/conv_test.cpp src/conv_layer.cu
	$(NVCC) $(NVCCFLAGS) $(INCFLAGS) -ccbin=$(MPICC) src/conv_test.cpp src/conv_layer.cu -o conv_test

.PHONY: clean
clean:
	rm -f $(TARGET) conv_test
