#!/usr/bin/env bash
set -e

# Go to the project root
cd final_project

# Remove any old v3 folder
rm -rf v3_cuda_only

# 1) Copy the serial baseline
cp -r v1_serial v3_cuda_only

pushd v3_cuda_only

# 2) Rename sources for CUDA
mv src/alexnet_serial.cpp src/alexnet_cuda.cu
mv src/layers_serial.cpp src/layers_cuda.cu
mv src/main.cpp            src/main_cuda.cpp

# 3) Update includes in main to refer to the CUDA headers (if you rename them)
#    (optional, depending on your include file names)
sed -i 's/alexnet_serial/alexnet_cuda/' src/main_cuda.cpp
sed -i 's/layers_serial/layers_cuda/' src/main_cuda.cpp

# 4) Overwrite Makefile with a CUDA‐only version
cat > Makefile << 'EOF'
# v3_cuda_only: single‐GPU CUDA implementation

NVCC      := nvcc
TARGET    := template
INCLUDES  := -I./include
SRCS_CU   := src/alexnet_cuda.cu src/layers_cuda.cu
SRCS_CPP  := src/main_cuda.cpp
OBJS      := $(SRCS_CU:.cu=.o) $(SRCS_CPP:.cpp=.o)
NVCCFLAGS := -std=c++11 -O3 -Xcompiler="-Wall -Wextra -pedantic"
LDFLAGS   := -lcudart -lm

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
EOF

popd

echo "✅ v3_cuda_only scaffolded. Next: implement your CUDA kernels in alexnet_cuda.cu and layers_cuda.cu, then run:"
echo "   cd final_project/v3_cuda_only && make && ./template"
