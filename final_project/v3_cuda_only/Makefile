# final_project/v3_cuda_only/Makefile

# CUDA / GPU‐only version of AlexNet forward pass

NVCC      := nvcc
TARGET    := template
INCLUDES  := -I./include

# Add both a PTX entry (compute_75) and a SASS entry (sm_75)
# so that if your GPU is e.g. Ampere (sm_80/86) it'll JIT from PTX.
NVCCFLAGS := -std=c++11 -O3 \
             -Xcompiler="-Wall -Wextra" \
             -gencode arch=compute_75,code=compute_75 \
             -gencode arch=compute_75,code=sm_75

SRCS_CU   := src/alexnet_cuda.cu src/layers_cuda.cu
SRCS_CPP  := src/main_cuda.cpp
OBJS      := $(SRCS_CU:.cu=.o) $(SRCS_CPP:.cpp=.o)
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
