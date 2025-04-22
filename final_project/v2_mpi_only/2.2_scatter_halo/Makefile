# MPI‑only (Scatter+Halo) build
CXX       := mpicxx
CXXFLAGS  := -std=c++11 -O3 -Wall -Wextra -pedantic \
             -Wno-cast-function-type
LDFLAGS   := -lm
INCLUDES  := -I./include

SRCS      := src/main.cpp \
             src/alexnet_mpi.cpp \
             src/layers_mpi.cpp

OBJS      := $(SRCS:.cpp=.o)
TARGET    := template

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp $(wildcard include/*.hpp)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
