# Makefile for Serial Version (V1) - Use g++
# Access shared data/docs using relative paths like '../data/' from src files.

CXX = g++
# Add -O3 for optimization, -g for debugging if needed
CXXFLAGS = -Wall -std=c++11 -O3 #-g
# Add -lm if using math functions like sqrt, exp etc.
LDFLAGS = -lm
INCLUDES = -I./include

# --- List the final .cpp source files ---
SRCS = src/main.cpp src/alexnet_serial.cpp src/layers_serial.cpp

OBJS = $(SRCS:.cpp=.o)

TARGET = template

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDFLAGS) # Linker flags often go at the end

%.o: %.cpp $(wildcard include/*.hpp) # Recompile .o if corresponding .cpp or *any* .hpp changes
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)