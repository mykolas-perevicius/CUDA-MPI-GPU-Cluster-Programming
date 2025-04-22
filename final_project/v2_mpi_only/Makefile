# Makefile for MPI-Only Version (V2) - Use mpicxx
# !!! MANUALLY VERIFY AND UPDATE SRCS LIST BELOW !!!
# Access shared data/docs using relative paths like '../data/' from src files.

CXX = mpicxx # Use the MPI C++ compiler wrapper
CXXFLAGS = -O3 -Wall -std=c++11 # Add other necessary flags
LDFLAGS = # Add necessary libraries (e.g., -lm)
INCLUDES = -I./include

# --- !!! UPDATE THIS LIST based on final .cpp files after modification/renaming !!! ---
SRCS = src/main.cpp src/alexnet_mpi.cpp.stub src/layers_mpi.cpp.stub # Example names - EDIT ME
OBJS = $(SRCS:.cpp=.o) $(SRCS:.cpp.stub=.o) # Handle potential .stub extension temporarily

TARGET = template

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile .cpp files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile .cpp.stub files (treat as .cpp for compilation, assuming you rename later)
%.o: %.cpp.stub
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
