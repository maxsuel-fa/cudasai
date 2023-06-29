# Compiler flags
CXX = nvcc
CUDAFLAGS = -arch=sm_61

# Directories
SRC_DIR = .
OBJ_DIR = .
BIN_DIR = .

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Object files
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Executable
TARGET = $(BIN_DIR)/main

# Default target
all: $(TARGET)

# Compiling object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CXX) $(CUDAFLAGS) -c -o $@ $<

# Linking object files
$(TARGET): $(OBJS)
	$(CXX) $(CUDAFLAGS) -o $(TARGET) $(OBJS)

# Cleaning object files and executable
clean:
	$(RM) $(OBJS) $(TARGET)

test:
	./$(TARGET) < input_file > test/output_file
	diff test/output_file test/expected_output_file

# Running the psrs
run:
	clear & ./$(TARGET) $(DIM)


