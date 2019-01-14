# Unix commands.
PYTHON := python3
NVCC_COMPILE := nvcc -c -o
RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared

# File structure.
BUILD_DIR := build
INCLUDE_DIRS := src
TORCH_FFI_BUILD := build_ffi.py
KNN_KERNEL := $(BUILD_DIR)/knn_cuda_kernel.so
TORCH_FFI_TARGET := $(BUILD_DIR)/knn_pytorch/_knn_pytorch.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$d)

DEBUB := 0

# Debugging
ifeq ($(DEBUG), 1)
  COMMON_FLAGS += -DDEBUG -g -O0
  NVCC_FLAGS += -G
else
  COMMON_FLAGS += -DNDEBUG -O2
endif

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(KNN_KERNEL) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)

$(BUILD_DIR)/%.so: src/%.cu
	@ mkdir -p $(BUILD_DIR)
	# Separate cpp shared library that will be loaded to the extern C ffi
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS)

clean:
	$(RM_RF) $(BUILD_DIR) $(KNN_KERNEL)
