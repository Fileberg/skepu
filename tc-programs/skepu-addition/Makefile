#######################################################################
#                     ____  _        ____  _   _                      #
#                    / ___|| | _____|  _ \| | | |                     #
#                    \___ \| |/ / _ \ |_) | | | |                     #
#                     ___) |   <  __/  __/| |_| |                     #
#                    |____/|_|\_\___|_|    \___/                      #
#                                                                     #
#                       ~ SkePU main Makefile ~                       #
#######################################################################
.DEFAULT_GOAL=all-parallel

# Default backend set for SkePU precompiler.
# BACKENDS = -openmp -opencl -cuda
BACKENDS = -cuda

SKEPU_OPTIONS =
PRECOMPILER_OPTIONS = --no-preserve-lines

# Backend-specific flags
# BACKEND_FLAGS = -O3 -g -DSKEPU_DEBUG=3 -DSKEPU_ENABLE_EXCEPTIONS
BACKEND_FLAGS = -O3 -g #-DSKEPU_DEBUG=3 #-DSKEPU_ENABLE_EXCEPTIONS

# Default OpenCL specific flags (matching a CUDA-provided installation)
OPENCL_FLAGS = -I /usr/local/cuda/include -L /usr/local/cuda/lib64/ -lOpenCL

# path to Clang source directory (repository root)
CLANG_INCLUDE = $(PWD)/../llvm/clang/lib/Headers

# path to the SkePU sources
SKEPU_INCLUDE = $(PWD)/../skepu-headers/src
CUTLASS_INCLUDE =
CUTLASS_INCLUDE += -I /usr/local/cuda/include
CUTLASS_INCLUDE += -I $(PWD)/../cutlass/include
CUTLASS_INCLUDE += -I $(PWD)/../cutlass/tools/util/include
CUTLASS_INCLUDE += -gencode arch=compute_80,code=sm_80
CXX = g++-10
SEQ_CXX = g++-10
#DBGR = lldb --

CLUSTER_CXX      = mpic++
CLUSTER_BACKENDS = -openmp
CLUSTER_CXXFLAGS = -std=c++17 -fopenmp $$(pkg-config --cflags starpu-1.3 starpumpi-1.3) -I ./  -DSKEPU_DEBUG=3  -g -O3
# CLUSTER_CXXFLAGS = -std=c++17 -fopenmp $$(pkg-config --cflags starpu-1.3 starpumpi-1.3) -I ./  -g -O3
CLUSTER_LIBS     = $$(pkg-config --libs starpu-1.3 starpumpi-1.3)


CLANGTOOL_USER_FLAGS =
# ---------------------------------------------------------------------------- #
# Compilers.

# Conditionally choose either g++ or nvcc for target compiler,
# depending on whether CUDA is in the list of backends
# This will also be used for non-preprocessed compilation of SkePU programs.
ifneq (,$(findstring cuda, $(BACKENDS)))
CXX = nvcc
endif

# Location of SkePU precompiler binary.
SKEPU = $(PWD)/../build/llvm/bin/skepu-tool


# ---------------------------------------------------------------------------- #
# Test program names and target directories.

PAR_DIR     = parallel
SEQ_DIR     = sequential
CLUSTER_DIR = cluster
# RUN_DIR     = run

# All SkePU example programs (used for 'make all-[parallel|sequential|cluster]' and 'make clean').

PROGS=
PROGS += my_thing


CLUSTER_PROGS =


TESTS          = $(addprefix $(SEQ_DIR)/,      $(PROGS))
PTESTS         = $(addprefix $(PAR_DIR)/,      $(PROGS))
CLUSTER_TESTS  = $(addprefix $(CLUSTER_DIR)/,  $(CLUSTER_PROGS))

# ---------------------------------------------------------------------------- #
# Compiler flags begin here.

# Flags for precompiler.
SKEPU_FLAGS = $(PRECOMPILER_OPTIONS)
SKEPU_FLAGS += -- -std=c++17 -Wno-expansion-to-defined
SKEPU_FLAGS += -I $(CLANG_INCLUDE)
SKEPU_FLAGS += -I $(SKEPU_INCLUDE)
# SKEPU_FLAGS += $(CUTLASS_INCLUDE)
SKEPU_FLAGS += $(CLANGTOOL_USER_FLAGS)

# Activate backend flags for CUDA backend
ifneq (,$(findstring cuda, $(BACKENDS)))
BACKEND_FLAGS += -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" $(CUTLASS_INCLUDE)
NVCCPASS = -Xcompiler
FILETYPE = cu
else
FILETYPE = cpp
endif

# Activate backend flags for OpenMP backend
ifneq (,$(findstring openmp, $(BACKENDS)))
BACKEND_FLAGS += $(NVCCPASS) -fopenmp
endif

# Activate backend flags for OpenCL backend
ifneq (,$(findstring opencl, $(BACKENDS)))
BACKEND_FLAGS += $(OPENCL_FLAGS)
endif

# Flags for target compiler (preprocessed sources).
TARGET_FLAGS = -std=c++17 $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE) -I ./  $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)

# Flags for non-preprocessed compilation.
# SEQ_FLAGS = -std=c++17 -O0 -g -DSKEPU_DEBUG=3 -I $(SKEPU_INCLUDE) -Wno-attributes # -fno-rtti -fno-exceptions
SEQ_FLAGS = -std=c++17 -O0 -g -I $(SKEPU_INCLUDE) $(CUTLASS_INCLUDE) -Wno-attributes # -fno-rtti -fno-exceptions

# ---------------------------------------------------------------------------- #
# Make recipies begin here.

# Builds all test programs, sequential implementation, witout using the precompiler.
all-sequential: $(TESTS)

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

all-cluster: $(CLUSTER_TESTS)

all-run: $(RUN_TESTS)


all: all-sequential all-parallel all-cluster

$(PAR_DIR):
	@mkdir -p $(PAR_DIR)

$(SEQ_DIR):
	@mkdir -p $(SEQ_DIR)

$(CLUSTER_DIR):
	@mkdir -p $(CLUSTER_DIR)


# Builds a single sequential test program.
# $(SEQ_DIR)/% : %.cpp $(SEQ_DIR)
# 	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) $< -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/%: %.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name $*_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o $@


# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	$(RM) -R $(SEQ_DIR)
	$(RM) -R $(PAR_DIR)
	$(RM) -R $(CLUSTER_DIR)
