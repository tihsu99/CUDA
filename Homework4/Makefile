################################################################################
#
# Build script for project
#
################################################################################

CXX     := icc
CFLAGS  := -O2
NVCC    := nvcc
#CUFLAGS := -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 --compiler-options -fno-strict-aliasing -DUNIX -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler -fopenmp -rdc=true

CUFLAGS := -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 --compiler-options -fopenmp
#CUFLAGS := -O3 --compiler-options -fopenmp

BIN     := vecDot_ngpu
OBJ     := vecDot_ngpu.o
LIB     :=

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

all: $(BIN)

$(BIN): $(OBJ)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
	$(NVCC) -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)

vecAdd_ngpu.o:  vecDot_ngpu.cu

BIN    := HeatDiff
OBJ    := HeatDiff.o

all:   $(BIN)
$(BIN): $(OBJ)
	$(NVCC) -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
HeatDiff.o: HeatDiff.cu

clean:
	rm -f $(OBJ) $(BIN)

