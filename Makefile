SRC_DIR  = ./src
OBJ_DIR  = ./obj
LIB_DIR  = ./libs
BIN_DIR  = ./bin

CFILES   = $(wildcard $(SRC_DIR)/*.c)
CUFILES  = $(wildcard $(SRC_DIR)/*.cu)
LIBFILES = $(wildcard $(LIB_DIR)/*.c)
OBJFILES = $(CFILES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o) $(CUFILES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o) $(LIBFILES:$(LIB_DIR)/%.c=$(OBJ_DIR)/%.o)
OUT      = $(BIN_DIR)/fglt_aorl

CC  =  /usr/local/cuda/bin/nvcc
LDFLAGS = -lgomp

all: CFLAGS += -O3 -Xcompiler -fopenmp
all: $(OUT)

debug: CFLAGS += -DDEBUG --generate-line-info
debug: $(OUT)

$(OUT): $(OBJFILES)
	$(CC) $(LDFLAGS) -o $@ $^ 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(LIB_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean test run
run:
	./bin/fglt_aorl assets/mine.mtx

clean:
	rm -f $(OBJ_DIR)/* $(BIN_DIR)/*
