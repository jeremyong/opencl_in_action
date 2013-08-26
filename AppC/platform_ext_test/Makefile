PROJ=platform_ext_test
CC=mingw32-gcc
CFLAGS=-std=c99 -Wall
LIB=-lOpenCL

# Check for AMD
ifdef AMDAPPSDKROOT
   INC_DIRS="$(AMDAPPSDKROOT)include"
   LIB_DIRS="$(AMDAPPSDKROOT)lib\x86"
else

# Check for Nvidia
ifdef NVSDKCOMPUTE_ROOT
   INC_DIRS="$(NVSDKCOMPUTE_ROOT)\OpenCL\common\inc"
   LIB_DIRS="$(NVSDKCOMPUTE_ROOT)\OpenCL\common\lib\Win32"
endif

endif

$(PROJ): $(PROJ).c
	$(CC) $(CFLAGS) -o $@ $^ -I$(INC_DIRS) -L$(LIB_DIRS) $(LIB)

.PHONY: clean

clean:
	rm $(PROJ).exe