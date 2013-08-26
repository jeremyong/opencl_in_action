PROJ=read_mm

CC=gcc

CFLAGS=-std=c99 -Wall -DUNIX -g -DDEBUG

INC_DIRS=.

$(PROJ): $(PROJ).c mmio.c
	$(CC) $(CFLAGS) -o $@ $^ $(INC_DIRS:%=-I%)

.PHONY: clean

clean:
	rm $(PROJ)
