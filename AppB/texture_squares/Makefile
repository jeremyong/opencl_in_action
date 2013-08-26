PROJ=texture_squares

CC=gcc

CFLAGS=-std=c99 -Wall -DUNIX -g -DDEBUG

LIBS=-lglut -lGLEW -lpng

$(PROJ): $(PROJ).c
	$(CC) $(CFLAGS) -o $@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)

.PHONY: clean

clean:
	rm $(PROJ)
