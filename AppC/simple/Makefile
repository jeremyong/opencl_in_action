PROJ=simple
CC=mingw32-gcc
CFLAGS=-std=c99 -Wall

$(PROJ): $(PROJ).c src1.o src2.o
	$(CC) $(CFLAGS) -o $@ $^
   
src1.o: src1.c
	$(CC) $(CFLAGS) -c -o $@ $^
   
src2.o: src2.c
	$(CC) $(CFLAGS) -c -o $@ $^

.PHONY: clean

clean:
	rm $(PROJ).exe *.o
