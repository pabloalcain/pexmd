.PHONY: all clean
CC = gcc
LD = gcc
LD_FLAGS = -shared
C_FLAGS = -g -O3 -fPIC -std=gnu99


all: box.so

%.so: %.o
	$(LD) $(LD_FLAGS) $^ -o $@

%.o: %.c
	$(CC) $(C_FLAGS) $^ -c

clean:
	rm -rfv box.so box.o
