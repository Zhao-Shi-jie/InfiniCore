.PHONY: build clean

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))

build:
	mkdir -p build && cd build && cmake .. && make -j8

clean:
	rm -rf build
