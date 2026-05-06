# Build the FFI shared library used by Bun runtime.
#
# Architecture detection: assumes the host the Dockerfile builder runs on is
# the same as where the runtime image runs (we don't cross-compile). amd64
# enables AVX2+FMA; arm64 (NEON) is on by default.

UNAME_M := $(shell uname -m)

CFLAGS := -O3 -fPIC -shared -Wall -Wextra -Wno-unused-parameter

ifeq ($(UNAME_M),x86_64)
  CFLAGS += -mavx2 -mfma -msse4.1
endif
ifeq ($(UNAME_M),amd64)
  CFLAGS += -mavx2 -mfma -msse4.1
endif
ifeq ($(UNAME_M),aarch64)
  # NEON is mandatory on AArch64.
endif
ifeq ($(UNAME_M),arm64)
  # macOS uname reports arm64; treat as aarch64.
endif

# Output extension: .so on Linux. (macOS .dylib not needed because we run
# inside Linux containers — local Bun tests can also use .so via Linux Bun
# but on macOS we'd want .dylib. For now we only support Linux container
# execution.)
TARGET := libsearch.so

all: $(TARGET)

$(TARGET): search.c
	$(CC) $(CFLAGS) -o $@ search.c

clean:
	rm -f $(TARGET)

.PHONY: all clean
