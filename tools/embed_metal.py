"""Embed an offline-compiled Metal library; no runtime shader path or compiler."""
import pathlib
import sys

source, target = map(pathlib.Path, sys.argv[1:])
data = source.read_bytes()
with target.open("w") as out:
    out.write("// Generated from Kernels.metal; do not edit.\n")
    out.write("alignas(16) static const unsigned char METAL_LIBRARY[] = {\n")
    for i in range(0, len(data), 24):
        out.write(",".join(str(b) for b in data[i:i + 24]) + ",\n")
    out.write("};\n")
