#!/bin/bash
# Build libkv_vmm.so beside this script (the patcher copies it into the tree).
# usage: kv-vmm/build.sh   (CUDA_HOME, default /usr/local/cuda)
set -e
here="$(cd "$(dirname "$0")" && pwd)"
cuda="${CUDA_HOME:-/usr/local/cuda}"
gcc -O2 -Wall -shared -fPIC "$here/kv_vmm.c" -I"$cuda/include" -L"$cuda/lib64/stubs" -lcuda -o "$here/libkv_vmm.so.new"
mv "$here/libkv_vmm.so.new" "$here/libkv_vmm.so"
echo "built $here/libkv_vmm.so"
