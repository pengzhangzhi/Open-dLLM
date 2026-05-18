#!/usr/bin/env bash
# Run after: sudo apt-get install -y libaio-dev
set -e
cd "$(dirname "$0")/.."
echo "Checking libaio..."
dpkg -l libaio-dev | grep "^ii" || { echo "ERROR: libaio-dev not installed"; exit 1; }

echo "Building DeepSpeed async_io extension..."
DS_BUILD_AIO=1 .venv/bin/python -c "
from deepspeed.ops.op_builder import AsyncIOBuilder
aio = AsyncIOBuilder()
print('Compatible:', aio.is_compatible(verbose=True))
aio.load(verbose=True)
print('async_io built and loaded OK')
"
echo "Done. DeepSpeed NVMe offload is now available."
