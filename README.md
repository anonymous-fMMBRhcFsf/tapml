# TapML

## Installation

### Set up build dependency

First clone the repository and navigate to the root directory of the repository.

```bash
git clone --recursive https://github.com/anonymous-fMMBRhcFsf/tapml.git && cd tapml/
```

> [!NOTE]
> If you have already cloned the repository, you can update the submodules by running:
>
> ```bash
> git submodule update --init --recursive
> ```

You need to ensure that the following build dependencies are satisfied:

- CMake >= 3.24
- LLVM >= 15
- Git
- Rust and Cargo, required by Hugging Face’s tokenizer
- One of the GPU runtime:
  - CUDA >= 12.0 (NVIDIA GPUs)
  - Metal (Apple GPUs)
  - Vulkan (NVIDIA, AMD)

```bash
# start with a fresh environment
conda env remove -n tapml-venv
# create the conda environment with build dependency
conda create -n tapml-venv -c conda-forge \
    "cmake>=3.24" \
    "llvmdev>=15" \
    rust \
    git \
    python=3.11
# enter the build environment
conda activate tapml-venv
```

### Install Apache TVM

#### Configure and build TVM

```bash
# navigate to the TVM directory
cd 3rdparty/tvm
# create the build directory
mkdir build && cd build
# specify build requirements in `config.cmake`
cp ../cmake/config.cmake .
```

We want to specifically tweak the following flags by appending them to the end of the configuration file:

```bash
# controls default compilation flags
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
# LLVM is a must dependency
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
# GPU SDKs, turn on if needed
echo "set(USE_CUDA   OFF)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake
# FlashInfer related, requires CUDA w/ compute capability 80;86;89;90
echo "set(USE_FLASHINFER OFF)" >> config.cmake
echo "set(FLASHINFER_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake
echo "set(CMAKE_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake
```

> [!NOTE]
> If you are using CUDA and your compute capability is above 80, then it is require to build with set`(USE_FLASHINFER ON)`. Otherwise, you may run into `Cannot find PackedFunc` issue during runtime.
>
> To check your CUDA compute capability, you can use `nvidia-smi --query-gpu=compute_cap --format=csv`.

Build libtvm using cmake and cmake¶

```bash
cmake .. && cmake --build . --parallel $(nproc)
```

A success build should produce `libtvm` and `libtvm_runtime` under `build/` directory.

#### Install Apache TVM via Python

```bash
cd ../python
pip install -e .
```

### Install TapML Python Package

#### Configure and build TapML

```bash
# create build directory
mkdir -p build && cd build
# generate build configuration
python3 ../cmake/gen_cmake_config.py
# build mlc_llm libraries
cmake .. && cmake --build . --parallel $(nproc) && cd ..
```

> [!NOTE]
> If you are using CUDA and your compute capability is above 80, then it is require to build with set`(USE_FLASHINFER ON)`. Otherwise, you may run into `Cannot find PackedFunc` issue during runtime.
>
> To check your CUDA compute capability, you can use `nvidia-smi --query-gpu=compute_cap --format=csv`.

#### Install TapML via Python

```bash
cd /path-to-tapml/python
pip install -e .
```

#### Verify installation

```bash
tapml chat -h
```

## Compile Models

### Clone from HF and convert_weight

```bash
# Create directory
mkdir -p dist/models && cd dist/models
# Clone HF weights
git lfs install
git clone https://huggingface.co/togethercomputer/Llama-2-7b-chat-hf
cd ../..
# Convert weight
tapml convert_weight ./dist/models/Llama-2-7b-chat-hf/ \
    --quantization q4f16_1 \
    -o dist/Llama-2-7b-chat-hf-q4f16_1
```

### Generate tapml-chat-config and compile

```bash
# Create output directory for the model library compiled
mkdir dist/libs

# 1. gen_config: generate tapml-chat-config.json and process tokenizers
tapml gen_config ./dist/models/Llama-2-7b-chat-hf/ \
    --quantization q4f16_1 --conv-template redpajama_chat \
    -o dist/Llama-2-7b-chat-hf-q4f16_1/
# 2. compile: compile model library with specification in tapml-chat-config.json
tapml compile ./dist/Llama-2-7b-chat-hf-q4f16_1/tapml-chat-config.json \
    --device cuda -o dist/libs/Llama-2-7b-chat-hf-q4f16_1-cuda.so
```

> [!NOTE]
> For other devices, you can replace `--device cuda` with `--device metal` or `--device vulkan`.

## Chat with Models

```bash
tapml chat dist/libs/Llama-2-7b-chat-hf-q4f16_1 --model-lib-path dist/libs/Llama-2-7b-chat-hf-q4f16_1-cuda.so
```
