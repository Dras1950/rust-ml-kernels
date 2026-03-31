# Rust ML Kernels

Optimized CUDA/ROCm kernels written in Rust for accelerating transformer-based model training.

## Features
- **High Performance:** Leverages Rust's performance and safety features for efficient kernel execution.
- **GPU Acceleration:** Designed for NVIDIA CUDA and AMD ROCm platforms to maximize training speed.
- **Transformer-Friendly:** Specifically optimized for common operations in transformer architectures (e.g., attention mechanisms, feed-forward networks).
- **Seamless Integration:** Provides FFI (Foreign Function Interface) for easy integration with existing Python-based ML frameworks.

## Getting Started

### Prerequisites
- Rust toolchain (latest stable)
- NVIDIA CUDA Toolkit or AMD ROCm platform
- `cargo` (Rust's package manager)

### Installation

```bash
git clone https://github.com/Dras1950/rust-ml-kernels.git
cd rust-ml-kernels
cargo build --release
```

### Usage (Example with Python FFI)

```python
import ctypes

# Load the Rust shared library
lib = ctypes.CDLL("target/release/librust_ml_kernels.so")

# Example: Call a Rust function for matrix multiplication
# lib.matrix_multiply.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ...]
# lib.matrix_multiply.restype = ctypes.POINTER(ctypes.c_float)

# result = lib.matrix_multiply(matrix_a, matrix_b, ...)
# print(result)
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
