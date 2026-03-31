// src/lib.rs
// Main library entry point for Rust ML Kernels

pub mod kernels;
pub mod utils;

// Re-export commonly used functions for easier access
pub use kernels::matrix_multiplication::matrix_multiply;
pub use utils::{transpose_matrix, dot_product};

/// A simple function to demonstrate library usage.
pub fn greet_ml_engineer(name: &str) -> String {
    format!("Hello, {}! Welcome to Rust ML Kernels.", name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet_ml_engineer() {
        assert_eq!(greet_ml_engineer("David"), "Hello, David! Welcome to Rust ML Kernels.");
    }
}
