// src/utils/utils.rs
// Utility functions for ML kernels

pub fn transpose_matrix(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }
    transposed
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_matrix() {
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let rows = 2;
        let cols = 3;
        let transposed = transpose_matrix(&matrix, rows, cols);
        // Expected 3x2 matrix: [[1,4],[2,5],[3,6]]
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }
}
