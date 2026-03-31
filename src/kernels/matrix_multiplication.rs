// src/kernels/matrix_multiplication.rs
// Optimized matrix multiplication kernel for ML workloads

pub fn matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // Assumes square matrices of size n x n
    // C = A * B
    // C[i][j] = sum(A[i][k] * B[k][j]) for k from 0 to n-1

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiply_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b = [5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]
        let mut c = [0.0; 4];
        let n = 2;

        matrix_multiply(&a, &b, &mut c, n);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[5+14, 6+16], [15+28, 18+32]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matrix_multiply_identity() {
        let a = [1.0, 0.0, 0.0, 1.0]; // Identity matrix
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        let n = 2;

        matrix_multiply(&a, &b, &mut c, n);
        assert_eq!(c, b);
    }

    #[test]
    fn test_matrix_multiply_zero() {
        let a = [0.0, 0.0, 0.0, 0.0]; // Zero matrix
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        let n = 2;

        matrix_multiply(&a, &b, &mut c, n);
        assert_eq!(c, [0.0, 0.0, 0.0, 0.0]);
    }
}
