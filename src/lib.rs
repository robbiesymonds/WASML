use std::vec;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/* Subtracts one matrix from another. */
#[wasm_bindgen]
pub fn subtract_matrix(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    return a.iter().zip(b.iter()).map(|(&a, &b)| a - b).collect();
}

/* Subtracts a scalar value from the matrix. */
#[wasm_bindgen]
pub fn subtract_scalar(a: Vec<f64>, b: f64) -> Vec<f64> {
    return a.iter().map(|x| x - b).collect();
}

/* Adds to matrices together element-wise. */
#[wasm_bindgen]
pub fn add_matrix(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    return a.iter().zip(b.iter()).map(|(&a, &b)| a + b).collect();
}

/* Adds a scalar value to the matrix. */
#[wasm_bindgen]
pub fn add_scalar(a: Vec<f64>, b: f64) -> Vec<f64> {
    return a.iter().map(|x| x + b).collect();
}

/* Performs matrix multiplication. */
#[wasm_bindgen]
pub fn multiply(a: Vec<f64>, b: Vec<f64>, rows: usize, m: usize, n: usize) -> Vec<f64> {
    let mut result = vec![0.0; rows * m];

    for i in 0..rows {
        for j in 0..m {
            let mut sum: f64 = 0.0;
            for k in 0..n {
                let x: f64 = a[i * n + k];
                let y: f64 = b[k * m + j];
                sum += x * y;
            }
            result[i * m + j] = sum;
        }
    }

    return result;
}

/* Element-wise production of matrix and scalar. */
#[wasm_bindgen]
pub fn dot_matrix(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    return a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect();
}

/* Performs element-wise mutliplication on two matrices. */
#[wasm_bindgen]
pub fn dot_scalar(a: Vec<f64>, b: f64) -> Vec<f64> {
    return a.iter().map(|x| x * b).collect();
}

/* Calculates the transpose of the given matrix. */
#[wasm_bindgen]
pub fn transpose(a: Vec<f64>, rows: usize, columns: usize) -> Vec<f64> {
    let mut result: Vec<f64> = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..columns {
            result[(j * rows + i) as usize] = a[(i * columns + j) as usize];
        }
    }
    return result;
}

/* Element-wise sum of all matrix values. */
#[wasm_bindgen]
pub fn sum(a: Vec<f64>) -> f64 {
    return a.iter().sum();
}

/* Calculates the index of largest matrix element value. */
#[wasm_bindgen]
pub fn argmax(a: Vec<f64>) -> Option<usize> {
    return a
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index);
}
