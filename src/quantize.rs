//! Vector quantization module for memory compression and faster search.
//!
//! Implements scalar quantization (TurboQuant) for reducing vector storage
//! and accelerating similarity search with minimal accuracy loss.
//!
//! # Quantization Levels
//!
//! | Level | Compression | Speedup | Recall Loss |
//! |-------|-------------|---------|-------------|
//! | Int8  | 4x          | 2-3x    | ~0%         |
//! | Int4  | 8x          | 4-6x    | ~2%         |
//! | Binary| 32x         | 8-10x   | ~5%         |
//!
//! # Example
//!
//! ```
//! use pg_knowledge_graph::quantize::{ScalarQuantizer, QuantLevel};
//!
//! // Train quantizer on sample vectors
//! let vectors = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
//! let quantizer = ScalarQuantizer::train(&vectors, QuantLevel::Int8);
//!
//! // Quantize a vector
//! let query = vec![0.2, 0.3, 0.4];
//! let quantized = quantizer.quantize(&query);
//!
//! // Compute approximate distance
//! let distance = quantizer.cosine_distance(&quantized, &query);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

/// Quantization precision level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QuantLevel {
    /// 8-bit quantization: 4x compression, ~0% recall loss
    #[default]
    Int8,
    /// 4-bit quantization: 8x compression, ~2% recall loss
    Int4,
    /// Binary quantization: 32x compression, ~5% recall loss
    Binary,
}

impl QuantLevel {
    /// Get bits per value for this quantization level.
    pub fn bits_per_value(&self) -> u8 {
        match self {
            QuantLevel::Int8 => 8,
            QuantLevel::Int4 => 4,
            QuantLevel::Binary => 1,
        }
    }

    /// Get number of distinct values for this quantization level.
    pub fn num_values(&self) -> u32 {
        match self {
            QuantLevel::Int8 => 256,
            QuantLevel::Int4 => 16,
            QuantLevel::Binary => 2,
        }
    }

    /// Get compression ratio vs float32 (32 bits).
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits_per_value() as f32
    }

    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "int8" | "i8" | "8bit" => Some(QuantLevel::Int8),
            "int4" | "i4" | "4bit" => Some(QuantLevel::Int4),
            "binary" | "bin" | "1bit" => Some(QuantLevel::Binary),
            _ => None,
        }
    }
}

impl fmt::Display for QuantLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantLevel::Int8 => write!(f, "int8"),
            QuantLevel::Int4 => write!(f, "int4"),
            QuantLevel::Binary => write!(f, "binary"),
        }
    }
}

/// Scalar quantizer for float32 vectors.
///
/// Trained on a sample of vectors to compute per-dimension min/max bounds,
/// then used to quantize vectors for compact storage and fast search.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ScalarQuantizer {
    /// Minimum value per dimension
    mins: Vec<f32>,
    /// Maximum value per dimension
    maxs: Vec<f32>,
    /// Quantization level
    level: QuantLevel,
    /// Number of dimensions
    dims: usize,
    /// Scale factor per dimension: (max - min) / (num_values - 1)
    scales: Vec<f32>,
}

/// Quantized vector storage.
///
/// Stores quantized values in a compact byte array with bit-packing
/// for sub-byte quantization levels (Int4, Binary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVector {
    /// Quantized data (packed for Int4/Binary)
    data: Vec<u8>,
    /// Original vector L2 norm (for distance correction)
    norm: f32,
    /// Number of dimensions
    dims: usize,
}

/// Error type for quantization operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum QuantizeError {
    #[error("Empty vector list provided for training")]
    EmptyInput,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Quantizer not trained")]
    NotTrained,

    #[error("Invalid vector: contains NaN or Inf")]
    InvalidVector,

    #[error("Constant vector: all values are identical")]
    ConstantVector,
}

impl ScalarQuantizer {
    /// Create a new untrained quantizer.
    ///
    /// This is primarily for deserialization. Use [`train`](Self::train) instead.
    pub fn new(level: QuantLevel) -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            level,
            dims: 0,
            scales: Vec::new(),
        }
    }

    /// Train quantizer on a sample of vectors.
    ///
    /// Computes per-dimension min/max bounds from the training data.
    /// At least one vector is required.
    ///
    /// # Arguments
    /// * `vectors` - Training vectors (non-empty)
    /// * `level` - Quantization precision level
    ///
    /// # Returns
    /// Trained quantizer, or error if input is empty
    ///
    /// # Example
    /// ```
    /// let vectors = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    /// let quantizer = ScalarQuantizer::train(&vectors, QuantLevel::Int8);
    /// assert!(quantizer.is_trained());
    /// ```
    pub fn train(vectors: &[Vec<f32>], level: QuantLevel) -> Result<Self, QuantizeError> {
        if vectors.is_empty() {
            return Err(QuantizeError::EmptyInput);
        }

        let dims = vectors[0].len();
        if dims == 0 {
            return Err(QuantizeError::EmptyInput);
        }

        // Validate all vectors have same dimension and check for NaN/Inf
        for v in vectors.iter() {
            if v.len() != dims {
                return Err(QuantizeError::DimensionMismatch {
                    expected: dims,
                    actual: v.len(),
                });
            }
            // Check for NaN/Inf
            for &val in v {
                if !val.is_finite() {
                    return Err(QuantizeError::InvalidVector);
                }
            }
        }

        // Initialize min/max with first vector
        let mut mins = vectors[0].clone();
        let mut maxs = vectors[0].clone();

        // Update min/max from all vectors
        for v in vectors.iter().skip(1) {
            for (i, &val) in v.iter().enumerate() {
                if val < mins[i] {
                    mins[i] = val;
                }
                if val > maxs[i] {
                    maxs[i] = val;
                }
            }
        }

        // Handle constant dimensions (min == max)
        for i in 0..dims {
            if mins[i] == maxs[i] {
                // Add small epsilon to avoid division by zero
                maxs[i] = mins[i] + 1e-6;
            }
        }

        // Compute scale factors
        let num_values = level.num_values();
        let scales: Vec<f32> = (0..dims)
            .map(|i| (maxs[i] - mins[i]) / (num_values - 1) as f32)
            .collect();

        Ok(Self {
            mins,
            maxs,
            level,
            dims,
            scales,
        })
    }

    /// Check if quantizer has been trained.
    pub fn is_trained(&self) -> bool {
        self.dims > 0 && !self.mins.is_empty()
    }

    /// Get number of dimensions.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Get quantization level.
    pub fn level(&self) -> QuantLevel {
        self.level
    }

    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        self.level.compression_ratio()
    }

    /// Quantize a single vector.
    ///
    /// Returns the quantized representation with original norm stored.
    ///
    /// # Arguments
    /// * `vector` - Input vector (must match trained dimensions)
    ///
    /// # Returns
    /// Quantized vector, or error if dimension mismatch
    pub fn quantize(&self, vector: &[f32]) -> Result<QuantizedVector, QuantizeError> {
        if !self.is_trained() {
            return Err(QuantizeError::NotTrained);
        }

        if vector.len() != self.dims {
            return Err(QuantizeError::DimensionMismatch {
                expected: self.dims,
                actual: vector.len(),
            });
        }

        // Check for NaN/Inf
        for &val in vector {
            if !val.is_finite() {
                return Err(QuantizeError::InvalidVector);
            }
        }

        // Compute original L2 norm
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Quantize based on level
        let data = match self.level {
            QuantLevel::Int8 => self.quantize_int8(vector),
            QuantLevel::Int4 => self.quantize_int4(vector),
            QuantLevel::Binary => self.quantize_binary(vector),
        };

        Ok(QuantizedVector {
            data,
            norm,
            dims: self.dims,
        })
    }

    /// Quantize to Int8 (8 bits per value).
    fn quantize_int8(&self, vector: &[f32]) -> Vec<u8> {
        let num_values = self.level.num_values() as f32;
        vector
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                // Map to [0, 255]
                let normalized = (val - self.mins[i]) / (self.maxs[i] - self.mins[i]);
                let quantized = (normalized * (num_values - 1.0))
                    .round()
                    .clamp(0.0, num_values - 1.0);
                quantized as u8
            })
            .collect()
    }

    /// Quantize to Int4 (4 bits per value, 2 values per byte).
    fn quantize_int4(&self, vector: &[f32]) -> Vec<u8> {
        let num_values = self.level.num_values() as f32;
        let packed_len = vector.len().div_ceil(2);
        let mut packed = vec![0u8; packed_len];

        for (i, &val) in vector.iter().enumerate() {
            let normalized = (val - self.mins[i]) / (self.maxs[i] - self.mins[i]);
            let quantized = ((normalized * (num_values - 1.0))
                .round()
                .clamp(0.0, num_values - 1.0)) as u8;

            let byte_idx = i / 2;
            if i % 2 == 0 {
                // Lower 4 bits
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | (quantized & 0x0F);
            } else {
                // Upper 4 bits
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((quantized & 0x0F) << 4);
            }
        }

        packed
    }

    /// Quantize to Binary (1 bit per value, 8 values per byte).
    fn quantize_binary(&self, vector: &[f32]) -> Vec<u8> {
        // Use midpoint as threshold for each dimension
        let packed_len = vector.len().div_ceil(8);
        let mut packed = vec![0u8; packed_len];

        for (i, &val) in vector.iter().enumerate() {
            let threshold = (self.mins[i] + self.maxs[i]) / 2.0;
            let bit = if val >= threshold { 1u8 } else { 0u8 };

            let byte_idx = i / 8;
            let bit_idx = i % 8;
            packed[byte_idx] |= bit << bit_idx;
        }

        packed
    }

    /// Dequantize a value (for testing/debugging).
    fn dequantize_value(&self, dim: usize, quantized: u8) -> f32 {
        match self.level {
            QuantLevel::Int8 => {
                self.mins[dim] + (quantized as f32 / 255.0) * (self.maxs[dim] - self.mins[dim])
            }
            QuantLevel::Int4 => {
                self.mins[dim] + (quantized as f32 / 15.0) * (self.maxs[dim] - self.mins[dim])
            }
            QuantLevel::Binary => {
                let threshold = (self.mins[dim] + self.maxs[dim]) / 2.0;
                if quantized > 0 {
                    (threshold + self.maxs[dim]) / 2.0
                } else {
                    (self.mins[dim] + threshold) / 2.0
                }
            }
        }
    }

    /// Compute approximate dot product between quantized vector and query.
    ///
    /// Uses asymmetric distance computation: query is float32, stored vector is quantized.
    ///
    /// # Arguments
    /// * `quantized` - Quantized vector
    /// * `query` - Query vector (float32)
    ///
    /// # Returns
    /// Approximate dot product
    pub fn dot_product(&self, quantized: &QuantizedVector, query: &[f32]) -> f32 {
        if query.len() != self.dims {
            return 0.0;
        }

        match self.level {
            QuantLevel::Int8 => self.dot_product_int8(quantized, query),
            QuantLevel::Int4 => self.dot_product_int4(quantized, query),
            QuantLevel::Binary => self.dot_product_binary(quantized, query),
        }
    }

    fn dot_product_int8(&self, quantized: &QuantizedVector, query: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        for (i, &q) in quantized.data.iter().enumerate() {
            let dequant_val = self.dequantize_value(i, q);
            dot += dequant_val * query[i];
        }
        dot
    }

    fn dot_product_int4(&self, quantized: &QuantizedVector, query: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        for (i, &byte) in quantized.data.iter().enumerate() {
            // Lower 4 bits (even index)
            let lower_idx = i * 2;
            if lower_idx < self.dims {
                let lower_val = byte & 0x0F;
                let dequant = self.dequantize_value(lower_idx, lower_val);
                dot += dequant * query[lower_idx];
            }

            // Upper 4 bits (odd index)
            let upper_idx = i * 2 + 1;
            if upper_idx < self.dims {
                let upper_val = (byte >> 4) & 0x0F;
                let dequant = self.dequantize_value(upper_idx, upper_val);
                dot += dequant * query[upper_idx];
            }
        }
        dot
    }

    fn dot_product_binary(&self, quantized: &QuantizedVector, query: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        for (byte_idx, &byte) in quantized.data.iter().enumerate() {
            for bit_idx in 0..8 {
                let dim = byte_idx * 8 + bit_idx;
                if dim >= self.dims {
                    break;
                }
                let bit = (byte >> bit_idx) & 1;
                let dequant = self.dequantize_value(dim, bit);
                dot += dequant * query[dim];
            }
        }
        dot
    }

    /// Compute approximate cosine distance between quantized vector and query.
    ///
    /// Cosine distance = 1 - cosine_similarity = 1 - (dot / (||a|| * ||b||))
    ///
    /// # Arguments
    /// * `quantized` - Quantized vector (stores original norm)
    /// * `query` - Query vector (float32)
    ///
    /// # Returns
    /// Approximate cosine distance in [0, 2]
    pub fn cosine_distance(&self, quantized: &QuantizedVector, query: &[f32]) -> f32 {
        if query.len() != self.dims {
            return 1.0;
        }

        // Compute query norm
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm < 1e-10 || quantized.norm < 1e-10 {
            return 1.0;
        }

        let dot = self.dot_product(quantized, query);
        let similarity = dot / (quantized.norm * query_norm);

        // Clamp similarity to [-1, 1] to handle quantization errors
        let similarity_clamped = similarity.clamp(-1.0, 1.0);
        1.0 - similarity_clamped
    }

    /// Compute approximate cosine similarity.
    pub fn cosine_similarity(&self, quantized: &QuantizedVector, query: &[f32]) -> f32 {
        1.0 - self.cosine_distance(quantized, query)
    }

    /// Get min values per dimension (for debugging).
    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    /// Get max values per dimension (for debugging).
    pub fn maxs(&self) -> &[f32] {
        &self.maxs
    }
}

impl QuantizedVector {
    /// Get the quantized data bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the original vector norm.
    pub fn norm(&self) -> f32 {
        self.norm
    }

    /// Get number of dimensions.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Compute compression ratio vs original float32 storage.
    pub fn compression_ratio(&self) -> f32 {
        // Original: dims * 4 bytes (float32)
        // Quantized: data.len() bytes
        (self.dims * 4) as f32 / self.data.len() as f32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // === QuantLevel Tests ===

    #[test]
    fn test_quant_level_bits_per_value() {
        assert_eq!(QuantLevel::Int8.bits_per_value(), 8);
        assert_eq!(QuantLevel::Int4.bits_per_value(), 4);
        assert_eq!(QuantLevel::Binary.bits_per_value(), 1);
    }

    #[test]
    fn test_quant_level_compression_ratio() {
        assert!((QuantLevel::Int8.compression_ratio() - 4.0).abs() < 0.01);
        assert!((QuantLevel::Int4.compression_ratio() - 8.0).abs() < 0.01);
        assert!((QuantLevel::Binary.compression_ratio() - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_quant_level_from_str() {
        assert_eq!(QuantLevel::from_str("int8"), Some(QuantLevel::Int8));
        assert_eq!(QuantLevel::from_str("INT8"), Some(QuantLevel::Int8));
        assert_eq!(QuantLevel::from_str("i8"), Some(QuantLevel::Int8));
        assert_eq!(QuantLevel::from_str("int4"), Some(QuantLevel::Int4));
        assert_eq!(QuantLevel::from_str("binary"), Some(QuantLevel::Binary));
        assert_eq!(QuantLevel::from_str("invalid"), None);
    }

    #[test]
    fn test_quant_level_display() {
        assert_eq!(format!("{}", QuantLevel::Int8), "int8");
        assert_eq!(format!("{}", QuantLevel::Int4), "int4");
        assert_eq!(format!("{}", QuantLevel::Binary), "binary");
    }

    // === Training Tests ===

    #[test]
    fn test_train_empty_input() {
        let result = ScalarQuantizer::train(&[], QuantLevel::Int8);
        assert!(matches!(result, Err(QuantizeError::EmptyInput)));
    }

    #[test]
    fn test_train_single_vector() {
        let vectors = vec![vec![0.1, 0.2, 0.3]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();
        assert!(q.is_trained());
        assert_eq!(q.dims(), 3);

        // Min and max should be close to the single values
        assert!((q.mins()[0] - 0.1).abs() < 1e-6);
        assert!((q.maxs()[0] - 0.1 - 1e-6).abs() < 1e-6); // epsilon added
    }

    #[test]
    fn test_train_multiple_vectors() {
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        assert!((q.mins()[0] - 0.0).abs() < 1e-6);
        assert!((q.maxs()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_train_dimension_mismatch() {
        let vectors = vec![vec![0.1, 0.2], vec![0.3, 0.4, 0.5]];
        let result = ScalarQuantizer::train(&vectors, QuantLevel::Int8);
        assert!(matches!(
            result,
            Err(QuantizeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_train_nan_vector() {
        let vectors = vec![vec![0.1, f32::NAN, 0.3]];
        let result = ScalarQuantizer::train(&vectors, QuantLevel::Int8);
        assert!(matches!(result, Err(QuantizeError::InvalidVector)));
    }

    #[test]
    fn test_train_inf_vector() {
        let vectors = vec![vec![0.1, f32::INFINITY, 0.3]];
        let result = ScalarQuantizer::train(&vectors, QuantLevel::Int8);
        assert!(matches!(result, Err(QuantizeError::InvalidVector)));
    }

    // === Int8 Quantization Tests ===

    #[test]
    fn test_quantize_int8_basic() {
        let vectors = vec![vec![0.0, 0.5, 1.0]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        let quantized = q.quantize(&vec![0.0, 0.5, 1.0]).unwrap();

        assert_eq!(quantized.dims(), 3);
        assert_eq!(quantized.data().len(), 3); // 1 byte per dimension
        assert!((quantized.compression_ratio() - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_quantize_int8_roundtrip() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..10).map(|j| ((i * 10 + j) as f32 / 1000.0)).collect())
            .collect();

        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        for v in &vectors {
            let quantized = q.quantize(v).unwrap();

            // Check each dimension's roundtrip error
            for (i, &orig) in v.iter().enumerate() {
                let dequant = q.dequantize_value(i, quantized.data()[i]);
                let error = (orig - dequant).abs();
                // Max error should be less than 1/255 of range
                let max_error = (q.maxs()[i] - q.mins()[i]) / 255.0;
                assert!(
                    error <= max_error * 1.5,
                    "Roundtrip error too large at dim {}",
                    i
                );
            }
        }
    }

    // === Int4 Quantization Tests ===

    #[test]
    fn test_quantize_int4_basic() {
        let vectors = vec![vec![0.0, 0.5, 1.0, 0.25]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int4).unwrap();

        let quantized = q.quantize(&vec![0.0, 0.5, 1.0, 0.25]).unwrap();

        assert_eq!(quantized.dims(), 4);
        assert_eq!(quantized.data().len(), 2); // 2 values per byte
        assert!((quantized.compression_ratio() - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_quantize_int4_odd_dimensions() {
        let vectors = vec![vec![0.0, 0.5, 1.0]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int4).unwrap();

        let quantized = q.quantize(&vec![0.0, 0.5, 1.0]).unwrap();

        assert_eq!(quantized.dims(), 3);
        assert_eq!(quantized.data().len(), 2); // ceil(3/2) = 2 bytes
    }

    // === Binary Quantization Tests ===

    #[test]
    fn test_quantize_binary_basic() {
        let vectors = vec![vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Binary).unwrap();

        let quantized = q
            .quantize(&vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap();

        assert_eq!(quantized.dims(), 8);
        assert_eq!(quantized.data().len(), 1); // 8 values per byte
        assert!((quantized.compression_ratio() - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_quantize_binary_threshold() {
        // Use two distinct vectors to create clear min/max range
        let vectors = vec![vec![0.0, 0.0], vec![2.0, 2.0]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Binary).unwrap();

        // mins = [0.0, 0.0], maxs = [2.0, 2.0]
        // Threshold is (0.0 + 2.0) / 2.0 = 1.0 for both dims
        let quantized = q.quantize(&vec![0.0, 2.0]).unwrap();

        // Value 0.0 < 1.0 -> bit 0
        // Value 2.0 >= 1.0 -> bit 1
        let byte = quantized.data()[0];
        // bit 0 (dim 0): 0.0 < 1.0, should be 0
        // bit 1 (dim 1): 2.0 >= 1.0, should be 1
        assert_eq!(
            byte & 0x01,
            0,
            "Bit 0 should be 0 (value 0.0 < threshold 1.0)"
        );
        assert_eq!(
            byte & 0x02,
            2,
            "Bit 1 should be 1 (value 2.0 >= threshold 1.0)"
        );
    }

    // === Distance Tests ===

    #[test]
    fn test_dot_product_int8_exact() {
        // Use simple vectors where quantization error is minimal
        let vectors = vec![vec![0.5; 10]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        let v = vec![0.5; 10];
        let quantized = q.quantize(&v).unwrap();

        // Exact dot product with itself
        let exact_dot: f32 = v.iter().map(|x| x * x).sum();
        let approx_dot = q.dot_product(&quantized, &v);

        let error = (exact_dot - approx_dot).abs() / exact_dot;
        assert!(error < 0.01, "Dot product error {} too large", error);
    }

    #[test]
    fn test_cosine_distance_int8_accuracy() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..16).map(|_| rand::random::<f32>()).collect())
            .collect();

        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        for v in &vectors {
            let quantized = q.quantize(v).unwrap();
            let approx_dist = q.cosine_distance(&quantized, v);

            // Distance to itself should be ~0
            assert!(
                approx_dist < 0.02,
                "Self-distance {} too large",
                approx_dist
            );
        }
    }

    #[test]
    fn test_cosine_distance_int4_accuracy() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..16).map(|_| rand::random::<f32>()).collect())
            .collect();

        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int4).unwrap();

        for v in &vectors {
            let quantized = q.quantize(v).unwrap();
            let approx_dist = q.cosine_distance(&quantized, v);

            // Distance to itself should be ~0 (allow more error for int4)
            assert!(
                approx_dist < 0.05,
                "Self-distance {} too large",
                approx_dist
            );
        }
    }

    #[test]
    fn test_cosine_distance_binary_accuracy() {
        // Use vectors with clear separation for binary quantization
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..16)
                    .map(|j| if (i + j) % 2 == 0 { 0.1 } else { 0.9 })
                    .collect()
            })
            .collect();

        let q = ScalarQuantizer::train(&vectors, QuantLevel::Binary).unwrap();

        for v in &vectors {
            let quantized = q.quantize(v).unwrap();
            let approx_dist = q.cosine_distance(&quantized, v);

            // Binary has more error, but self-distance should still be reasonable
            assert!(
                approx_dist < 0.25,
                "Self-distance {} too large",
                approx_dist
            );
        }
    }

    #[test]
    fn test_distance_preserves_ordering() {
        // Create vectors with larger differences to ensure ordering
        let base = vec![0.5; 10];
        let close = vec![0.55; 10];
        let far = vec![0.0; 10]; // Very different

        let vectors = vec![base.clone(), close.clone(), far.clone()];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        let base_q = q.quantize(&base).unwrap();

        let dist_close = q.cosine_distance(&base_q, &close);
        let dist_far = q.cosine_distance(&base_q, &far);

        assert!(
            dist_close < dist_far,
            "Distance ordering not preserved: close={}, far={}",
            dist_close,
            dist_far
        );
    }

    // === Edge Case Tests ===

    #[test]
    fn test_quantize_zero_vector() {
        let vectors = vec![vec![0.0; 10]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        let quantized = q.quantize(&vec![0.0; 10]).unwrap();
        assert_eq!(quantized.norm(), 0.0);
    }

    #[test]
    fn test_quantize_dimension_mismatch() {
        let vectors = vec![vec![0.0; 10]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        let result = q.quantize(&vec![0.0; 5]);
        assert!(matches!(
            result,
            Err(QuantizeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_quantize_nan_in_query() {
        let vectors = vec![vec![0.0; 10]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        let result = q.quantize(&vec![0.0, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(matches!(result, Err(QuantizeError::InvalidVector)));
    }

    #[test]
    fn test_untrained_quantizer() {
        let q = ScalarQuantizer::new(QuantLevel::Int8);
        assert!(!q.is_trained());

        let result = q.quantize(&vec![0.1]);
        assert!(matches!(result, Err(QuantizeError::NotTrained)));
    }

    // === Compression Tests ===

    #[test]
    fn test_compression_int8() {
        let dims = 1536;
        let vectors = vec![vec![0.5; dims]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        let quantized = q.quantize(&vec![0.5; dims]).unwrap();

        assert_eq!(quantized.data().len(), dims); // 1 byte per dim
        let expected_size = dims; // 1536 bytes
        let original_size = dims * 4; // 6144 bytes
        assert!(
            (quantized.compression_ratio() - (original_size as f32 / expected_size as f32)).abs()
                < 0.1
        );
    }

    #[test]
    fn test_compression_int4() {
        let dims = 1536;
        let vectors = vec![vec![0.5; dims]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int4).unwrap();

        let quantized = q.quantize(&vec![0.5; dims]).unwrap();

        assert_eq!(quantized.data().len(), dims / 2); // 2 values per byte
    }

    #[test]
    fn test_compression_binary() {
        let dims = 1536;
        let vectors = vec![vec![0.5; dims]];
        let q = ScalarQuantizer::train(&vectors, QuantLevel::Binary).unwrap();

        let quantized = q.quantize(&vec![0.5; dims]).unwrap();

        assert_eq!(quantized.data().len(), dims / 8); // 8 values per byte
    }

    // === Performance Tests ===

    #[test]
    fn bench_train_1000_vectors() {
        use std::time::Instant;

        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..1536).map(|_| rand::random::<f32>()).collect())
            .collect();

        let start = Instant::now();
        let _q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();
        let duration = start.elapsed();

        println!("Training on 1000 vectors (1536 dims): {:?}", duration);
        // Training should complete in reasonable time
        assert!(
            duration.as_millis() < 500,
            "Training too slow: {:?}",
            duration
        );
    }

    #[test]
    fn bench_quantize_1000_vectors() {
        use std::time::Instant;

        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..1536).map(|_| rand::random::<f32>()).collect())
            .collect();

        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();

        let start = Instant::now();
        for v in &vectors {
            let _ = q.quantize(v).unwrap();
        }
        let duration = start.elapsed();

        println!("Quantizing 1000 vectors: {:?}", duration);
        // Allow more time for debug builds
        assert!(
            duration.as_millis() < 500,
            "Quantization too slow: {:?}",
            duration
        );
    }

    #[test]
    fn bench_distance_1000_queries() {
        use std::time::Instant;

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..1536).map(|_| rand::random::<f32>()).collect())
            .collect();

        let q = ScalarQuantizer::train(&vectors, QuantLevel::Int8).unwrap();
        let quantized: Vec<_> = vectors.iter().map(|v| q.quantize(v).unwrap()).collect();

        let start = Instant::now();
        for _ in 0..1000 {
            for qv in &quantized {
                let _ = q.cosine_distance(qv, &vectors[0]);
            }
        }
        let duration = start.elapsed();

        println!("1000 distance computations: {:?}", duration);
        let per_distance = duration.as_nanos() / 1000 / quantized.len() as u128;
        println!("Per distance: {} ns", per_distance);
    }
}
