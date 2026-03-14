use candle_core::{Tensor, Error};

#[derive(Debug, Clone)]
pub struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
    
    k_roll: Option<Tensor>,
    v_roll: Option<Tensor>,
    roll_len: usize
}

impl ConcatKvCache {
    /// Create a new empty concatenation-based KV-cache
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to concatenate
    ///   - For attention with shape `[batch, heads, seq, head_dim]`, use `dim=2`
    ///   - For attention with shape `[batch, seq, heads, head_dim]`, use `dim=1`
    ///
    /// # Example
    /// ```ignore
    /// // For standard transformer attention: [B, H, S, D]
    /// let cache = ConcatKvCache::new(2);
    /// ```
    pub fn new(dim: usize) -> Self {
        Self {
            k: None,
            v: None,
            dim,

            k_roll: None,
            v_roll: None,
            roll_len: 0
        }
    }
    
    pub fn set_rollback(&mut self) {
        self.k_roll = self.k.clone();
        self.v_roll = self.v.clone();

        self.roll_len = self.current_seq_len();
    }
    
    pub fn rollback(&mut self) {
        self.k = self.k_roll.clone();
        self.v = self.v_roll.clone();
    }

    fn clear_rollback(&mut self) {
        self.k_roll = None;
        self.v_roll = None;
        self.roll_len = 0;
    }

    /// Get current sequence length in the cache
    ///
    /// Returns 0 if the cache is empty.
    pub fn current_seq_len(&self) -> usize {
        self.k
            .as_ref()
            .and_then(|k| k.dims().get(self.dim).copied())
            .unwrap_or(0)
    }

    /// Append key and value tensors to the cache
    ///
    /// This is the core operation that uses optimized concatenation kernels.
    ///
    /// # Arguments
    /// * `k` - Key tensor to append (shape: [..., seq_len, ...])
    /// * `v` - Value tensor to append (shape: [..., seq_len, ...])
    ///
    /// # Returns
    /// Tuple of `(full_k, full_v)` containing all cached keys and values,
    /// including the newly appended data.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        // Ensure inputs are contiguous for optimal concatenation performance
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        self.k = Some(match &self.k {
            None => k.clone(),
            Some(k_cache) => Tensor::cat(&[k_cache, &k], self.dim)?
        });

        self.v = Some(match &self.v {
            None => v.clone(),
            Some(v_cache) => Tensor::cat(&[v_cache, &v], self.dim)?,
        });
        
        let k = self.k.as_ref().ok_or(Error::UnwrapNone)?.clone();
        let v = self.v.as_ref().ok_or(Error::UnwrapNone)?.clone();

        Ok((k, v))
    }

    /// Reset the cache (clear all stored keys and values)
    fn clear_kv(&mut self) {
        self.k = None;
        self.v = None;
    }

    pub fn reset_all(&mut self) {
        self.clear_kv();
        self.clear_rollback();
    }
}