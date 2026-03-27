use std::ops::{Deref, DerefMut};
use candle_core::{Tensor, Error};

#[doc(hidden)]
pub struct KvCache {
    kv_cache: Vec<ConcatKvCache>
}

impl KvCache {
    pub fn new(len: usize, dim: usize) -> Self {
        let mut kv_cache = Vec::with_capacity(len);
        for _ in 0..len { kv_cache.push(ConcatKvCache::new(dim)); }
        Self { kv_cache }
    }

    pub fn clear(&mut self) {
        for cache in &mut self.kv_cache {
            cache.clear()
        }
    }

    pub fn truncate(&mut self, index: usize) -> Result<(), Error> {
        for cache in &mut self.kv_cache {
            cache.truncate(index)?;
        }
        Ok(())
    }

    pub fn current_pos(&self) -> Option<usize> {
        match self.kv_cache.get(0) {
            Some(cache) => Some(cache.current_seq_len()),
            None => None,
        }
    }
}

impl Deref for KvCache {
    type Target = Vec<ConcatKvCache>;
    fn deref(&self) -> &Self::Target {
        &self.kv_cache
    }
}

impl DerefMut for KvCache {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.kv_cache
    }
}

#[doc(hidden)]
pub struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
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
        }
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
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor), Error> {
        let mut k = k.contiguous()?;
        let mut v = v.contiguous()?;

        if let Some(k_cache) = &self.k {
            k = Tensor::cat(&[k_cache, &k], self.dim)?;
        }
        if let Some(v_cache) = &self.v {
            v = Tensor::cat(&[v_cache, &v], self.dim)?;
        }

        self.k = Some(k.clone());
        self.v = Some(v.clone());

        Ok((k, v))
    }

    /// truncate kv_cache: cache[..index]
    pub fn truncate(&mut self, index: usize) -> Result<(), Error> {
        let current = self.current_seq_len();

        if index >= current {
            return Ok(());
        }

        if index == 0 {
            self.clear();
            return Ok(());
        }

        if let Some(k) = &self.k {
            self.k = Some(k.narrow(self.dim, 0, index)?.contiguous()?);
        }
        if let Some(v) = &self.v {
            self.v = Some(v.narrow(self.dim, 0, index)?.contiguous()?);
        }

        Ok(())
    }

    /// Reset the cache (clear all stored keys and values)
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }
}