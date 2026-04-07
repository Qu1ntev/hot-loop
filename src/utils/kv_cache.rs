use std::ops::{Deref, DerefMut};
use candle_core::{Tensor, Error};

#[doc(hidden)]
pub struct KvCache(Vec<ConcatKvCache>);

impl KvCache {
    /// Create KvCache
    /// `dim=2` `[batch, heads, seq, head_dim]`
    /// `dim=1` `[batch, seq, heads, head_dim]`
    pub fn new(len: usize, dim: usize) -> Self {
        let mut kv_cache = Vec::with_capacity(len);
        for _ in 0..len { kv_cache.push(ConcatKvCache::new(dim)); }
        Self(kv_cache)
    }

    /// Clear all stored keys and values
    pub fn clear(&mut self) {
        for cache in &mut self.0 {
            cache.k = None;
            cache.v = None;
        }
    }

    /// Truncate: KvCache[..index]
    pub fn truncate(&mut self, index: usize) -> Result<(), Error> {
        let current = self.current_pos();

        if index >= current {
            return Ok(());
        }

        if index == 0 {
            self.clear();
            return Ok(());
        }

        for cache in &mut self.0 {
            if let Some(k_cache) = &cache.k {
                cache.k = Some(k_cache.narrow(cache.dim, 0, index)?.contiguous()?);
            }
            if let Some(v_cache) = &cache.v {
                cache.v = Some(v_cache.narrow(cache.dim, 0, index)?.contiguous()?);
            }
        }
        Ok(())
    }

    /// Get current sequence length in the cache.
    /// Returns 0 if the cache is empty.
    pub fn current_pos(&self) -> usize {
        match self.0.get(0) {
            Some(cache) => {
                cache.k.as_ref()
                    .and_then(|k| k.dims().get(cache.dim).copied())
                    .unwrap_or(0)
            },
            None => 0,
        }
    }
}

impl Deref for KvCache {
    type Target = Vec<ConcatKvCache>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for KvCache {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[doc(hidden)]
pub struct ConcatKvCache {
    pub(super) k: Option<Tensor>,
    pub(super) v: Option<Tensor>,
    pub(super) dim: usize,
}

impl ConcatKvCache {
    pub(super) fn new(dim: usize) -> Self {
        Self {
            k: None,
            v: None,
            dim,
        }
    }

    /// Append key and value tensors to the cache.
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
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use super::*;

    #[test]
    fn test() -> Result<(), Error> {
        let mut kv_cache = KvCache::new(10, 2);

        let tensor1 = Tensor::new(&[[[ 45f32, 1.0 ]]], &Device::Cpu)?;
        let tensor2 = Tensor::new(&[[[ 45f32, 1.0 ]]], &Device::Cpu)?;

        for cache in kv_cache.iter_mut() {
            cache.append(&tensor1, &tensor2)?;
        }

        assert_eq!(kv_cache.current_pos(), 2);

        kv_cache.truncate(2)?;
        assert_eq!(kv_cache.current_pos(), 2);

        kv_cache.truncate(1)?;
        assert_eq!(kv_cache.current_pos(), 1);

        kv_cache.clear();
        assert_eq!(kv_cache.current_pos(), 0);

        Ok(())
    }
}