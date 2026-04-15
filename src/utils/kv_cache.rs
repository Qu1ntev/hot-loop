use std::ops::{Deref, DerefMut};
use candle_core::{Tensor, Error, DType, Device};

#[doc(hidden)]
pub struct KvCache(Vec<PreallocKvCache>);

impl KvCache {
    /// Create KvCache
    /// `dim=2` `[batch, heads, seq, head_dim]`
    /// `dim=1` `[batch, seq, heads, head_dim]`
    pub fn new(
        layers_len: usize,
        head_dim: usize,
        max_size: usize,
        num_kv_heads: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, Error> {
        let mut kv_cache = Vec::with_capacity(layers_len);
        for _ in 0..layers_len {
            kv_cache.push(PreallocKvCache::new(
                num_kv_heads,
                head_dim,
                max_size,
                dtype,
                device,
            )?);
        }
        Ok(Self(kv_cache))
    }

    /// Clear all stored keys and values
    pub fn clear(&mut self) {
        for cache in &mut self.0 {
            cache.current_pos = 0;
        }
    }

    pub fn truncate(&mut self, index: usize) {
        let current = self.current_pos();

        if index >= current {
            return;
        }

        for cache in &mut self.0 {
            cache.current_pos = current;
        }
    }

    /// Get current sequence length in the cache.
    /// Returns 0 if the cache is empty.
    pub fn current_pos(&self) -> usize {
        match self.0.get(0) {
            Some(cache) => cache.current_pos,
            None => 0,
        }
    }
}

impl Deref for KvCache {
    type Target = Vec<PreallocKvCache>;
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
pub struct PreallocKvCache {
    k_buf: Tensor,
    v_buf: Tensor,
    current_pos: usize,
    max_seq_len: usize,
}

impl PreallocKvCache {
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, Error> {
        let shape = (1, num_kv_heads, max_seq_len, head_dim);
        let k_buf = Tensor::zeros(shape, dtype, device)?;
        let v_buf = Tensor::zeros(shape, dtype, device)?;
        Ok(Self {
            k_buf,
            v_buf,
            current_pos: 0,
            max_seq_len,
        })
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), Error> {
        let seq_len = new_k.dim(2)?;
        let end_pos = self.current_pos + seq_len;

        if end_pos > self.max_seq_len {
            candle_core::bail!("PreallocKvCache: sequence length {end_pos} exceeds max_seq_len {}. \
                Call reset() between conversations or increase max_seq_len.", self.max_seq_len
            );
        }

        self.k_buf = self.k_buf
            .slice_scatter(&new_k.detach(), 2, self.current_pos)?
            .detach();
        self.v_buf = self.v_buf
            .slice_scatter(&new_v.detach(), 2, self.current_pos)?
            .detach();

        self.current_pos = end_pos;

        let k_active = self.k_buf.narrow(2, 0, self.current_pos)?.detach();
        let v_active = self.v_buf.narrow(2, 0, self.current_pos)?.detach();

        Ok((k_active, v_active))
    }
}

// #[cfg(test)]
// mod tests {
//     use candle_core::Device;
//     use super::*;
//
//     #[test]
//     fn test() -> Result<(), Error> {
//         let mut kv_cache = KvCache::new(10, 2)?;
//
//         let tensor1 = Tensor::new(&[[[ 45f32, 1.0 ]]], &Device::Cpu)?;
//         let tensor2 = Tensor::new(&[[[ 45f32, 1.0 ]]], &Device::Cpu)?;
//
//         for cache in kv_cache.iter_mut() {
//             cache.append(&tensor1, &tensor2)?;
//         }
//
//         assert_eq!(kv_cache.current_pos(), 2);
//
//         kv_cache.truncate(2);
//         assert_eq!(kv_cache.current_pos(), 2);
//
//         kv_cache.truncate(1);
//         assert_eq!(kv_cache.current_pos(), 1);
//
//         kv_cache.clear();
//         assert_eq!(kv_cache.current_pos(), 0);
//
//         Ok(())
//     }
// }