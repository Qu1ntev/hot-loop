use candle_core::{DType, Device, Tensor, Result};

pub(crate) fn mask(
    b_sz: usize,
    seq_len: usize,
    offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..(seq_len + offset)).map(move |j| {
                let past_ok = j <= i + offset;

                let sw_ok = match sliding_window {
                    Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                    None => true,
                };

                if past_ok && sw_ok { 0. } else { f32::NEG_INFINITY }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (b_sz, 1, seq_len, seq_len + offset), device)?.to_dtype(dtype)
}