use mlx_rs::error::Exception;
use mlx_rs::Array;

/// RMSNorm — wraps mlx_rs::fast::rms_norm.
pub struct RMSNorm {
    pub weight: Array,
    pub eps: f32,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        mlx_rs::fast::rms_norm(x, &self.weight, self.eps)
    }
}

/// RMSNormNoScale — rms_norm without learnable weight (Gemma4 v_norm).
pub struct RMSNormNoScale {
    pub eps: f32,
}

impl RMSNormNoScale {
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let x2 = x * x;
        let mean = mlx_rs::ops::mean_axes(&x2, &[-1], Some(true))?;
        let eps = Array::from_f32(self.eps);
        let rms = mlx_rs::ops::rsqrt(&(&mean + &eps))?;
        Ok(x * &rms)
    }
}

