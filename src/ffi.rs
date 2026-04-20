//! FFI wrapper for mlx-c functions not yet exposed by mlx-rs.

use mlx_rs::{Array, Dtype};

extern "C" {
    fn mlx_array_from_mmap(
        mmap_ptr: *const std::ffi::c_void,
        byte_offset: usize,
        byte_length: usize,
        shape: *const i32,
        dim: i32,
        dtype: mlx_sys::mlx_dtype,
    ) -> mlx_sys::mlx_array;
}

/// Create a zero-copy MLX array backed by mmap'd memory via Metal's newBufferWithBytesNoCopy.
/// The mmap must outlive the returned array.
pub unsafe fn array_from_mmap(
    mmap_ptr: *const u8,
    byte_offset: usize,
    byte_length: usize,
    shape: &[i32],
    dtype: Dtype,
) -> Array {
    let result = mlx_array_from_mmap(
        mmap_ptr as *const std::ffi::c_void,
        byte_offset,
        byte_length,
        shape.as_ptr(),
        shape.len() as i32,
        dtype.into(),
    );
    Array::from_ptr(result)
}

