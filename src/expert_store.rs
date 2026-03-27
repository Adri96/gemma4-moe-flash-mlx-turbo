use std::fs::{self, File};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

pub const MAGIC: &[u8; 4] = b"FEXP";
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: u64 = 4096;
pub const ALIGNMENT: u64 = 16384; // 16KB alignment for SSD reads

/// Metadata for tensor components within an expert's data block.
/// All offsets are relative to the start of the expert's data block.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TensorLayout {
    pub weight_offset: u64,
    pub weight_size: u64,
    pub scales_offset: u64,
    pub scales_size: u64,
    pub biases_offset: u64,
    pub biases_size: u64,
}

/// File header stored at the beginning of each layer's expert file.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct ExpertFileHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub num_experts: u32,
    pub expert_stride: u64,
    pub quant_bits: u32,
    pub quant_group_size: u32,
    pub gate_proj: TensorLayout,
    pub up_proj: TensorLayout,
    pub down_proj: TensorLayout,
    // Shapes for reconstruction
    pub gate_weight_shape: [u32; 2], // [out, in_packed]
    pub gate_scales_shape: [u32; 2], // [out, in/group]
    pub up_weight_shape: [u32; 2],
    pub up_scales_shape: [u32; 2],
    pub down_weight_shape: [u32; 2],
    pub down_scales_shape: [u32; 2],
}

impl ExpertFileHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; HEADER_SIZE as usize];
        let mut pos = 0usize;

        buf[pos..pos + 4].copy_from_slice(&self.magic);
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.version.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.num_experts.to_le_bytes());
        pos += 4;
        // 4 bytes padding for alignment
        pos += 4;
        buf[pos..pos + 8].copy_from_slice(&self.expert_stride.to_le_bytes());
        pos += 8;
        buf[pos..pos + 4].copy_from_slice(&self.quant_bits.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.quant_group_size.to_le_bytes());
        pos += 4;

        // Write tensor layouts
        for layout in [&self.gate_proj, &self.up_proj, &self.down_proj] {
            buf[pos..pos + 8].copy_from_slice(&layout.weight_offset.to_le_bytes());
            pos += 8;
            buf[pos..pos + 8].copy_from_slice(&layout.weight_size.to_le_bytes());
            pos += 8;
            buf[pos..pos + 8].copy_from_slice(&layout.scales_offset.to_le_bytes());
            pos += 8;
            buf[pos..pos + 8].copy_from_slice(&layout.scales_size.to_le_bytes());
            pos += 8;
            buf[pos..pos + 8].copy_from_slice(&layout.biases_offset.to_le_bytes());
            pos += 8;
            buf[pos..pos + 8].copy_from_slice(&layout.biases_size.to_le_bytes());
            pos += 8;
        }

        // Write shapes
        for shape in [
            &self.gate_weight_shape,
            &self.gate_scales_shape,
            &self.up_weight_shape,
            &self.up_scales_shape,
            &self.down_weight_shape,
            &self.down_scales_shape,
        ] {
            buf[pos..pos + 4].copy_from_slice(&shape[0].to_le_bytes());
            pos += 4;
            buf[pos..pos + 4].copy_from_slice(&shape[1].to_le_bytes());
            pos += 4;
        }

        buf
    }

    pub fn from_bytes(buf: &[u8]) -> io::Result<Self> {
        if buf.len() < HEADER_SIZE as usize {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "header too short"));
        }
        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

        let mut pos = 4usize;
        let version = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let num_experts = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        pos += 4; // padding
        let expert_stride = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let quant_bits = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let quant_group_size = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let read_layout = |pos: &mut usize| -> TensorLayout {
            let r = |p: &mut usize| -> u64 {
                let v = u64::from_le_bytes(buf[*p..*p + 8].try_into().unwrap());
                *p += 8;
                v
            };
            TensorLayout {
                weight_offset: r(pos),
                weight_size: r(pos),
                scales_offset: r(pos),
                scales_size: r(pos),
                biases_offset: r(pos),
                biases_size: r(pos),
            }
        };

        let gate_proj = read_layout(&mut pos);
        let up_proj = read_layout(&mut pos);
        let down_proj = read_layout(&mut pos);

        let read_shape = |pos: &mut usize| -> [u32; 2] {
            let a = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            let b = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            [a, b]
        };

        let gate_weight_shape = read_shape(&mut pos);
        let gate_scales_shape = read_shape(&mut pos);
        let up_weight_shape = read_shape(&mut pos);
        let up_scales_shape = read_shape(&mut pos);
        let down_weight_shape = read_shape(&mut pos);
        let down_scales_shape = read_shape(&mut pos);

        Ok(ExpertFileHeader {
            magic,
            version,
            num_experts,
            expert_stride,
            quant_bits,
            quant_group_size,
            gate_proj,
            up_proj,
            down_proj,
            gate_weight_shape,
            gate_scales_shape,
            up_weight_shape,
            up_scales_shape,
            down_weight_shape,
            down_scales_shape,
        })
    }
}

/// Raw loaded data for a single expert.
#[derive(Debug, Clone)]
pub struct ExpertData {
    pub data: Vec<u8>,
    pub layer_idx: u32,
    pub expert_idx: u32,
}

impl ExpertData {
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

/// Manages open file handles to per-layer expert files for direct I/O.
pub struct ExpertStore {
    expert_dir: PathBuf,
    num_layers: u32,
    headers: Vec<ExpertFileHeader>,
    fds: Vec<i32>, // raw file descriptors with F_NOCACHE
}

impl ExpertStore {
    pub fn new(expert_dir: &Path) -> io::Result<Self> {
        let mut headers = Vec::new();
        let mut fds = Vec::new();

        // Discover layer files
        let mut layer_files: Vec<(u32, PathBuf)> = Vec::new();
        for entry in fs::read_dir(expert_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(rest) = name.strip_prefix("layer_") {
                if let Some(num_str) = rest.strip_suffix("_experts.bin") {
                    if let Ok(idx) = num_str.parse::<u32>() {
                        layer_files.push((idx, entry.path()));
                    }
                }
            }
        }
        layer_files.sort_by_key(|(idx, _)| *idx);

        let num_layers = layer_files.len() as u32;

        for (_, path) in &layer_files {
            let file = File::open(path)?;
            let fd = file.as_raw_fd();

            // Set F_NOCACHE to bypass page cache
            unsafe {
                libc::fcntl(fd, libc::F_NOCACHE, 1);
            }

            // Read header
            let mut header_buf = vec![0u8; HEADER_SIZE as usize];
            let n = unsafe {
                libc::pread(
                    fd,
                    header_buf.as_mut_ptr() as *mut libc::c_void,
                    HEADER_SIZE as usize,
                    0,
                )
            };
            if n != HEADER_SIZE as isize {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("failed to read header from {:?}", path),
                ));
            }

            let header = ExpertFileHeader::from_bytes(&header_buf)?;
            headers.push(header);

            // Keep file descriptor alive by leaking the File
            // (we manage the fd lifetime ourselves)
            let raw_fd = file.as_raw_fd();
            std::mem::forget(file);
            fds.push(raw_fd);
        }

        Ok(ExpertStore {
            expert_dir: expert_dir.to_path_buf(),
            num_layers,
            headers,
            fds,
        })
    }

    pub fn num_layers(&self) -> u32 {
        self.num_layers
    }

    pub fn header(&self, layer_idx: u32) -> &ExpertFileHeader {
        &self.headers[layer_idx as usize]
    }

    /// Load a single expert's raw data from SSD.
    pub fn load_expert(&self, layer_idx: u32, expert_idx: u32) -> io::Result<ExpertData> {
        let header = &self.headers[layer_idx as usize];
        let fd = self.fds[layer_idx as usize];

        let offset = HEADER_SIZE + (expert_idx as u64) * header.expert_stride;
        let size = header.expert_stride as usize;

        let mut buf = vec![0u8; size];
        let n = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr() as *mut libc::c_void,
                size,
                offset as i64,
            )
        };
        if n != size as isize {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "pread returned {} bytes, expected {} (layer={}, expert={})",
                    n, size, layer_idx, expert_idx
                ),
            ));
        }

        Ok(ExpertData {
            data: buf,
            layer_idx,
            expert_idx,
        })
    }

    /// Load multiple experts in parallel using rayon.
    pub fn load_experts(
        &self,
        layer_idx: u32,
        expert_indices: &[u32],
    ) -> io::Result<Vec<ExpertData>> {
        expert_indices
            .par_iter()
            .map(|&idx| self.load_expert(layer_idx, idx))
            .collect()
    }
}

impl Drop for ExpertStore {
    fn drop(&mut self) {
        for &fd in &self.fds {
            unsafe {
                libc::close(fd);
            }
        }
    }
}

/// Align a value up to the given alignment.
pub fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}
