use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

use crate::expert_store::{align_up, ExpertFileHeader, TensorLayout, ALIGNMENT, VERSION};

/// Split a model into resident weights (safetensors) and per-layer expert files.
pub fn split_model(model_path: &Path, output_path: &Path) -> io::Result<()> {
    fs::create_dir_all(output_path)?;
    let resident_dir = output_path.join("resident");
    let expert_dir = output_path.join("experts");
    fs::create_dir_all(&resident_dir)?;
    fs::create_dir_all(&expert_dir)?;

    // Copy config files
    for name in &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        let src = model_path.join(name);
        if src.exists() {
            fs::copy(&src, output_path.join(name))?;
        }
    }

    // Read weight index
    let index_path = model_path.join("model.safetensors.index.json");
    let index_str = fs::read_to_string(&index_path)?;
    let index: Value = serde_json::from_str(&index_str)?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no weight_map in index"))?;

    // Classify tensors
    let mut resident_tensors: Vec<String> = Vec::new();
    let mut expert_tensors: Vec<String> = Vec::new();
    for key in weight_map.keys() {
        if key.contains("switch_mlp") {
            expert_tensors.push(key.clone());
        } else {
            resident_tensors.push(key.clone());
        }
    }

    eprintln!(
        "Found {} resident tensors, {} expert tensors",
        resident_tensors.len(),
        expert_tensors.len()
    );

    // Open all shard files
    let mut shard_mmaps: HashMap<String, Mmap> = HashMap::new();
    let mut shard_files: Vec<String> = weight_map.values().filter_map(|v| v.as_str().map(String::from)).collect();
    shard_files.sort();
    shard_files.dedup();

    for shard_name in &shard_files {
        let shard_path = model_path.join(shard_name);
        let file = File::open(&shard_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        shard_mmaps.insert(shard_name.clone(), mmap);
    }

    // Step 1: Write resident weights as safetensors
    write_resident_weights(
        &shard_mmaps,
        weight_map,
        &resident_tensors,
        &resident_dir,
    )?;

    // Step 2: Write expert weights as per-layer binary files
    write_expert_files(&shard_mmaps, weight_map, &expert_dir)?;

    // Write a split metadata file
    let meta = serde_json::json!({
        "original_model": model_path.to_str(),
        "resident_dir": "resident",
        "expert_dir": "experts",
    });
    fs::write(
        output_path.join("split_config.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )?;

    eprintln!("Model split complete: {}", output_path.display());
    Ok(())
}

/// Parse a safetensors shard and return (header_json, data_offset).
fn parse_shard(mmap: &[u8]) -> io::Result<(Value, usize)> {
    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&mmap[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok((header, 8 + header_size))
}

/// Extract raw tensor bytes from a shard mmap.
fn extract_tensor<'a>(mmap: &'a [u8], header: &Value, tensor_name: &str) -> io::Result<&'a [u8]> {
    let info = header
        .get(tensor_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("tensor {} not in shard", tensor_name)))?;
    let offsets = info["data_offsets"]
        .as_array()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no data_offsets"))?;
    let start = offsets[0].as_u64().unwrap() as usize;
    let end = offsets[1].as_u64().unwrap() as usize;

    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_base = 8 + header_size;

    Ok(&mmap[data_base + start..data_base + end])
}

/// Get tensor shape from shard header.
fn tensor_shape(header: &Value, tensor_name: &str) -> io::Result<Vec<usize>> {
    let info = header
        .get(tensor_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, tensor_name.to_string()))?;
    Ok(info["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect())
}

/// Write resident (non-expert) weights as safetensors files.
fn write_resident_weights(
    shard_mmaps: &HashMap<String, Mmap>,
    weight_map: &serde_json::Map<String, Value>,
    resident_tensors: &[String],
    output_dir: &Path,
) -> io::Result<()> {
    // Parse all shard headers
    let mut shard_headers: HashMap<String, Value> = HashMap::new();
    for (name, mmap) in shard_mmaps {
        let (header, _) = parse_shard(mmap)?;
        shard_headers.insert(name.clone(), header);
    }

    // Group resident tensors by output shard (keep them in reasonable sized files)
    // We'll write a single resident safetensors file since resident weights are only ~2.5GB
    let mut tensor_data: Vec<(String, Vec<u8>, String, Vec<usize>)> = Vec::new();

    for tensor_name in resident_tensors {
        let shard_name = weight_map[tensor_name].as_str().unwrap();
        let mmap = &shard_mmaps[shard_name];
        let header = &shard_headers[shard_name];

        let data = extract_tensor(mmap, header, tensor_name)?;
        let shape = tensor_shape(header, tensor_name)?;
        let dtype = header[tensor_name]["dtype"]
            .as_str()
            .unwrap_or("F32")
            .to_string();

        // Strip "language_model." prefix for MLX model compatibility
        let clean_name = tensor_name
            .strip_prefix("language_model.")
            .unwrap_or(tensor_name)
            .to_string();

        tensor_data.push((clean_name, data.to_vec(), dtype, shape));
    }

    // Sort by name for deterministic output
    tensor_data.sort_by(|a, b| a.0.cmp(&b.0));

    // Write safetensors using the safetensors crate format
    // Build the header manually since we need to preserve dtype
    write_safetensors_file(&tensor_data, &output_dir.join("resident.safetensors"))?;

    // Also write an index file for MLX compatibility
    let mut new_weight_map = serde_json::Map::new();
    for (name, _, _, _) in &tensor_data {
        new_weight_map.insert(
            name.clone(),
            Value::String("resident.safetensors".to_string()),
        );
    }
    let index = serde_json::json!({
        "metadata": { "format": "mlx" },
        "weight_map": new_weight_map,
    });
    fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )?;

    let total_bytes: usize = tensor_data.iter().map(|(_, d, _, _)| d.len()).sum();
    eprintln!(
        "Wrote {} resident tensors ({:.2} GB) to {}",
        tensor_data.len(),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        output_dir.display()
    );

    Ok(())
}

/// Write a safetensors file from tensor data.
fn write_safetensors_file(
    tensors: &[(String, Vec<u8>, String, Vec<usize>)],
    path: &Path,
) -> io::Result<()> {
    // Build header JSON
    let mut header_map = serde_json::Map::new();
    let mut offset = 0u64;

    for (name, data, dtype, shape) in tensors {
        let end = offset + data.len() as u64;
        header_map.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }

    // Add __metadata__
    header_map.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt"}),
    );

    let header_json = serde_json::to_string(&Value::Object(header_map))?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;

    for (_, data, _, _) in tensors {
        file.write_all(data)?;
    }

    file.sync_all()?;
    Ok(())
}

/// Write per-layer expert binary files.
fn write_expert_files(
    shard_mmaps: &HashMap<String, Mmap>,
    weight_map: &serde_json::Map<String, Value>,
    expert_dir: &Path,
) -> io::Result<()> {
    // Parse all shard headers
    let mut shard_headers: HashMap<String, Value> = HashMap::new();
    for (name, mmap) in shard_mmaps {
        let (header, _) = parse_shard(mmap)?;
        shard_headers.insert(name.clone(), header);
    }

    // Discover number of layers by scanning tensor names
    let num_layers = {
        let mut max_layer = 0u32;
        for key in weight_map.keys() {
            if key.contains("switch_mlp") {
                if let Some(rest) = key.strip_prefix("language_model.model.layers.") {
                    if let Some(dot_pos) = rest.find('.') {
                        if let Ok(idx) = rest[..dot_pos].parse::<u32>() {
                            max_layer = max_layer.max(idx);
                        }
                    }
                }
            }
        }
        max_layer + 1
    };

    eprintln!("Processing {} layers of expert weights...", num_layers);

    for layer_idx in 0..num_layers {
        write_layer_experts(
            layer_idx,
            shard_mmaps,
            &shard_headers,
            weight_map,
            expert_dir,
        )?;

        if (layer_idx + 1) % 10 == 0 || layer_idx == num_layers - 1 {
            eprintln!("  Processed layer {}/{}", layer_idx + 1, num_layers);
        }
    }

    Ok(())
}

/// Write a single layer's expert file.
fn write_layer_experts(
    layer_idx: u32,
    shard_mmaps: &HashMap<String, Mmap>,
    shard_headers: &HashMap<String, Value>,
    weight_map: &serde_json::Map<String, Value>,
    expert_dir: &Path,
) -> io::Result<()> {
    let prefix = format!("language_model.model.layers.{}.mlp.switch_mlp", layer_idx);

    // Tensor names for this layer's experts
    let proj_names = ["gate_proj", "up_proj", "down_proj"];
    let component_names = ["weight", "scales", "biases"];

    // Load all 9 tensors (3 projections x 3 components)
    // Each tensor has shape [256, ...] (all experts stacked)
    struct ProjData {
        weight: Vec<u8>,
        weight_shape: Vec<usize>,
        scales: Vec<u8>,
        scales_shape: Vec<usize>,
        biases: Vec<u8>,
        biases_shape: Vec<usize>,
    }

    let mut projections: Vec<ProjData> = Vec::new();
    let mut num_experts = 0u32;

    for proj_name in &proj_names {
        let mut proj = ProjData {
            weight: Vec::new(),
            weight_shape: Vec::new(),
            scales: Vec::new(),
            scales_shape: Vec::new(),
            biases: Vec::new(),
            biases_shape: Vec::new(),
        };

        for comp_name in &component_names {
            let tensor_name = format!("{}.{}.{}", prefix, proj_name, comp_name);
            let shard_name = weight_map
                .get(&tensor_name)
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("tensor {} not in weight map", tensor_name),
                    )
                })?;
            let mmap = &shard_mmaps[shard_name];
            let header = &shard_headers[shard_name];
            let data = extract_tensor(mmap, header, &tensor_name)?;
            let shape = tensor_shape(header, &tensor_name)?;

            if num_experts == 0 {
                num_experts = shape[0] as u32;
            }

            match *comp_name {
                "weight" => {
                    proj.weight = data.to_vec();
                    proj.weight_shape = shape;
                }
                "scales" => {
                    proj.scales = data.to_vec();
                    proj.scales_shape = shape;
                }
                "biases" => {
                    proj.biases = data.to_vec();
                    proj.biases_shape = shape;
                }
                _ => unreachable!(),
            }
        }
        projections.push(proj);
    }

    // Calculate per-expert sizes
    // Each tensor is [num_experts, d1, d2] - we need the per-expert slice size
    let expert_byte_size = |proj: &ProjData| -> (usize, usize, usize) {
        let n = proj.weight_shape[0];
        let w_per = proj.weight.len() / n;
        let s_per = proj.scales.len() / n;
        let b_per = proj.biases.len() / n;
        (w_per, s_per, b_per)
    };

    let gate = &projections[0];
    let up = &projections[1];
    let down = &projections[2];

    let (gw, gs, gb) = expert_byte_size(gate);
    let (uw, us, ub) = expert_byte_size(up);
    let (dw, ds, db) = expert_byte_size(down);

    let raw_expert_size = gw + gs + gb + uw + us + ub + dw + ds + db;
    let expert_stride = align_up(raw_expert_size as u64, ALIGNMENT);

    // Build header
    let mut offset = 0u64;
    let mut make_layout = |w: usize, s: usize, b: usize| -> TensorLayout {
        let layout = TensorLayout {
            weight_offset: offset,
            weight_size: w as u64,
            scales_offset: offset + w as u64,
            scales_size: s as u64,
            biases_offset: offset + w as u64 + s as u64,
            biases_size: b as u64,
        };
        offset += (w + s + b) as u64;
        layout
    };

    let gate_layout = make_layout(gw, gs, gb);
    let up_layout = make_layout(uw, us, ub);
    let down_layout = make_layout(dw, ds, db);

    let to_shape2 = |shape: &[usize]| -> [u32; 2] {
        // Shape is [num_experts, d1, d2] -> per-expert shape is [d1, d2]
        [shape[1] as u32, shape[2] as u32]
    };

    let header = ExpertFileHeader {
        magic: *crate::expert_store::MAGIC,
        version: VERSION,
        num_experts,
        expert_stride,
        quant_bits: 8,
        quant_group_size: 32,
        gate_proj: gate_layout,
        up_proj: up_layout,
        down_proj: down_layout,
        gate_weight_shape: to_shape2(&gate.weight_shape),
        gate_scales_shape: to_shape2(&gate.scales_shape),
        up_weight_shape: to_shape2(&up.weight_shape),
        up_scales_shape: to_shape2(&up.scales_shape),
        down_weight_shape: to_shape2(&down.weight_shape),
        down_scales_shape: to_shape2(&down.scales_shape),
    };

    // Write the file
    let file_path = expert_dir.join(format!("layer_{:02}_experts.bin", layer_idx));
    let mut file = File::create(&file_path)?;

    // Header (padded to 4096 bytes)
    file.write_all(&header.to_bytes())?;

    // Expert data blocks
    let n = num_experts as usize;
    for expert_idx in 0..n {
        let mut expert_buf = vec![0u8; expert_stride as usize];
        let mut pos = 0usize;

        // Helper to copy a slice of a stacked tensor
        let copy_slice =
            |buf: &mut [u8], pos: &mut usize, data: &[u8], idx: usize, per_expert: usize| {
                let start = idx * per_expert;
                buf[*pos..*pos + per_expert].copy_from_slice(&data[start..start + per_expert]);
                *pos += per_expert;
            };

        // gate_proj: weight, scales, biases
        copy_slice(&mut expert_buf, &mut pos, &gate.weight, expert_idx, gw);
        copy_slice(&mut expert_buf, &mut pos, &gate.scales, expert_idx, gs);
        copy_slice(&mut expert_buf, &mut pos, &gate.biases, expert_idx, gb);

        // up_proj: weight, scales, biases
        copy_slice(&mut expert_buf, &mut pos, &up.weight, expert_idx, uw);
        copy_slice(&mut expert_buf, &mut pos, &up.scales, expert_idx, us);
        copy_slice(&mut expert_buf, &mut pos, &up.biases, expert_idx, ub);

        // down_proj: weight, scales, biases
        copy_slice(&mut expert_buf, &mut pos, &down.weight, expert_idx, dw);
        copy_slice(&mut expert_buf, &mut pos, &down.scales, expert_idx, ds);
        copy_slice(&mut expert_buf, &mut pos, &down.biases, expert_idx, db);

        // Remaining bytes are zero (padding to expert_stride)
        file.write_all(&expert_buf)?;
    }

    file.sync_all()?;
    Ok(())
}
