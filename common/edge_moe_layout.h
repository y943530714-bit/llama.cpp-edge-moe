#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class edge_moe_tensor_part : uint8_t {
    none,
    gate,
    up,
    gate_up,
    down,
};

struct edge_moe_tensor_metadata {
    uint32_t file_idx = 0;
    std::string name;
    size_t file_offset = 0;
    size_t tensor_size = 0;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    ggml_type type = GGML_TYPE_COUNT;
};

struct edge_moe_expert_file_range {
    uint32_t file_idx = 0;
    uint32_t expert_id = 0;
    size_t offset = 0;
    size_t size = 0;
    ggml_type type = GGML_TYPE_COUNT;
    edge_moe_tensor_part part = edge_moe_tensor_part::none;
};

struct edge_moe_expert_layout {
    edge_moe_tensor_metadata tensor;
    edge_moe_tensor_part part = edge_moe_tensor_part::none;
    size_t expert_stride = 0;
    uint32_t expert_count = 0;
    std::vector<edge_moe_expert_file_range> ranges;
};

edge_moe_tensor_part edge_moe_tensor_part_from_name(const std::string & name);
const char * edge_moe_tensor_part_name(edge_moe_tensor_part part);

bool edge_moe_parse_layer(const std::string & name, int & layer);

// Build byte ranges for the expert dimension of a contiguous 3-D tensor.
// file_offset is the absolute offset of the tensor data in its source file.
// Pass SIZE_MAX as file_size when the caller cannot provide a file bound.
bool edge_moe_build_expert_layout(
        const edge_moe_tensor_metadata & metadata,
        size_t file_size,
        edge_moe_expert_layout & layout,
        std::string & error);
