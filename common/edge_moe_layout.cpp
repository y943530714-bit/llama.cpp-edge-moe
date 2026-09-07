#include "edge_moe_layout.h"

#include <limits>
#include <string_view>

namespace {

bool has_suffix(const std::string & value, const std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix.data(), suffix.size()) == 0;
}

bool checked_add(const size_t lhs, const size_t rhs, size_t & result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_mul(const size_t lhs, const size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool valid_type(const ggml_type type) {
    return type >= 0 && type < GGML_TYPE_COUNT;
}

} // namespace

edge_moe_tensor_part edge_moe_tensor_part_from_name(const std::string & name) {
    if (has_suffix(name, ".ffn_gate_up_exps.weight")) {
        return edge_moe_tensor_part::gate_up;
    }
    if (has_suffix(name, ".ffn_gate_exps.weight")) {
        return edge_moe_tensor_part::gate;
    }
    if (has_suffix(name, ".ffn_up_exps.weight")) {
        return edge_moe_tensor_part::up;
    }
    if (has_suffix(name, ".ffn_down_exps.weight")) {
        return edge_moe_tensor_part::down;
    }
    return edge_moe_tensor_part::none;
}

const char * edge_moe_tensor_part_name(const edge_moe_tensor_part part) {
    switch (part) {
        case edge_moe_tensor_part::gate:    return "gate";
        case edge_moe_tensor_part::up:      return "up";
        case edge_moe_tensor_part::gate_up: return "gate_up";
        case edge_moe_tensor_part::down:    return "down";
        case edge_moe_tensor_part::none:    return "none";
    }
    return "none";
}

bool edge_moe_parse_layer(const std::string & name, int & layer) {
    constexpr std::string_view prefix = "blk.";
    if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix.data(), prefix.size()) != 0) {
        return false;
    }

    size_t pos = prefix.size();
    uint64_t value = 0;
    bool have_digit = false;
    while (pos < name.size() && name[pos] != '.') {
        const char ch = name[pos++];
        if (ch < '0' || ch > '9') {
            return false;
        }
        have_digit = true;
        const uint64_t digit = static_cast<uint64_t>(ch - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
        if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return false;
        }
    }

    if (!have_digit || pos >= name.size() || name[pos] != '.') {
        return false;
    }

    layer = static_cast<int>(value);
    return true;
}

bool edge_moe_build_expert_layout(
        const edge_moe_tensor_metadata & metadata,
        const size_t file_size,
        edge_moe_expert_layout & layout,
        std::string & error) {
    layout = {};
    error.clear();

    const edge_moe_tensor_part part = edge_moe_tensor_part_from_name(metadata.name);
    if (part == edge_moe_tensor_part::none) {
        error = "tensor name is not a routed expert weight";
        return false;
    }
    if (!valid_type(metadata.type)) {
        error = "tensor has an invalid ggml type";
        return false;
    }
    if (metadata.ne[0] <= 0 || metadata.ne[1] <= 0 || metadata.ne[2] <= 0 || metadata.ne[3] != 1) {
        error = "expected a non-empty 3-D tensor with experts in dimension 2";
        return false;
    }

    const int64_t block_size = ggml_blck_size(metadata.type);
    if (block_size <= 0 || metadata.ne[0] % block_size != 0) {
        error = "tensor row is not aligned to the ggml type block size";
        return false;
    }

    size_t expert_stride = 0;
    if (!checked_mul(ggml_row_size(metadata.type, metadata.ne[0]), static_cast<size_t>(metadata.ne[1]), expert_stride)) {
        error = "expert stride overflows size_t";
        return false;
    }

    if (metadata.ne[2] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        error = "expert count does not fit in uint32_t";
        return false;
    }
    const uint32_t expert_count = static_cast<uint32_t>(metadata.ne[2]);

    size_t expected_tensor_size = 0;
    if (!checked_mul(expert_stride, static_cast<size_t>(expert_count), expected_tensor_size)) {
        error = "tensor size overflows size_t";
        return false;
    }
    if (expected_tensor_size != metadata.tensor_size) {
        error = "tensor size does not match contiguous expert slices";
        return false;
    }

    size_t tensor_end = 0;
    if (!checked_add(metadata.file_offset, metadata.tensor_size, tensor_end)) {
        error = "tensor file range overflows size_t";
        return false;
    }
    if (file_size != std::numeric_limits<size_t>::max() && tensor_end > file_size) {
        error = "tensor file range is outside the source file";
        return false;
    }

    layout.tensor = metadata;
    layout.part = part;
    layout.expert_stride = expert_stride;
    layout.expert_count = expert_count;
    layout.ranges.reserve(expert_count);

    for (uint32_t expert = 0; expert < expert_count; ++expert) {
        size_t relative_offset = 0;
        size_t absolute_offset = 0;
        if (!checked_mul(expert_stride, static_cast<size_t>(expert), relative_offset) ||
            !checked_add(metadata.file_offset, relative_offset, absolute_offset)) {
            layout = {};
            error = "expert file range overflows size_t";
            return false;
        }

        layout.ranges.push_back({
            metadata.file_idx,
            expert,
            absolute_offset,
            expert_stride,
            metadata.type,
            part,
        });
    }

    return true;
}
