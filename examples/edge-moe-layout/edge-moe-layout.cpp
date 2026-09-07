#include "edge_moe_layout.h"

#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace {

struct options {
    std::vector<std::string> files;
    uint32_t expert = 0;
    size_t alignment = 0;
    bool all_experts = false;
    bool verify = false;
    bool self_test = false;
    bool show_help = false;
};

struct scan_result {
    size_t matched_tensors = 0;
    size_t expert_ranges = 0;
    size_t verified_ranges = 0;
    size_t errors = 0;
};

struct gguf_context_deleter {
    void operator()(gguf_context * ctx) const {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

using gguf_context_ptr = std::unique_ptr<gguf_context, gguf_context_deleter>;

void print_usage(const char * program) {
    std::printf(
        "Usage: %s --model MODEL.gguf [options]\n"
        "       %s --self-test\n\n"
        "Options:\n"
        "  --model PATH          GGUF file or first split\n"
        "  --split PATH          additional GGUF split (repeatable)\n"
        "  --expert N            expert to print/verify (default: 0)\n"
        "  --all-experts         print/verify every expert range\n"
        "  --alignment N         require range offset and size alignment\n"
        "  --verify              read selected ranges and validate row data\n"
        "  --self-test           run the in-memory slice mapping test\n"
        "  --help                show this help\n",
        program, program);
}

bool parse_u64(const char * text, uint64_t & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }

    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }

    value = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_options(int argc, char ** argv, options & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            params.show_help = true;
            return true;
        }
        if (arg == "--self-test") {
            params.self_test = true;
            continue;
        }
        if (arg == "--all-experts") {
            params.all_experts = true;
            continue;
        }
        if (arg == "--verify") {
            params.verify = true;
            continue;
        }

        if (arg == "--model" || arg == "--split" || arg == "--expert" || arg == "--alignment") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", arg.c_str());
                return false;
            }

            const char * value = argv[++i];
            if (arg == "--model" || arg == "--split") {
                params.files.emplace_back(value);
            } else {
                uint64_t parsed = 0;
                if (!parse_u64(value, parsed)) {
                    std::fprintf(stderr, "invalid numeric value for %s: %s\n", arg.c_str(), value);
                    return false;
                }
                if (arg == "--expert") {
                    if (parsed > std::numeric_limits<uint32_t>::max()) {
                        std::fprintf(stderr, "expert id is too large: %s\n", value);
                        return false;
                    }
                    params.expert = static_cast<uint32_t>(parsed);
                } else {
                    if (parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
                        std::fprintf(stderr, "alignment must be greater than zero: %s\n", value);
                        return false;
                    }
                    params.alignment = static_cast<size_t>(parsed);
                }
            }
            continue;
        }

        std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
        return false;
    }

    if (params.self_test) {
        if (!params.files.empty()) {
            std::fprintf(stderr, "--self-test cannot be combined with --model or --split\n");
            return false;
        }
        return true;
    }
    if (params.files.empty()) {
        std::fprintf(stderr, "--model is required\n");
        return false;
    }
    return true;
}

std::string shape_string(const std::array<int64_t, GGML_MAX_DIMS> & ne) {
    std::string result = "[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(ne[i]);
    }
    result += "]";
    return result;
}

uint64_t fnv1a64(const std::vector<uint8_t> & data) {
    uint64_t hash = 14695981039346656037ull;
    for (const uint8_t value : data) {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    return hash;
}

bool read_range(const std::string & path, const edge_moe_expert_file_range & range, std::vector<uint8_t> & data) {
    if (range.size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) ||
        range.offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    data.resize(range.size);
    file.seekg(static_cast<std::streamoff>(range.offset), std::ios::beg);
    file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(range.size));
    return file.gcount() == static_cast<std::streamsize>(range.size);
}

bool range_is_aligned(const edge_moe_expert_file_range & range, const size_t alignment) {
    return alignment == 0 || (range.offset % alignment == 0 && range.size % alignment == 0);
}

bool run_self_test() {
    edge_moe_tensor_metadata metadata;
    metadata.name = "blk.0.ffn_down_exps.weight";
    metadata.file_offset = 4096;
    metadata.ne = { 8, 2, 4, 1 };
    metadata.type = GGML_TYPE_F32;
    metadata.tensor_size = 8 * 2 * 4 * sizeof(float);

    edge_moe_expert_layout layout;
    std::string error;
    if (!edge_moe_build_expert_layout(metadata, std::numeric_limits<size_t>::max(), layout, error)) {
        std::fprintf(stderr, "self-test: layout build failed: %s\n", error.c_str());
        return false;
    }

    std::vector<uint8_t> tensor(metadata.tensor_size);
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<uint8_t>((i * 17 + 3) & 0xff);
    }

    std::vector<uint8_t> file(metadata.file_offset + tensor.size(), 0);
    std::memcpy(file.data() + metadata.file_offset, tensor.data(), tensor.size());

    for (const auto & range : layout.ranges) {
        const size_t relative_offset = range.offset - metadata.file_offset;
        if (std::memcmp(file.data() + range.offset, tensor.data() + relative_offset, range.size) != 0) {
            std::fprintf(stderr, "self-test: expert %u does not map byte-for-byte\n", range.expert_id);
            return false;
        }
    }

    std::printf("self-test: %u expert ranges map byte-for-byte\n", layout.expert_count);
    return true;
}

bool scan_file(const std::string & path, const uint32_t file_idx, const options & params, scan_result & result) {
    std::error_code ec;
    const uintmax_t file_size_u = std::filesystem::file_size(path, ec);
    if (ec || file_size_u > std::numeric_limits<size_t>::max()) {
        std::fprintf(stderr, "file[%u] %s: failed to get file size\n", file_idx, path.c_str());
        ++result.errors;
        return false;
    }
    const size_t file_size = static_cast<size_t>(file_size_u);

    gguf_init_params init_params;
    init_params.no_alloc = true;
    init_params.ctx = nullptr;
    gguf_context_ptr ctx(gguf_init_from_file(path.c_str(), init_params));
    if (!ctx) {
        std::fprintf(stderr, "file[%u] %s: failed to read GGUF metadata\n", file_idx, path.c_str());
        ++result.errors;
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx.get());
    std::printf("file[%u] %s: size=%zu data_offset=%zu gguf_alignment=%zu\n",
        file_idx, path.c_str(), file_size, data_offset, gguf_get_alignment(ctx.get()));

    const int64_t n_tensors = gguf_get_n_tensors(ctx.get());
    for (int64_t tensor_id = 0; tensor_id < n_tensors; ++tensor_id) {
        const char * tensor_name = gguf_get_tensor_name(ctx.get(), tensor_id);
        if (tensor_name == nullptr) {
            continue;
        }

        const std::string name(tensor_name);
        if (edge_moe_tensor_part_from_name(name) == edge_moe_tensor_part::none) {
            continue;
        }
        ++result.matched_tensors;

        const size_t tensor_offset = gguf_get_tensor_offset(ctx.get(), tensor_id);
        if (data_offset > std::numeric_limits<size_t>::max() - tensor_offset) {
            std::fprintf(stderr, "  tensor=%s: file offset overflows size_t\n", name.c_str());
            ++result.errors;
            continue;
        }

        edge_moe_tensor_metadata metadata;
        metadata.file_idx = file_idx;
        metadata.name = name;
        metadata.file_offset = data_offset + tensor_offset;
        metadata.tensor_size = gguf_get_tensor_size(ctx.get(), tensor_id);
        metadata.type = gguf_get_tensor_type(ctx.get(), tensor_id);
        const int64_t * ne = gguf_get_tensor_ne(ctx.get(), tensor_id);
        if (ne == nullptr) {
            std::fprintf(stderr, "  tensor=%s: missing shape metadata\n", name.c_str());
            ++result.errors;
            continue;
        }
        std::copy(ne, ne + GGML_MAX_DIMS, metadata.ne.begin());

        edge_moe_expert_layout layout;
        std::string error;
        if (!edge_moe_build_expert_layout(metadata, file_size, layout, error)) {
            std::fprintf(stderr, "  tensor=%s: invalid expert layout: %s\n", name.c_str(), error.c_str());
            ++result.errors;
            continue;
        }
        result.expert_ranges += layout.ranges.size();

        bool alignment_ok = true;
        if (params.alignment != 0) {
            for (const auto & range : layout.ranges) {
                if (!range_is_aligned(range, params.alignment)) {
                    alignment_ok = false;
                    break;
                }
            }
            if (!alignment_ok) {
                std::fprintf(stderr, "  tensor=%s: expert ranges are not aligned to %zu bytes\n",
                    name.c_str(), params.alignment);
                ++result.errors;
            }
        }

        int layer = -1;
        edge_moe_parse_layer(name, layer);
        std::printf("  layer=%d tensor=%s part=%s type=%s shape=%s tensor_offset=%zu tensor_bytes=%zu expert_count=%u expert_stride=%zu alignment=%s\n",
            layer,
            name.c_str(),
            edge_moe_tensor_part_name(layout.part),
            ggml_type_name(metadata.type),
            shape_string(metadata.ne).c_str(),
            metadata.file_offset,
            metadata.tensor_size,
            layout.expert_count,
            layout.expert_stride,
            alignment_ok ? "ok" : "failed");

        if (!params.all_experts && params.expert >= layout.expert_count) {
            std::fprintf(stderr, "  tensor=%s: expert %u is outside [0, %u)\n",
                name.c_str(), params.expert, layout.expert_count);
            ++result.errors;
            continue;
        }

        const auto verify_one = [&](const edge_moe_expert_file_range & range) {
            std::vector<uint8_t> data;
            if (!read_range(path, range, data)) {
                std::fprintf(stderr, "  tensor=%s expert=%u: failed to read offset=%zu size=%zu\n",
                    name.c_str(), range.expert_id, range.offset, range.size);
                ++result.errors;
                return;
            }
            if (!ggml_validate_row_data(range.type, data.data(), data.size())) {
                std::fprintf(stderr, "  tensor=%s expert=%u: row data validation failed\n",
                    name.c_str(), range.expert_id);
                ++result.errors;
                return;
            }
            ++result.verified_ranges;
            std::printf("    expert=%u offset=%zu size=%zu hash=0x%016" PRIx64 " row_data=ok\n",
                range.expert_id, range.offset, range.size, fnv1a64(data));
        };

        if (params.verify) {
            if (params.all_experts) {
                for (const auto & range : layout.ranges) {
                    verify_one(range);
                }
            } else {
                verify_one(layout.ranges[params.expert]);
            }
        } else if (params.all_experts) {
            for (const auto & range : layout.ranges) {
                std::printf("    expert=%u offset=%zu size=%zu\n", range.expert_id, range.offset, range.size);
            }
        } else {
            const auto & range = layout.ranges[params.expert];
            std::printf("    expert=%u offset=%zu size=%zu\n", range.expert_id, range.offset, range.size);
        }
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    options params;
    if (!parse_options(argc, argv, params)) {
        return 2;
    }
    if (params.show_help) {
        return 0;
    }
    if (params.self_test) {
        return run_self_test() ? 0 : 1;
    }

    scan_result result;
    for (uint32_t file_idx = 0; file_idx < params.files.size(); ++file_idx) {
        scan_file(params.files[file_idx], file_idx, params, result);
    }

    std::printf("summary: matched_tensors=%zu expert_ranges=%zu verified_ranges=%zu errors=%zu\n",
        result.matched_tensors, result.expert_ranges, result.verified_ranges, result.errors);
    if (result.matched_tensors == 0) {
        std::fprintf(stderr, "no routed expert tensors were found\n");
        return 1;
    }
    return result.errors == 0 ? 0 : 1;
}
