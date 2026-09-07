#include "edge_moe_layout.h"
#include "edge_moe_resident_slots.h"

#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct options {
    std::vector<std::string> files;
    std::vector<uint32_t> experts;
    edge_moe_tensor_part part = edge_moe_tensor_part::gate_up;
    uint32_t layer = 0;
    uint32_t slots = 0;
    bool self_test = false;
    bool show_help = false;
};

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

bool parse_part(const std::string & text, edge_moe_tensor_part & part) {
    if (text == "gate") {
        part = edge_moe_tensor_part::gate;
    } else if (text == "up") {
        part = edge_moe_tensor_part::up;
    } else if (text == "gate_up") {
        part = edge_moe_tensor_part::gate_up;
    } else if (text == "down") {
        part = edge_moe_tensor_part::down;
    } else {
        return false;
    }
    return true;
}

bool parse_experts(const std::string & text, std::vector<uint32_t> & experts) {
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t end = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        uint64_t value = 0;
        if (item.empty() || !parse_u64(item.c_str(), value) || value > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        experts.push_back(static_cast<uint32_t>(value));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return !experts.empty();
}

void print_usage(const char * program) {
    std::printf(
        "Usage: %s --model MODEL.gguf [options]\n"
        "       %s --self-test\n\n"
        "Options:\n"
        "  --model PATH          GGUF file or first split\n"
        "  --split PATH          additional GGUF split (repeatable)\n"
        "  --layer N             layer to probe (default: 0)\n"
        "  --part NAME           gate, up, gate_up, or down (default: gate_up)\n"
        "  --experts LIST        comma-separated logical expert IDs (default: 0)\n"
        "  --slots N             number of resident slots (default: expert count)\n"
        "  --self-test           test remapping with ggml_mul_mat_id on CPU\n"
        "  --help                show this help\n",
        program, program);
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

        if (arg == "--model" || arg == "--split" || arg == "--layer" || arg == "--part" ||
            arg == "--experts" || arg == "--slots") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", arg.c_str());
                return false;
            }
            const std::string value = argv[++i];
            if (arg == "--model" || arg == "--split") {
                params.files.push_back(value);
            } else if (arg == "--part") {
                if (!parse_part(value, params.part)) {
                    std::fprintf(stderr, "unknown tensor part: %s\n", value.c_str());
                    return false;
                }
            } else {
                uint64_t parsed = 0;
                if (arg == "--experts") {
                    if (!parse_experts(value, params.experts)) {
                        std::fprintf(stderr, "invalid expert list: %s\n", value.c_str());
                        return false;
                    }
                } else if (!parse_u64(value.c_str(), parsed) || parsed > std::numeric_limits<uint32_t>::max()) {
                    std::fprintf(stderr, "invalid numeric value for %s: %s\n", arg.c_str(), value.c_str());
                    return false;
                } else if (arg == "--layer") {
                    params.layer = static_cast<uint32_t>(parsed);
                } else {
                    params.slots = static_cast<uint32_t>(parsed);
                }
            }
            continue;
        }

        std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
        return false;
    }

    if (params.experts.empty()) {
        params.experts.push_back(0);
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

uint64_t fnv1a64(const std::vector<uint8_t> & data) {
    uint64_t hash = 14695981039346656037ull;
    for (const uint8_t value : data) {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    return hash;
}

bool run_backend_self_test() {
    constexpr uint32_t logical_expert_count = 4;
    constexpr uint32_t resident_slot_count = 2;
    constexpr size_t expert_elements = 8 * 4;
    const size_t expert_bytes = expert_elements * sizeof(float);

    edge_moe_resident_slots slots(logical_expert_count, resident_slot_count, expert_bytes);
    std::vector<float> weights(logical_expert_count * expert_elements);
    for (uint32_t expert = 0; expert < logical_expert_count; ++expert) {
        for (size_t i = 0; i < expert_elements; ++i) {
            weights[expert * expert_elements + i] = 0.01f * static_cast<float>(expert + 1) + static_cast<float>(i);
        }
    }

    std::string error;
    if (!slots.load_blocking(1, weights.data() + expert_elements, expert_bytes, 1, error) ||
        !slots.load_blocking(3, weights.data() + 3 * expert_elements, expert_bytes, 1, error)) {
        std::fprintf(stderr, "self-test: resident slot load failed: %s\n", error.c_str());
        return false;
    }

    uint32_t physical = 0;
    if (!slots.resolve(1, physical) || physical != 0 || !slots.resolve(3, physical) || physical != 1) {
        std::fprintf(stderr, "self-test: logical-to-physical mapping failed\n");
        return false;
    }
    if (std::memcmp(slots.slot_data(0), weights.data() + expert_elements, expert_bytes) != 0 ||
        std::memcmp(slots.slot_data(1), weights.data() + 3 * expert_elements, expert_bytes) != 0) {
        std::fprintf(stderr, "self-test: resident slot data failed byte comparison\n");
        return false;
    }

    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!backend) {
        std::fprintf(stderr, "self-test: CPU backend initialization failed\n");
        return false;
    }

    ggml_init_params init_params = {};
    init_params.mem_size = 32 * 1024 * 1024;
    init_params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(init_params));
    if (!ctx) {
        std::fprintf(stderr, "self-test: ggml context initialization failed\n");
        return false;
    }

    constexpr int64_t cols = 8;
    constexpr int64_t rows = 4;
    constexpr int64_t used = 2;
    constexpr int64_t tokens = 3;
    ggml_tensor * full_weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, rows, logical_expert_count);
    ggml_tensor * slot_weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, rows, resident_slot_count);
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, used, tokens);
    ggml_tensor * logical_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, tokens);
    ggml_tensor * physical_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, tokens);
    if (full_weights == nullptr || slot_weights == nullptr || input == nullptr ||
        logical_ids == nullptr || physical_ids == nullptr) {
        std::fprintf(stderr, "self-test: tensor creation failed\n");
        return false;
    }

    const int32_t logical_id_data[used * tokens] = { 1, 3, 3, 1, 1, 3 };
    int32_t physical_id_data[used * tokens] = {};
    for (size_t i = 0; i < sizeof(logical_id_data) / sizeof(logical_id_data[0]); ++i) {
        if (!slots.resolve(static_cast<uint32_t>(logical_id_data[i]), physical)) {
            std::fprintf(stderr, "self-test: selected expert is not resident\n");
            return false;
        }
        physical_id_data[i] = static_cast<int32_t>(physical);
    }

    std::vector<float> input_data(cols * used * tokens);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = 0.02f * static_cast<float>(i + 1);
    }

    ggml_tensor * full_output = ggml_mul_mat_id(ctx.get(), full_weights, input, logical_ids);
    ggml_tensor * slot_output = ggml_mul_mat_id(ctx.get(), slot_weights, input, physical_ids);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    if (full_output == nullptr || slot_output == nullptr || graph == nullptr) {
        std::fprintf(stderr, "self-test: graph creation failed\n");
        return false;
    }
    ggml_build_forward_expand(graph, full_output);
    ggml_build_forward_expand(graph, slot_output);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        std::fprintf(stderr, "self-test: backend buffer allocation failed\n");
        return false;
    }
    ggml_backend_tensor_set(full_weights, weights.data(), 0, ggml_nbytes(full_weights));
    ggml_backend_tensor_set(slot_weights, weights.data() + expert_elements, 0, expert_bytes);
    ggml_backend_tensor_set(slot_weights, weights.data() + 3 * expert_elements, expert_bytes, expert_bytes);
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(logical_ids, logical_id_data, 0, sizeof(logical_id_data));
    ggml_backend_tensor_set(physical_ids, physical_id_data, 0, sizeof(physical_id_data));

    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "self-test: ggml_mul_mat_id graph compute failed\n");
        return false;
    }

    std::vector<float> full_output_data(ggml_nelements(full_output));
    std::vector<float> slot_output_data(ggml_nelements(slot_output));
    ggml_backend_tensor_get(full_output, full_output_data.data(), 0, ggml_nbytes(full_output));
    ggml_backend_tensor_get(slot_output, slot_output_data.data(), 0, ggml_nbytes(slot_output));
    float max_diff = 0.0f;
    for (size_t i = 0; i < full_output_data.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(full_output_data[i] - slot_output_data[i]));
    }

    std::printf("resident slots: logical 1->physical 0, logical 3->physical 1\n");
    std::printf("backend remap: ggml_mul_mat_id max_abs_diff=%.8g\n", max_diff);
    return max_diff <= 1e-5f;
}

bool run_model_probe(const options & params) {
    bool found = false;
    std::string model_path;
    edge_moe_expert_layout layout;

    for (uint32_t file_idx = 0; file_idx < params.files.size() && !found; ++file_idx) {
        const std::string & path = params.files[file_idx];
        std::error_code ec;
        const uintmax_t file_size_u = std::filesystem::file_size(path, ec);
        if (ec || file_size_u > std::numeric_limits<size_t>::max()) {
            std::fprintf(stderr, "file[%u] %s: failed to get file size\n", file_idx, path.c_str());
            return false;
        }

        gguf_init_params init_params = {};
        init_params.no_alloc = true;
        init_params.ctx = nullptr;
        gguf_context_ptr ctx(gguf_init_from_file(path.c_str(), init_params));
        if (!ctx) {
            std::fprintf(stderr, "file[%u] %s: failed to read GGUF metadata\n", file_idx, path.c_str());
            return false;
        }

        const size_t data_offset = gguf_get_data_offset(ctx.get());
        for (int64_t tensor_id = 0; tensor_id < gguf_get_n_tensors(ctx.get()); ++tensor_id) {
            const char * tensor_name = gguf_get_tensor_name(ctx.get(), tensor_id);
            if (tensor_name == nullptr) {
                continue;
            }
            const std::string name(tensor_name);
            if (edge_moe_tensor_part_from_name(name) != params.part) {
                continue;
            }
            int layer = -1;
            if (!edge_moe_parse_layer(name, layer) || layer != static_cast<int>(params.layer)) {
                continue;
            }

            const size_t tensor_offset = gguf_get_tensor_offset(ctx.get(), tensor_id);
            if (data_offset > std::numeric_limits<size_t>::max() - tensor_offset) {
                std::fprintf(stderr, "tensor=%s: file offset overflows size_t\n", name.c_str());
                return false;
            }
            edge_moe_tensor_metadata metadata;
            metadata.file_idx = file_idx;
            metadata.name = name;
            metadata.file_offset = data_offset + tensor_offset;
            metadata.tensor_size = gguf_get_tensor_size(ctx.get(), tensor_id);
            metadata.type = gguf_get_tensor_type(ctx.get(), tensor_id);
            const int64_t * ne = gguf_get_tensor_ne(ctx.get(), tensor_id);
            if (ne == nullptr) {
                std::fprintf(stderr, "tensor=%s: missing shape metadata\n", name.c_str());
                return false;
            }
            std::copy(ne, ne + GGML_MAX_DIMS, metadata.ne.begin());

            std::string error;
            if (!edge_moe_build_expert_layout(metadata, static_cast<size_t>(file_size_u), layout, error)) {
                std::fprintf(stderr, "tensor=%s: invalid expert layout: %s\n", name.c_str(), error.c_str());
                return false;
            }
            model_path = path;
            found = true;
            break;
        }
    }

    if (!found) {
        std::fprintf(stderr, "no tensor found for layer=%u part=%s\n",
            params.layer, edge_moe_tensor_part_name(params.part));
        return false;
    }

    uint32_t slot_count = params.slots == 0 ? static_cast<uint32_t>(params.experts.size()) : params.slots;
    if (slot_count == 0) {
        std::fprintf(stderr, "slot count must be greater than zero\n");
        return false;
    }
    edge_moe_resident_slots slots(layout.expert_count, slot_count, layout.expert_stride);
    std::printf("tensor=%s layer=%u part=%s expert_count=%u slot_count=%u slot_bytes=%zu\n",
        layout.tensor.name.c_str(), params.layer, edge_moe_tensor_part_name(layout.part),
        layout.expert_count, slot_count, layout.expert_stride);

    size_t loaded = 0;
    for (const uint32_t expert : params.experts) {
        if (expert >= layout.expert_count) {
            std::fprintf(stderr, "expert %u is outside [0, %u)\n", expert, layout.expert_count);
            return false;
        }
        std::vector<uint8_t> data;
        if (!read_range(model_path, layout.ranges[expert], data)) {
            std::fprintf(stderr, "expert %u: failed to read source range\n", expert);
            return false;
        }
        std::string error;
        if (!slots.load_blocking(expert, data.data(), data.size(), loaded, error)) {
            std::fprintf(stderr, "expert %u: blocking load failed: %s\n", expert, error.c_str());
            return false;
        }
        uint32_t physical_slot = 0;
        if (!slots.resolve(expert, physical_slot) ||
            std::memcmp(slots.slot_data(physical_slot), data.data(), data.size()) != 0) {
            std::fprintf(stderr, "expert %u: resident slot data mismatch\n", expert);
            return false;
        }
        if (!slots.begin_use(expert, loaded, physical_slot, error) || !slots.end_use(expert, error)) {
            std::fprintf(stderr, "expert %u: use lifetime check failed: %s\n", expert, error.c_str());
            return false;
        }
        std::printf("  logical_expert=%u physical_slot=%u offset=%zu size=%zu hash=0x%016" PRIx64 " state=resident\n",
            expert, physical_slot, layout.ranges[expert].offset, data.size(), fnv1a64(data));
        ++loaded;
    }
    std::printf("summary: loaded=%zu slots=%u errors=0\n", loaded, slot_count);
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
        return run_backend_self_test() ? 0 : 1;
    }
    return run_model_probe(params) ? 0 : 1;
}
