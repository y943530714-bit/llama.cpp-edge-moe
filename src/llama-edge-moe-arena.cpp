#include "llama-edge-moe-arena.h"

#include "llama-edge-moe-io.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

enum class arena_part : uint8_t {
    gate,
    up,
    gate_up,
    down,
    count,
};

constexpr size_t part_count = static_cast<size_t>(arena_part::count);

size_t part_index(const arena_part part) {
    return static_cast<size_t>(part);
}

const char * part_name(const arena_part part) {
    switch (part) {
        case arena_part::gate:    return "gate";
        case arena_part::up:      return "up";
        case arena_part::gate_up: return "gate_up";
        case arena_part::down:    return "down";
        case arena_part::count:   break;
    }
    return "unknown";
}

bool checked_mul(const size_t lhs, const size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool checked_add(const size_t lhs, const size_t rhs, size_t & result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool parse_layer(const char * name, uint32_t & layer) {
    constexpr const char prefix[] = "ffn_moe_arena_ids-";
    if (name == nullptr || std::strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return false;
    }

    const char * value = name + sizeof(prefix) - 1;
    if (*value == '\0') {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }

    layer = static_cast<uint32_t>(parsed);
    return true;
}

struct tensor_signature {
    arena_part part;
    ggml_type type;
    int64_t ne0;
    int64_t ne1;
    size_t nb0;
    size_t nb1;

    bool operator==(const tensor_signature & other) const {
        return part == other.part && type == other.type && ne0 == other.ne0 && ne1 == other.ne1 && nb0 == other.nb0 && nb1 == other.nb1;
    }
};

} // namespace

struct llama_edge_moe_arena::impl {
    struct source {
        const ggml_tensor * tensor = nullptr;
        uint32_t file_index = 0;
        size_t offset = 0;
        size_t size = 0;
    };

    struct layer_sources {
        std::array<source, part_count> tensors = {};
    };

    struct variant {
        tensor_signature signature;
        ggml_tensor * tensor = nullptr;
    };

    struct slot {
        int64_t key = -1;
        uint64_t last_used = 0;
        uint32_t refcnt = 0;
        bool hot = false;
        bool transient = false;
        bool prefill_admitted = false;
        bool loading = false;
        bool prefetched = false;
    };

    struct pending_load {
        uint32_t logical = 0;
        uint32_t physical = 0;
        size_t key = 0;
    };

    struct phase_counters {
        uint64_t batches = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t bytes_read = 0;
        uint64_t bytes_transferred = 0;
        uint64_t io_requests = 0;
        uint64_t io_wait_us = 0;
        uint64_t read_calls = 0;
        uint64_t experts_loaded = 0;
        uint64_t prefetch_loaded = 0;
        uint64_t prefetch_useful = 0;
        uint64_t prefetch_wasted = 0;
        uint64_t prefetch_wait_us = 0;
    };

    struct full_layer_result {
        bool ok = false;
        std::string error;
        uint64_t io_requests = 0;
        uint64_t bytes_transferred = 0;
    };

    const uint32_t n_layer;
    const uint32_t n_expert;
    const size_t budget_bytes;
    const bool direct_io;
    const bool layered_cache;
    const bool prefill_full_layer_enabled;
    const bool decode_prefetch_enabled;
    const uint32_t configured_hot_slots_per_layer;
    const std::vector<uint32_t> configured_hot_slots_by_layer;
    const bool profile_batches;
    size_t allocated_bytes = 0;

    std::vector<layer_sources> layers;
    std::array<size_t, part_count> part_strides = {};
    std::array<ggml_backend_buffer_ptr, part_count> buffers;
    std::vector<variant> variants;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> replacements;

    ggml_context_ptr tensor_ctx;
    std::vector<slot> slots;
    std::vector<int32_t> key_to_slot;
    std::vector<uint32_t> active_slots;
    std::vector<ggml_tensor *> router_weights;
    std::vector<std::string> reader_paths;
    std::unique_ptr<llama_edge_moe_reader> reader;
    size_t hot_count = 0;
    size_t hot_capacity = 0;
    size_t resident_capacity = 0;
    size_t scratch_capacity = 0;
    size_t prefetch_capacity = 0;
    size_t cold_capacity = 0;
    std::vector<uint32_t> layer_slot_begin;
    std::vector<uint32_t> layer_slot_count;
    std::vector<uint32_t> layer_hot_capacity;
    std::vector<uint32_t> layer_hot_count;
    std::vector<uint32_t> layer_hot_begin;
    std::vector<uint32_t> layer_dynamic_begin;
    std::vector<uint32_t> layer_dynamic_count;
    uint32_t staging_begin = 0;
    bool full_prefill_decided = false;
    bool full_prefill_active = false;
    int32_t pending_full_layer = -1;
    int32_t pending_full_buffer = -1;
    std::future<full_layer_result> full_layer_future;
    int32_t pending_decode_prefetch_layer = -1;
    std::future<full_layer_result> decode_prefetch_future;
    std::vector<pending_load> pending_decode_prefetch_loads;
    bool prefill = false;
    bool prefill_sequence = false;
    std::vector<uint32_t> prefill_frequency;
    std::vector<float> prefill_score;
    std::vector<uint8_t> decode_frequency;
    std::vector<float> request_score;
    std::vector<uint32_t> request_score_epoch;
    std::vector<uint16_t> route_transitions;
    std::vector<int32_t> previous_route_ids;
    int32_t previous_route_layer = -1;
    size_t previous_route_width = 0;
    size_t previous_route_tokens = 0;
    uint32_t decode_epoch = 0;
    phase_counters prefill_stats;
    phase_counters decode_stats;
    phase_counters batch_start_stats;

    uint64_t clock = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t io_wait_us = 0;
    uint64_t promotions = 0;
    uint64_t demotions = 0;
    uint64_t prefill_admissions = 0;
    uint64_t prefill_drops = 0;
    uint64_t transient_reclaims = 0;
    uint64_t decode_admissions = 0;
    uint64_t decode_admission_rejections = 0;
    uint64_t decode_resident_fallbacks = 0;
    uint64_t decode_prefetch_loaded = 0;
    uint64_t decode_prefetch_useful = 0;
    uint64_t decode_prefetch_wasted = 0;
    uint64_t decode_prefetch_wait_us = 0;
    std::string error;

    impl(
            const llama_model & model,
            ggml_backend * backend_cpu,
            const size_t budget,
            const bool use_direct_io,
            const size_t io_depth,
            const bool use_layered_cache,
            const bool use_prefill_full_layer,
            const bool use_decode_prefetch,
            const uint32_t hot_slots_per_layer,
            const std::vector<uint32_t> & hot_slots_by_layer) :
        n_layer(model.hparams.n_layer()),
        n_expert(model.hparams.n_expert),
        budget_bytes(budget),
        direct_io(use_direct_io),
        layered_cache(use_direct_io && use_layered_cache),
        prefill_full_layer_enabled(use_direct_io && use_layered_cache && use_prefill_full_layer),
        decode_prefetch_enabled(use_direct_io && use_layered_cache && use_decode_prefetch),
        configured_hot_slots_per_layer(hot_slots_per_layer),
        configured_hot_slots_by_layer(hot_slots_by_layer),
        profile_batches(use_direct_io && std::getenv("LLAMA_EDGE_MOE_PROFILE") != nullptr),
        layers(n_layer) {
        if (backend_cpu == nullptr || n_layer == 0 || n_expert == 0 || budget_bytes == 0) {
            throw std::invalid_argument("invalid edge MoE arena configuration");
        }
        for (uint32_t il = 0; il < n_layer; ++il) {
            ggml_backend_dev_t device = model.dev_layer(il);
            if (device != nullptr && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::runtime_error("edge MoE arena supports CPU-only model layers");
            }
            const llama_layer & layer = model.layers[il];
            add_source(model, il, arena_part::gate,    layer.ffn_gate_exps);
            add_source(model, il, arena_part::up,      layer.ffn_up_exps);
            add_source(model, il, arena_part::gate_up, layer.ffn_gate_up_exps);
            add_source(model, il, arena_part::down,    layer.ffn_down_exps);

            const bool have_input = layer.ffn_gate_up_exps != nullptr ||
                (layer.ffn_gate_exps != nullptr && layer.ffn_up_exps != nullptr);
            if (!have_input || layer.ffn_down_exps == nullptr) {
                throw std::runtime_error("edge MoE arena requires complete routed expert tensors in every layer");
            }
        }

        size_t slot_bytes = 0;
        for (const size_t stride : part_strides) {
            if (!checked_add(slot_bytes, stride, slot_bytes)) {
                throw std::length_error("edge MoE arena slot size overflows size_t");
            }
        }
        if (slot_bytes == 0) {
            throw std::runtime_error("edge MoE arena found no routed expert tensors");
        }

        size_t key_count = 0;
        if (!checked_mul(n_layer, n_expert, key_count)) {
            throw std::length_error("edge MoE arena key count overflows size_t");
        }
        const size_t slot_count_size = std::min(budget_bytes / slot_bytes, key_count);
        if (slot_count_size < n_expert) {
            throw std::runtime_error("edge MoE arena budget is too small for one layer of experts");
        }
        if (slot_count_size > static_cast<size_t>(INT32_MAX)) {
            throw std::runtime_error("edge MoE arena slot count exceeds int32 range");
        }
        const uint32_t slot_count = static_cast<uint32_t>(slot_count_size);
        if (layered_cache && slot_count < n_layer) {
            throw std::runtime_error("edge MoE layered cache requires at least one expert slot per layer");
        }

        if (direct_io) {
            reader = std::make_unique<llama_edge_moe_reader>(reader_paths, io_depth);
        }

        const size_t ctx_size = std::max<size_t>(1024 * 1024, (variants.size() + 1) * ggml_tensor_overhead() * 2);
        tensor_ctx.reset(ggml_init({ctx_size, nullptr, true}));
        if (!tensor_ctx) {
            throw std::runtime_error("failed to create edge MoE arena tensor context");
        }

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_cpu);
        if (!buft || !ggml_backend_buft_is_host(buft)) {
            throw std::runtime_error("edge MoE arena requires a host CPU buffer");
        }

        for (size_t i = 0; i < part_count; ++i) {
            if (part_strides[i] == 0) {
                continue;
            }
            size_t buffer_size = 0;
            if (!checked_mul(part_strides[i], slot_count, buffer_size)) {
                throw std::length_error("edge MoE arena buffer size overflows size_t");
            }
            buffers[i].reset(ggml_backend_buft_alloc_buffer(buft, buffer_size));
            if (!buffers[i]) {
                throw std::runtime_error("failed to allocate edge MoE arena buffer");
            }
            ggml_backend_buffer_set_usage(buffers[i].get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            allocated_bytes += ggml_backend_buffer_get_size(buffers[i].get());
        }

        for (variant & item : variants) {
            const tensor_signature & sig = item.signature;
            const size_t index = part_index(sig.part);
            ggml_tensor * tensor = ggml_new_tensor_3d(tensor_ctx.get(), sig.type, sig.ne0, sig.ne1, slot_count);
            tensor->nb[0] = sig.nb0;
            tensor->nb[1] = sig.nb1;
            tensor->nb[2] = part_strides[index];
            tensor->nb[3] = part_strides[index] * slot_count;
            ggml_format_name(tensor, "edge_moe_arena_%s_%s", part_name(sig.part), ggml_type_name(sig.type));

            void * base = ggml_backend_buffer_get_base(buffers[index].get());
            if (ggml_backend_tensor_alloc(buffers[index].get(), tensor, base) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("failed to attach an edge MoE arena tensor");
            }
            item.tensor = tensor;
        }

        for (uint32_t il = 0; il < n_layer; ++il) {
            for (size_t i = 0; i < part_count; ++i) {
                const ggml_tensor * source_tensor = layers[il].tensors[i].tensor;
                if (source_tensor == nullptr) {
                    continue;
                }
                const arena_part part = static_cast<arena_part>(i);
                const tensor_signature signature = make_signature(part, source_tensor);
                auto it = std::find_if(variants.begin(), variants.end(), [&](const variant & item) {
                    return item.signature == signature;
                });
                if (it == variants.end() || it->tensor == nullptr) {
                    throw std::runtime_error("edge MoE arena tensor variant is missing");
                }
                replacements.emplace(source_tensor, it->tensor);
            }
        }

        slots.resize(slot_count);
        key_to_slot.resize(key_count, -1);
        router_weights.resize(n_layer, nullptr);
        if (layered_cache) {
            layer_slot_begin.resize(n_layer);
            layer_slot_count.resize(n_layer);
            layer_hot_capacity.resize(n_layer);
            layer_hot_count.resize(n_layer, 0);
            layer_hot_begin.resize(n_layer);
            layer_dynamic_begin.resize(n_layer);
            layer_dynamic_count.resize(n_layer);

            const uint32_t base = slot_count / n_layer;
            const uint32_t remainder = slot_count % n_layer;
            if (!configured_hot_slots_by_layer.empty() && configured_hot_slots_by_layer.size() != n_layer) {
                throw std::runtime_error("edge MoE per-layer hot quota count must match the model layer count");
            }
            if (configured_hot_slots_per_layer >= base && configured_hot_slots_per_layer != 0) {
                throw std::runtime_error("edge MoE hot quota must leave at least one dynamic slot per layer");
            }
            for (uint32_t layer = 0; layer < n_layer; ++layer) {
                const uint32_t count = base + (layer < remainder ? 1 : 0);
                const uint32_t hot = !configured_hot_slots_by_layer.empty() ?
                    configured_hot_slots_by_layer[layer] : configured_hot_slots_per_layer == 0 ?
                    (count * 3 + 3) / 4 : configured_hot_slots_per_layer + (layer < remainder ? 1 : 0);
                if (hot >= count) {
                    throw std::runtime_error("edge MoE per-layer hot quota must leave at least one dynamic slot");
                }
                const uint32_t dynamic = count - hot;
                const uint32_t prefetch = std::min<uint32_t>(4, dynamic);
                layer_slot_count[layer] = count;
                layer_hot_capacity[layer] = hot;
                layer_dynamic_count[layer] = dynamic;
                hot_capacity += hot;
                prefetch_capacity += prefetch;
                cold_capacity += dynamic - prefetch;
            }
            resident_capacity = hot_capacity;
            scratch_capacity = slot_count_size - hot_capacity;

            if (prefill_full_layer_enabled) {
                if (scratch_capacity < 2 * static_cast<size_t>(n_expert)) {
                    throw std::runtime_error("edge MoE full-layer prefill requires two expert-layer staging buffers");
                }
                uint32_t hot_offset = 0;
                uint32_t dynamic_offset = static_cast<uint32_t>(hot_capacity);
                for (uint32_t layer = 0; layer < n_layer; ++layer) {
                    layer_hot_begin[layer] = hot_offset;
                    layer_dynamic_begin[layer] = dynamic_offset;
                    hot_offset += layer_hot_capacity[layer];
                    dynamic_offset += layer_dynamic_count[layer];
                }
                staging_begin = static_cast<uint32_t>(hot_capacity);
            } else {
                uint32_t offset = 0;
                for (uint32_t layer = 0; layer < n_layer; ++layer) {
                    layer_slot_begin[layer] = offset;
                    layer_hot_begin[layer] = offset;
                    layer_dynamic_begin[layer] = offset + layer_hot_capacity[layer];
                    offset += layer_slot_count[layer];
                }
            }
        } else if (direct_io) {
            scratch_capacity = std::min<size_t>({n_expert, 64, slot_count_size});
            resident_capacity = slot_count_size - scratch_capacity;
            hot_capacity = resident_capacity;
        } else {
            hot_capacity = std::max<size_t>(n_expert, slot_count_size * 3 / 4);
            resident_capacity = slot_count_size;
        }
        prefill_frequency.resize(key_count, 0);
        prefill_score.resize(key_count, 0.0f);
        decode_frequency.resize(key_count, 0);
        request_score.resize(key_count, 0.0f);
        request_score_epoch.resize(key_count, 0);
        if (decode_prefetch_enabled && n_layer > 1) {
            size_t transition_count = 0;
            size_t layer_pair_count = 0;
            if (!checked_mul(n_expert, n_expert, layer_pair_count) ||
                !checked_mul(layer_pair_count, n_layer - 1, transition_count)) {
                throw std::length_error("edge MoE route transition table overflows size_t");
            }
            route_transitions.resize(transition_count, 0);
        }

        LLAMA_LOG_INFO("%s: mode = %s, budget = %.2f MiB, allocated = %.2f MiB, slot size = %.2f MiB, slots = %u, resident slots = %zu, scratch slots = %zu, I/O depth = %zu\n",
                __func__, direct_io ? "unbuffered" : "mmap-copy",
                budget_bytes / 1024.0 / 1024.0, allocated_bytes / 1024.0 / 1024.0,
                slot_bytes / 1024.0 / 1024.0, slot_count, resident_capacity, scratch_capacity,
                direct_io ? io_depth : 0);
        if (layered_cache) {
            const auto [min_slots, max_slots] = std::minmax_element(layer_slot_count.begin(), layer_slot_count.end());
            LLAMA_LOG_INFO("%s: layered cache = enabled, slots/layer = %u-%u, hot = %zu, prefetch reserve = %zu, cold = %zu (prefetch reserve is shared until a predictor is enabled)\n",
                    __func__, *min_slots, *max_slots, hot_capacity, prefetch_capacity, cold_capacity);
            if (configured_hot_slots_per_layer != 0) {
                LLAMA_LOG_INFO("%s: configured hot quota = %u per normal layer\n",
                        __func__, configured_hot_slots_per_layer);
            }
            if (!configured_hot_slots_by_layer.empty()) {
                const auto [min_hot, max_hot] = std::minmax_element(
                    configured_hot_slots_by_layer.begin(), configured_hot_slots_by_layer.end());
                LLAMA_LOG_INFO("%s: configured per-layer hot quotas = %u-%u, total = %zu\n",
                        __func__, *min_hot, *max_hot, hot_capacity);
            }
            if (prefill_full_layer_enabled) {
                LLAMA_LOG_INFO("%s: long-prefill full-layer pipeline = enabled, staging buffers = 2 x %u slots, threshold = 512 tokens\n",
                        __func__, n_expert);
            }
            if (decode_prefetch_enabled) {
                LLAMA_LOG_INFO("%s: decode next-layer prefetch = enabled, max experts/layer = 1\n", __func__);
            }
        }
    }

    ~impl() {
        drain_decode_prefetch();
        drain_full_layer();
        const double hit_rate = hits + misses == 0 ? 0.0 : 100.0 * hits / static_cast<double>(hits + misses);
        LLAMA_LOG_INFO("%s: hits = %llu, misses = %llu, hit rate = %.2f%%, evictions = %llu, promotions = %llu, demotions = %llu, bytes read = %.2f MiB, I/O wait = %.2f ms\n",
                __func__,
                static_cast<unsigned long long>(hits),
                static_cast<unsigned long long>(misses),
                hit_rate,
                static_cast<unsigned long long>(evictions),
                static_cast<unsigned long long>(promotions),
                static_cast<unsigned long long>(demotions),
                bytes_read / 1024.0 / 1024.0,
                io_wait_us / 1000.0);
        if (reader) {
            const llama_edge_moe_io_stats & stats = reader->stats();
            LLAMA_LOG_INFO("%s: direct I/O requests = %llu, requested = %.2f MiB, transferred = %.2f MiB, peak bounce = %.2f MiB\n",
                    __func__,
                    static_cast<unsigned long long>(stats.requests),
                    stats.bytes_requested / 1024.0 / 1024.0,
                    stats.bytes_transferred / 1024.0 / 1024.0,
                    stats.peak_bounce_bytes / 1024.0 / 1024.0);
        }
        log_phase("prefill", prefill_stats);
        log_phase("decode/verify", decode_stats);
        LLAMA_LOG_INFO("%s: prefill admissions = %llu, replaced admissions = %llu, transient reclaims = %llu, decode admissions = %llu, decode admission rejections = %llu, decode resident fallbacks = %llu\n",
                __func__,
                static_cast<unsigned long long>(prefill_admissions),
                static_cast<unsigned long long>(prefill_drops),
                static_cast<unsigned long long>(transient_reclaims),
                static_cast<unsigned long long>(decode_admissions),
                static_cast<unsigned long long>(decode_admission_rejections),
                static_cast<unsigned long long>(decode_resident_fallbacks));
        if (decode_prefetch_enabled) {
            const double accuracy = decode_prefetch_loaded == 0 ? 0.0 :
                100.0 * decode_prefetch_useful / static_cast<double>(decode_prefetch_loaded);
            LLAMA_LOG_INFO("%s: decode prefetch loaded = %llu, useful = %llu, wasted = %llu, accuracy = %.2f%%, foreground wait = %.2f ms\n",
                    __func__,
                    static_cast<unsigned long long>(decode_prefetch_loaded),
                    static_cast<unsigned long long>(decode_prefetch_useful),
                    static_cast<unsigned long long>(decode_prefetch_wasted),
                    accuracy,
                    decode_prefetch_wait_us / 1000.0);
        }
    }

    void log_phase(const char * name, const phase_counters & stats) const {
        const double hit_rate = stats.hits + stats.misses == 0 ?
            0.0 : 100.0 * stats.hits / static_cast<double>(stats.hits + stats.misses);
        LLAMA_LOG_INFO("%s: %s batches = %llu, hits = %llu, misses = %llu, hit rate = %.2f%%, evictions = %llu, requests = %llu, requested = %.2f MiB, transferred = %.2f MiB, I/O wait = %.2f ms, read calls = %llu, loaded experts = %llu\n",
                __func__, name,
                static_cast<unsigned long long>(stats.batches),
                static_cast<unsigned long long>(stats.hits),
                static_cast<unsigned long long>(stats.misses),
                hit_rate,
                static_cast<unsigned long long>(stats.evictions),
                static_cast<unsigned long long>(stats.io_requests),
                stats.bytes_read / 1024.0 / 1024.0,
                stats.bytes_transferred / 1024.0 / 1024.0,
                stats.io_wait_us / 1000.0,
                static_cast<unsigned long long>(stats.read_calls),
                static_cast<unsigned long long>(stats.experts_loaded));
    }

    phase_counters & current_stats() {
        return prefill ? prefill_stats : decode_stats;
    }

    void set_router_weights(const uint32_t layer, ggml_tensor * weights) {
        if (layer < router_weights.size()) {
            router_weights[layer] = weights;
        }
    }

    float routed_weight(const uint32_t layer, const size_t index, const size_t count) const {
        const ggml_tensor * weights = layer < router_weights.size() ? router_weights[layer] : nullptr;
        if (weights == nullptr || weights->type != GGML_TYPE_F32 || weights->data == nullptr ||
                !ggml_is_contiguous(weights) || static_cast<size_t>(ggml_nelements(weights)) != count) {
            return 1.0f;
        }
        const float value = static_cast<const float *>(weights->data)[index];
        return std::isfinite(value) && value > 0.0f ? value : 0.0f;
    }

    float decayed_request_score(const size_t key) const {
        constexpr float decay = 0.70f;
        const uint32_t age = decode_epoch - request_score_epoch[key];
        return request_score[key] * std::pow(decay, static_cast<float>(std::min<uint32_t>(age, 64)));
    }

    void add_request_score(const size_t key, const float score) {
        request_score[key] = decayed_request_score(key) + score;
        request_score_epoch[key] = decode_epoch;
    }

    tensor_signature make_signature(const arena_part part, const ggml_tensor * tensor) const {
        return {part, tensor->type, tensor->ne[0], tensor->ne[1], tensor->nb[0], tensor->nb[1]};
    }

    void add_source(
            const llama_model & model,
            const uint32_t il,
            const arena_part part,
            const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return;
        }
        if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != n_expert || tensor->ne[3] != 1) {
            throw std::runtime_error("edge MoE arena requires 3-D expert tensors with a common expert count");
        }
        if (tensor->buffer != nullptr && std::strcmp(
                    ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer)), "CPU_REPACK") == 0) {
            throw std::runtime_error("edge MoE arena requires the model to be loaded with use_extra_bufts=false");
        }
        if (tensor->nb[2] != tensor->nb[1] * static_cast<size_t>(tensor->ne[1])) {
            throw std::runtime_error("edge MoE arena requires contiguous expert slices");
        }

        const size_t index = part_index(part);
        source & item = layers[il].tensors[index];
        item.tensor = tensor;
        if (direct_io) {
            const llama_edge_moe_source * descriptor = model.edge_moe_source(tensor);
            if (descriptor == nullptr || descriptor->path.empty() || descriptor->size != ggml_nbytes(tensor)) {
                throw std::runtime_error("edge MoE arena is missing a valid direct-I/O source for " + std::string(tensor->name));
            }
            auto path = std::find(reader_paths.begin(), reader_paths.end(), descriptor->path);
            if (path == reader_paths.end()) {
                reader_paths.push_back(descriptor->path);
                path = reader_paths.end() - 1;
            }
            item.file_index = static_cast<uint32_t>(path - reader_paths.begin());
            item.offset = descriptor->offset;
            item.size = descriptor->size;
        }
        part_strides[index] = std::max(part_strides[index], tensor->nb[2]);

        const tensor_signature signature = make_signature(part, tensor);
        if (std::none_of(variants.begin(), variants.end(), [&](const variant & item) {
                return item.signature == signature;
            })) {
            variants.push_back({signature, nullptr});
        }
    }

    void clear_slot(const uint32_t physical) {
        slot & item = slots[physical];
        if (item.prefetched) {
            ++decode_prefetch_wasted;
            ++decode_stats.prefetch_wasted;
        }
        if (item.key >= 0) {
            key_to_slot[static_cast<size_t>(item.key)] = -1;
        }
        if (item.hot) {
            --hot_count;
            if (layered_cache && item.key >= 0) {
                const size_t layer = static_cast<size_t>(item.key) / n_expert;
                if (layer < layer_hot_count.size() && layer_hot_count[layer] > 0) {
                    --layer_hot_count[layer];
                }
            }
        }
        item = {};
    }

    void demote_slot(const uint32_t physical) {
        slot & item = slots[physical];
        if (!item.hot) {
            return;
        }
        item.hot = false;
        --hot_count;
        if (layered_cache && item.key >= 0) {
            const size_t layer = static_cast<size_t>(item.key) / n_expert;
            if (layer < layer_hot_count.size() && layer_hot_count[layer] > 0) {
                --layer_hot_count[layer];
            }
        }
        ++demotions;
    }

    void begin_batch(const bool is_prefill) {
        drain_decode_prefetch();
        drain_full_layer();
        release_active();
        const bool next_prefill = direct_io && is_prefill;
        if (next_prefill && !prefill_sequence) {
            for (slot & item : slots) {
                item.hot = false;
            }
            hot_count = 0;
            std::fill(layer_hot_count.begin(), layer_hot_count.end(), 0);
            std::fill(decode_frequency.begin(), decode_frequency.end(), 0);
            std::fill(request_score.begin(), request_score.end(), 0.0f);
            std::fill(request_score_epoch.begin(), request_score_epoch.end(), 0);
            std::fill(route_transitions.begin(), route_transitions.end(), 0);
            previous_route_ids.clear();
            previous_route_layer = -1;
            previous_route_width = 0;
            previous_route_tokens = 0;
            decode_epoch = 0;
        }
        if (next_prefill) {
            for (slot & item : slots) {
                item.prefill_admitted = false;
            }
            std::fill(prefill_frequency.begin(), prefill_frequency.end(), 0);
            std::fill(prefill_score.begin(), prefill_score.end(), 0.0f);
        } else if (!next_prefill && prefill_sequence) {
            for (slot & item : slots) {
                item.prefill_admitted = false;
            }
        }
        if (!next_prefill) {
            ++decode_epoch;
        }
        prefill_sequence = next_prefill;
        prefill = next_prefill;
        full_prefill_decided = false;
        full_prefill_active = false;
        phase_counters & stats = current_stats();
        batch_start_stats = stats;
        ++stats.batches;
    }

    void end_batch() {
        drain_decode_prefetch();
        drain_full_layer();
        release_active();
        if (profile_batches) {
            const phase_counters & stats = current_stats();
            LLAMA_LOG_INFO("%s: EDGEMOE phase=%s batches=%llu hits=%llu misses=%llu evictions=%llu requests=%llu requested_bytes=%llu transferred_bytes=%llu io_wait_us=%llu read_calls=%llu loaded_experts=%llu prefetch_loaded=%llu prefetch_useful=%llu prefetch_wasted=%llu prefetch_wait_us=%llu\n",
                    __func__, prefill ? "prefill" : "decode/verify",
                    static_cast<unsigned long long>(stats.batches - batch_start_stats.batches),
                    static_cast<unsigned long long>(stats.hits - batch_start_stats.hits),
                    static_cast<unsigned long long>(stats.misses - batch_start_stats.misses),
                    static_cast<unsigned long long>(stats.evictions - batch_start_stats.evictions),
                    static_cast<unsigned long long>(stats.io_requests - batch_start_stats.io_requests),
                    static_cast<unsigned long long>(stats.bytes_read - batch_start_stats.bytes_read),
                    static_cast<unsigned long long>(stats.bytes_transferred - batch_start_stats.bytes_transferred),
                    static_cast<unsigned long long>(stats.io_wait_us - batch_start_stats.io_wait_us),
                    static_cast<unsigned long long>(stats.read_calls - batch_start_stats.read_calls),
                    static_cast<unsigned long long>(stats.experts_loaded - batch_start_stats.experts_loaded),
                    static_cast<unsigned long long>(stats.prefetch_loaded - batch_start_stats.prefetch_loaded),
                    static_cast<unsigned long long>(stats.prefetch_useful - batch_start_stats.prefetch_useful),
                    static_cast<unsigned long long>(stats.prefetch_wasted - batch_start_stats.prefetch_wasted),
                    static_cast<unsigned long long>(stats.prefetch_wait_us - batch_start_stats.prefetch_wait_us));
        }
        prefill = false;
    }

    void release_active() {
        for (const uint32_t physical : active_slots) {
            if (physical < slots.size() && slots[physical].refcnt > 0) {
                --slots[physical].refcnt;
            }
        }
        for (const uint32_t physical : active_slots) {
            if (physical < slots.size() && slots[physical].refcnt == 0 && slots[physical].transient) {
                clear_slot(physical);
                ++transient_reclaims;
            }
        }
        active_slots.clear();
    }

    bool fail(const char * message) {
        error = message;
        LLAMA_LOG_ERROR("%s: %s\n", __func__, message);
        return false;
    }

    int find_victim_in_range(const size_t begin, const size_t count, const bool protect_hot) const {
        int cold_victim = -1;
        int hot_victim = -1;
        uint64_t cold_oldest = std::numeric_limits<uint64_t>::max();
        uint64_t hot_oldest = std::numeric_limits<uint64_t>::max();
        const size_t end = std::min(slots.size(), begin + count);
        for (size_t i = begin; i < end; ++i) {
            const slot & item = slots[i];
            if (item.refcnt != 0 || item.loading) {
                continue;
            }
            if (item.key < 0) {
                return static_cast<int>(i);
            }
            if (!item.hot && item.last_used < cold_oldest) {
                cold_oldest = item.last_used;
                cold_victim = static_cast<int>(i);
            } else if (item.hot && item.last_used < hot_oldest) {
                hot_oldest = item.last_used;
                hot_victim = static_cast<int>(i);
            }
        }
        return cold_victim >= 0 ? cold_victim : (protect_hot ? -1 : hot_victim);
    }

    int find_victim(const bool protect_hot) const {
        return find_victim_in_range(0, slots.size(), protect_hot);
    }

    int find_layer_victim(const uint32_t layer, const bool protect_hot) const {
        if (prefill_full_layer_enabled) {
            const int dynamic = find_victim_in_range(layer_dynamic_begin[layer], layer_dynamic_count[layer], protect_hot);
            if (dynamic >= 0) {
                return dynamic;
            }
            return find_victim_in_range(layer_hot_begin[layer], layer_hot_capacity[layer], protect_hot);
        }
        return find_victim_in_range(layer_slot_begin[layer], layer_slot_count[layer], protect_hot);
    }

    int find_layer_hot_victim(const uint32_t layer) const {
        return find_victim_in_range(layer_hot_begin[layer], layer_hot_capacity[layer], false);
    }

    bool slot_belongs_to_layer(const size_t physical, const uint32_t layer) const {
        if (!prefill_full_layer_enabled) {
            return physical >= layer_slot_begin[layer] && physical < layer_slot_begin[layer] + layer_slot_count[layer];
        }
        const bool in_hot = physical >= layer_hot_begin[layer] &&
            physical < layer_hot_begin[layer] + layer_hot_capacity[layer];
        const bool in_dynamic = physical >= layer_dynamic_begin[layer] &&
            physical < layer_dynamic_begin[layer] + layer_dynamic_count[layer];
        return in_hot || in_dynamic;
    }

    int find_transient_victim() const {
        return find_victim(true);
    }

    size_t prefill_resident_limit(const uint32_t layer) const {
        if (layered_cache) {
            return layer_hot_capacity[layer];
        }
        const size_t base = resident_capacity / n_layer;
        return base + (layer < resident_capacity % n_layer ? 1 : 0);
    }

    void promote(const uint32_t physical, const size_t key) {
        slot & selected = slots[physical];
        if (selected.hot) {
            return;
        }
        selected.hot = true;
        ++hot_count;
        const size_t layer = key / n_expert;
        if (layered_cache) {
            ++layer_hot_count[layer];
        }
        ++promotions;

        if ((!layered_cache && hot_count <= hot_capacity) ||
                (layered_cache && layer_hot_count[layer] <= layer_hot_capacity[layer])) {
            return;
        }
        int victim = -1;
        float lowest_score = std::numeric_limits<float>::infinity();
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        const size_t begin = layered_cache && !prefill_full_layer_enabled ? layer_slot_begin[layer] : 0;
        const size_t end = layered_cache && !prefill_full_layer_enabled ? begin + layer_slot_count[layer] : slots.size();
        for (size_t i = begin; i < end; ++i) {
            const slot & candidate = slots[i];
            if (layered_cache && prefill_full_layer_enabled &&
                    (candidate.key < 0 || static_cast<size_t>(candidate.key) / n_expert != layer)) {
                continue;
            }
            if (i == physical || !candidate.hot || candidate.key < 0 ||
                    (!layered_cache && (candidate.refcnt != 0 || candidate.loading))) {
                continue;
            }
            const float score = decayed_request_score(static_cast<size_t>(candidate.key));
            if (score < lowest_score || (score == lowest_score && candidate.last_used < oldest)) {
                lowest_score = score;
                oldest = candidate.last_used;
                victim = static_cast<int>(i);
            }
        }
        if (victim >= 0) {
            demote_slot(static_cast<uint32_t>(victim));
        }
    }

    bool promote_decode(const uint32_t physical, const size_t key) {
        const slot & selected = slots[physical];
        if (selected.hot) {
            return false;
        }
        const size_t layer = key / n_expert;
        const size_t current_hot = layered_cache ? layer_hot_count[layer] : hot_count;
        const size_t capacity = layered_cache ? layer_hot_capacity[layer] : hot_capacity;
        if (current_hot < capacity) {
            promote(physical, key);
            return true;
        }

        float lowest_score = std::numeric_limits<float>::infinity();
        bool have_victim = false;
        const size_t begin = layered_cache && !prefill_full_layer_enabled ? layer_slot_begin[layer] : 0;
        const size_t end = layered_cache && !prefill_full_layer_enabled ? begin + layer_slot_count[layer] : slots.size();
        for (size_t i = begin; i < end; ++i) {
            const slot & candidate = slots[i];
            if (layered_cache && prefill_full_layer_enabled &&
                    (candidate.key < 0 || static_cast<size_t>(candidate.key) / n_expert != layer)) {
                continue;
            }
            if (i == physical || !candidate.hot || candidate.key < 0 ||
                    (!layered_cache && (candidate.refcnt != 0 || candidate.loading))) {
                continue;
            }
            lowest_score = std::min(lowest_score, decayed_request_score(static_cast<size_t>(candidate.key)));
            have_victim = true;
        }
        if (!have_victim || decayed_request_score(key) <= lowest_score) {
            return false;
        }
        promote(physical, key);
        return slots[physical].hot;
    }

    bool load_many(const uint32_t layer, const std::vector<pending_load> & loads) {
        if (loads.empty()) {
            return true;
        }
        std::vector<llama_edge_moe_io_request> requests;
        requests.reserve(loads.size() * part_count);
        for (size_t i = 0; i < part_count; ++i) {
            const source & item = layers[layer].tensors[i];
            const ggml_tensor * source_tensor = item.tensor;
            if (source_tensor == nullptr) {
                continue;
            }
            for (const pending_load & load : loads) {
                uint8_t * destination = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffers[i].get())) +
                    static_cast<size_t>(load.physical) * part_strides[i];
                size_t slice_offset = 0;
                if (!checked_mul(static_cast<size_t>(load.logical), source_tensor->nb[2], slice_offset)) {
                    return fail("edge MoE expert slice offset overflows size_t");
                }
                if (direct_io) {
                    if (slice_offset > item.size || source_tensor->nb[2] > item.size - slice_offset) {
                        return fail("edge MoE expert slice is outside its persisted source");
                    }
                    size_t file_offset = 0;
                    if (!checked_add(item.offset, slice_offset, file_offset)) {
                        return fail("edge MoE expert file offset overflows size_t");
                    }
                    requests.push_back({item.file_index, file_offset, source_tensor->nb[2], destination});
                } else {
                    if (source_tensor->data == nullptr) {
                        return fail("edge MoE arena cannot access expert tensor data");
                    }
                    const uint8_t * origin = static_cast<const uint8_t *>(source_tensor->data) + slice_offset;
                    std::memcpy(destination, origin, source_tensor->nb[2]);
                }
                bytes_read += source_tensor->nb[2];
                current_stats().bytes_read += source_tensor->nb[2];
            }
        }
        if (direct_io) {
            ++current_stats().read_calls;
            current_stats().experts_loaded += loads.size();
            std::string read_error;
            const llama_edge_moe_io_stats before = reader->stats();
            const int64_t begin = ggml_time_us();
            const bool ok = reader->read_many(requests, read_error);
            const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - begin);
            io_wait_us += elapsed;
            current_stats().io_wait_us += elapsed;
            if (!ok) {
                error = "unbuffered expert read failed: " + read_error;
                LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
                return false;
            }
            const llama_edge_moe_io_stats & after = reader->stats();
            current_stats().io_requests += after.requests - before.requests;
            current_stats().bytes_transferred += after.bytes_transferred - before.bytes_transferred;
        }
        return true;
    }

    bool start_full_layer(const uint32_t layer, const uint32_t buffer_index) {
        if (!reader || layer >= n_layer || buffer_index >= 2 || full_layer_future.valid()) {
            return fail("invalid edge MoE full-layer prefetch state");
        }

        const uint32_t first_slot = staging_begin + buffer_index * n_expert;
        if (static_cast<size_t>(first_slot) + n_expert > slots.size()) {
            return fail("edge MoE full-layer staging buffer is outside the arena");
        }
        for (uint32_t logical = 0; logical < n_expert; ++logical) {
            const uint32_t physical = first_slot + logical;
            if (slots[physical].refcnt != 0 || slots[physical].loading) {
                return fail("edge MoE full-layer staging buffer is still busy");
            }
            clear_slot(physical);
            slots[physical].loading = true;
            slots[physical].transient = true;
        }

        constexpr size_t request_target_bytes = 2 * 1024 * 1024;
        std::vector<llama_edge_moe_io_request> requests;
        size_t requested_bytes = 0;
        for (size_t i = 0; i < part_count; ++i) {
            const source & item = layers[layer].tensors[i];
            const ggml_tensor * source_tensor = item.tensor;
            if (source_tensor == nullptr) {
                continue;
            }
            const size_t expert_bytes = source_tensor->nb[2];
            const size_t experts_per_request = expert_bytes == part_strides[i] ?
                std::max<size_t>(1, request_target_bytes / expert_bytes) : 1;
            for (uint32_t logical = 0; logical < n_expert;) {
                const uint32_t expert_count = static_cast<uint32_t>(std::min<size_t>(
                    experts_per_request, static_cast<size_t>(n_expert - logical)));
                size_t slice_offset = 0;
                size_t request_bytes = 0;
                if (!checked_mul(static_cast<size_t>(logical), expert_bytes, slice_offset) ||
                    !checked_mul(static_cast<size_t>(expert_count), expert_bytes, request_bytes) ||
                    slice_offset > item.size || request_bytes > item.size - slice_offset) {
                    return fail("edge MoE full-layer read range overflows its persisted source");
                }
                size_t file_offset = 0;
                if (!checked_add(item.offset, slice_offset, file_offset)) {
                    return fail("edge MoE full-layer file offset overflows size_t");
                }
                uint8_t * destination = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffers[i].get())) +
                    static_cast<size_t>(first_slot + logical) * part_strides[i];
                requests.push_back({item.file_index, file_offset, request_bytes, destination});
                requested_bytes += request_bytes;
                logical += expert_count;
            }
        }

        bytes_read += requested_bytes;
        phase_counters & stats = current_stats();
        stats.bytes_read += requested_bytes;
        ++stats.read_calls;
        stats.experts_loaded += n_expert;
        pending_full_layer = static_cast<int32_t>(layer);
        pending_full_buffer = static_cast<int32_t>(buffer_index);
        llama_edge_moe_reader * const active_reader = reader.get();
        full_layer_future = std::async(std::launch::async,
                [active_reader, requests = std::move(requests)]() mutable {
                    full_layer_result result;
                    const llama_edge_moe_io_stats before = active_reader->stats();
                    result.ok = active_reader->read_many(requests, result.error);
                    const llama_edge_moe_io_stats & after = active_reader->stats();
                    result.io_requests = after.requests - before.requests;
                    result.bytes_transferred = after.bytes_transferred - before.bytes_transferred;
                    return result;
                });
        return true;
    }

    bool wait_full_layer(const uint32_t layer, const uint32_t buffer_index) {
        if (!full_layer_future.valid() || pending_full_layer != static_cast<int32_t>(layer) ||
                pending_full_buffer != static_cast<int32_t>(buffer_index)) {
            return fail("edge MoE full-layer prefetch completed out of order");
        }

        full_layer_result result;
        const int64_t begin = ggml_time_us();
        try {
            result = full_layer_future.get();
        } catch (const std::exception & exception) {
            result.error = exception.what();
        } catch (...) {
            result.error = "unknown asynchronous read failure";
        }
        const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - begin);
        io_wait_us += elapsed;
        phase_counters & stats = current_stats();
        stats.io_wait_us += elapsed;
        stats.io_requests += result.io_requests;
        stats.bytes_transferred += result.bytes_transferred;

        const uint32_t first_slot = staging_begin + buffer_index * n_expert;
        for (uint32_t logical = 0; logical < n_expert; ++logical) {
            slots[first_slot + logical].loading = false;
        }
        pending_full_layer = -1;
        pending_full_buffer = -1;
        if (!result.ok) {
            for (uint32_t logical = 0; logical < n_expert; ++logical) {
                clear_slot(first_slot + logical);
            }
            error = "unbuffered full-layer expert read failed: " + result.error;
            LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
            return false;
        }
        return true;
    }

    void drain_full_layer() {
        if (!full_layer_future.valid()) {
            return;
        }
        const uint32_t layer = static_cast<uint32_t>(pending_full_layer);
        const uint32_t buffer = static_cast<uint32_t>(pending_full_buffer);
        wait_full_layer(layer, buffer);
    }

    std::vector<uint32_t> observe_and_predict_route(
            const uint32_t layer,
            const ggml_tensor * ids_tensor,
            const int32_t * ids,
            const size_t count) {
        std::vector<uint32_t> predictions;
        if (!decode_prefetch_enabled || ids_tensor->ne[0] <= 0 || ids_tensor->ne[1] <= 0 ||
                static_cast<size_t>(ids_tensor->ne[0] * ids_tensor->ne[1]) != count) {
            return predictions;
        }
        const size_t width = static_cast<size_t>(ids_tensor->ne[0]);
        const size_t tokens = static_cast<size_t>(ids_tensor->ne[1]);
        const size_t pair_stride = static_cast<size_t>(n_expert) * n_expert;
        if (previous_route_layer + 1 == static_cast<int32_t>(layer) &&
                previous_route_width == width && previous_route_tokens == tokens) {
            uint16_t * transition = route_transitions.data() + static_cast<size_t>(layer - 1) * pair_stride;
            for (size_t token = 0; token < tokens; ++token) {
                for (size_t source_index = 0; source_index < width; ++source_index) {
                    const uint32_t source_expert = static_cast<uint32_t>(
                        previous_route_ids[token * width + source_index]);
                    for (size_t target_index = 0; target_index < width; ++target_index) {
                        const uint32_t target_expert = static_cast<uint32_t>(ids[token * width + target_index]);
                        uint16_t & value = transition[static_cast<size_t>(source_expert) * n_expert + target_expert];
                        if (value != std::numeric_limits<uint16_t>::max()) {
                            ++value;
                        }
                    }
                }
            }
        }

        if (!prefill && layer + 1 < n_layer) {
            const uint16_t * transition = route_transitions.data() + static_cast<size_t>(layer) * pair_stride;
            std::vector<uint64_t> scores(n_expert, 0);
            for (size_t i = 0; i < count; ++i) {
                const uint32_t source_expert = static_cast<uint32_t>(ids[i]);
                const uint16_t * row = transition + static_cast<size_t>(source_expert) * n_expert;
                for (uint32_t target_expert = 0; target_expert < n_expert; ++target_expert) {
                    scores[target_expert] += row[target_expert];
                }
            }
            for (uint32_t logical = 0; logical < n_expert; ++logical) {
                if (scores[logical] != 0) {
                    predictions.push_back(logical);
                }
            }
            std::sort(predictions.begin(), predictions.end(), [&](const uint32_t lhs, const uint32_t rhs) {
                return scores[lhs] != scores[rhs] ? scores[lhs] > scores[rhs] : lhs < rhs;
            });
        }

        previous_route_ids.assign(ids, ids + count);
        previous_route_layer = static_cast<int32_t>(layer);
        previous_route_width = width;
        previous_route_tokens = tokens;
        return predictions;
    }

    bool start_decode_prefetch(const uint32_t layer, const std::vector<uint32_t> & predictions) {
        if (!decode_prefetch_enabled) {
            return true;
        }
        if (layer >= n_layer || decode_prefetch_future.valid()) {
            return fail("invalid edge MoE decode prefetch state");
        }

        constexpr size_t max_prefetch = 1;
        std::vector<pending_load> loads;
        loads.reserve(std::min(max_prefetch, predictions.size()));
        phase_counters & stats = current_stats();
        for (const uint32_t logical : predictions) {
            if (loads.size() == max_prefetch) {
                break;
            }
            const size_t key = static_cast<size_t>(layer) * n_expert + logical;
            if (key_to_slot[key] >= 0) {
                break;
            }
            const int physical = find_layer_victim(layer, true);
            if (physical < 0) {
                break;
            }
            slot & victim = slots[static_cast<uint32_t>(physical)];
            if (victim.key >= 0) {
                clear_slot(static_cast<uint32_t>(physical));
                ++evictions;
                ++stats.evictions;
            }
            victim.key = static_cast<int64_t>(key);
            victim.loading = true;
            loads.push_back({logical, static_cast<uint32_t>(physical), key});
        }
        if (loads.empty()) {
            return true;
        }

        std::vector<llama_edge_moe_io_request> requests;
        requests.reserve(loads.size() * part_count);
        size_t requested_bytes = 0;
        for (size_t i = 0; i < part_count; ++i) {
            const source & item = layers[layer].tensors[i];
            const ggml_tensor * source_tensor = item.tensor;
            if (source_tensor == nullptr) {
                continue;
            }
            for (const pending_load & load : loads) {
                size_t slice_offset = 0;
                size_t file_offset = 0;
                if (!checked_mul(static_cast<size_t>(load.logical), source_tensor->nb[2], slice_offset) ||
                    slice_offset > item.size || source_tensor->nb[2] > item.size - slice_offset ||
                    !checked_add(item.offset, slice_offset, file_offset)) {
                    for (const pending_load & cleanup : loads) {
                        clear_slot(cleanup.physical);
                    }
                    return fail("edge MoE decode prefetch range overflows its persisted source");
                }
                uint8_t * destination = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffers[i].get())) +
                    static_cast<size_t>(load.physical) * part_strides[i];
                requests.push_back({item.file_index, file_offset, source_tensor->nb[2], destination});
                requested_bytes += source_tensor->nb[2];
            }
        }

        bytes_read += requested_bytes;
        stats.bytes_read += requested_bytes;
        ++stats.read_calls;
        stats.experts_loaded += loads.size();
        decode_prefetch_loaded += loads.size();
        stats.prefetch_loaded += loads.size();
        pending_decode_prefetch_layer = static_cast<int32_t>(layer);
        pending_decode_prefetch_loads = loads;
        llama_edge_moe_reader * const active_reader = reader.get();
        decode_prefetch_future = std::async(std::launch::async,
                [active_reader, requests = std::move(requests)]() mutable {
                    full_layer_result result;
                    const llama_edge_moe_io_stats before = active_reader->stats();
                    result.ok = active_reader->read_many(requests, result.error);
                    const llama_edge_moe_io_stats & after = active_reader->stats();
                    result.io_requests = after.requests - before.requests;
                    result.bytes_transferred = after.bytes_transferred - before.bytes_transferred;
                    return result;
                });
        return true;
    }

    bool wait_decode_prefetch(const uint32_t expected_layer) {
        if (!decode_prefetch_future.valid()) {
            return true;
        }
        if (pending_decode_prefetch_layer != static_cast<int32_t>(expected_layer)) {
            return fail("edge MoE decode prefetch completed out of order");
        }

        full_layer_result result;
        const int64_t begin = ggml_time_us();
        try {
            result = decode_prefetch_future.get();
        } catch (const std::exception & exception) {
            result.error = exception.what();
        } catch (...) {
            result.error = "unknown asynchronous read failure";
        }
        const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - begin);
        io_wait_us += elapsed;
        decode_prefetch_wait_us += elapsed;
        phase_counters & stats = current_stats();
        stats.io_wait_us += elapsed;
        stats.prefetch_wait_us += elapsed;
        stats.io_requests += result.io_requests;
        stats.bytes_transferred += result.bytes_transferred;

        if (!result.ok) {
            for (const pending_load & load : pending_decode_prefetch_loads) {
                slots[load.physical].loading = false;
                clear_slot(load.physical);
            }
            pending_decode_prefetch_loads.clear();
            pending_decode_prefetch_layer = -1;
            error = "unbuffered decode prefetch failed: " + result.error;
            LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
            return false;
        }
        for (const pending_load & load : pending_decode_prefetch_loads) {
            slot & selected = slots[load.physical];
            selected.loading = false;
            selected.prefetched = true;
            selected.last_used = ++clock;
            key_to_slot[load.key] = static_cast<int32_t>(load.physical);
        }
        pending_decode_prefetch_loads.clear();
        pending_decode_prefetch_layer = -1;
        return true;
    }

    void drain_decode_prefetch() {
        if (decode_prefetch_future.valid()) {
            wait_decode_prefetch(static_cast<uint32_t>(pending_decode_prefetch_layer));
        }
    }

    bool prepare_full_prefill(
            const uint32_t layer,
            int32_t * ids,
            const size_t count,
            const std::vector<uint8_t> & prefill_keep) {
        const uint32_t buffer_index = layer % 2;
        if (layer == 0 && !start_full_layer(layer, buffer_index)) {
            return false;
        }
        if (!wait_full_layer(layer, buffer_index)) {
            return false;
        }

        // Launch K+1 before retaining K's hot experts so the SSD read overlaps
        // both this copy and the remaining compute for layer K.
        if (layer + 1 < n_layer && !start_full_layer(layer + 1, (layer + 1) % 2)) {
            return false;
        }

        phase_counters & stats = current_stats();
        std::vector<uint8_t> seen(n_expert, 0);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t logical = static_cast<uint32_t>(ids[i]);
            if (seen[logical]) {
                continue;
            }
            seen[logical] = 1;
            const size_t key = static_cast<size_t>(layer) * n_expert + logical;
            if (key_to_slot[key] >= 0) {
                ++hits;
                ++stats.hits;
            } else {
                ++misses;
                ++stats.misses;
            }
        }

        const uint32_t hot_begin = layer_hot_begin[layer];
        for (uint32_t offset = 0; offset < layer_hot_capacity[layer]; ++offset) {
            const uint32_t physical = hot_begin + offset;
            if (slots[physical].refcnt != 0 || slots[physical].loading) {
                return fail("edge MoE full-layer hot region is still busy");
            }
            clear_slot(physical);
        }

        const uint32_t stage_begin = staging_begin + buffer_index * n_expert;
        uint32_t hot_offset = 0;
        for (uint32_t logical = 0; logical < n_expert; ++logical) {
            if (!prefill_keep[logical]) {
                continue;
            }
            if (hot_offset >= layer_hot_capacity[layer]) {
                return fail("edge MoE full-layer hot selection exceeds its layer budget");
            }
            const size_t key = static_cast<size_t>(layer) * n_expert + logical;
            const int32_t old_physical = key_to_slot[key];
            if (old_physical >= 0) {
                clear_slot(static_cast<uint32_t>(old_physical));
            }
            const uint32_t target = hot_begin + hot_offset++;
            for (size_t i = 0; i < part_count; ++i) {
                const ggml_tensor * source_tensor = layers[layer].tensors[i].tensor;
                if (source_tensor == nullptr) {
                    continue;
                }
                uint8_t * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffers[i].get()));
                std::memcpy(
                    base + static_cast<size_t>(target) * part_strides[i],
                    base + static_cast<size_t>(stage_begin + logical) * part_strides[i],
                    source_tensor->nb[2]);
            }
            slot & retained = slots[target];
            retained.key = static_cast<int64_t>(key);
            retained.last_used = ++clock;
            retained.prefill_admitted = true;
            key_to_slot[key] = static_cast<int32_t>(target);
            promote(target, key);
            ++prefill_admissions;
        }

        for (uint32_t logical = 0; logical < n_expert; ++logical) {
            if (!seen[logical]) {
                continue;
            }
            const uint32_t physical = stage_begin + logical;
            slot & selected = slots[physical];
            selected.last_used = ++clock;
            ++selected.refcnt;
            active_slots.push_back(physical);
        }
        for (size_t i = 0; i < count; ++i) {
            ids[i] = static_cast<int32_t>(stage_begin + static_cast<uint32_t>(ids[i]));
        }
        return true;
    }

    bool prepare(const uint32_t layer, ggml_tensor * ids_tensor) {
        release_active();
        if (layer >= n_layer || ids_tensor == nullptr || ids_tensor->type != GGML_TYPE_I32 || ids_tensor->data == nullptr) {
            return fail("invalid edge MoE arena ID tensor");
        }
        if (!prefill && decode_prefetch_enabled && !wait_decode_prefetch(layer)) {
            return false;
        }

        int32_t * ids = static_cast<int32_t *>(ids_tensor->data);
        const size_t count = static_cast<size_t>(ggml_nelements(ids_tensor));
        std::unordered_map<int32_t, uint32_t> remap;
        remap.reserve(std::min<size_t>(count, n_expert));

        std::vector<float> routed_scores(n_expert, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            const int32_t logical = ids[i];
            if (logical < 0 || static_cast<uint32_t>(logical) >= n_expert) {
                return fail("router returned an expert outside the model range");
            }
            routed_scores[static_cast<size_t>(logical)] += routed_weight(layer, i, count);
        }
        const std::vector<uint32_t> decode_predictions =
            observe_and_predict_route(layer, ids_tensor, ids, count);

        std::vector<uint8_t> prefill_keep;
        std::vector<uint8_t> prefill_hot;
        if (prefill) {
            prefill_keep.resize(n_expert, 0);
            prefill_hot.resize(n_expert, 0);
            for (size_t i = 0; i < count; ++i) {
                const int32_t logical = ids[i];
                if (logical < 0 || static_cast<uint32_t>(logical) >= n_expert) {
                    return fail("router returned an expert outside the model range");
                }
                uint32_t & frequency = prefill_frequency[static_cast<size_t>(layer) * n_expert + logical];
                if (frequency != std::numeric_limits<uint32_t>::max()) {
                    ++frequency;
                }
                prefill_score[static_cast<size_t>(layer) * n_expert + logical] += routed_weight(layer, i, count);
            }

            std::vector<uint32_t> candidates;
            candidates.reserve(n_expert);
            for (uint32_t logical = 0; logical < n_expert; ++logical) {
                if (prefill_frequency[static_cast<size_t>(layer) * n_expert + logical] != 0) {
                    candidates.push_back(logical);
                }
            }
            std::sort(candidates.begin(), candidates.end(), [&](const uint32_t lhs, const uint32_t rhs) {
                const float lhs_score = prefill_score[static_cast<size_t>(layer) * n_expert + lhs];
                const float rhs_score = prefill_score[static_cast<size_t>(layer) * n_expert + rhs];
                if (lhs_score != rhs_score) {
                    return lhs_score > rhs_score;
                }
                const uint32_t lhs_frequency = prefill_frequency[static_cast<size_t>(layer) * n_expert + lhs];
                const uint32_t rhs_frequency = prefill_frequency[static_cast<size_t>(layer) * n_expert + rhs];
                return lhs_frequency != rhs_frequency ? lhs_frequency > rhs_frequency : lhs < rhs;
            });
            const size_t resident_limit = prefill_resident_limit(layer);
            if (candidates.size() > resident_limit) {
                candidates.resize(resident_limit);
            }
            for (const uint32_t logical : candidates) {
                prefill_keep[logical] = 1;
                prefill_hot[logical] = 1;
                const size_t key = static_cast<size_t>(layer) * n_expert + logical;
                request_score[key] = prefill_score[key];
                request_score_epoch[key] = decode_epoch;
            }

            for (uint32_t physical = 0; physical < slots.size(); ++physical) {
                const slot & item = slots[physical];
                if (!item.prefill_admitted || item.refcnt != 0 || item.key < 0 ||
                        static_cast<uint32_t>(item.key / n_expert) != layer) {
                    continue;
                }
                const uint32_t logical = static_cast<uint32_t>(item.key % n_expert);
                if (!prefill_keep[logical]) {
                    clear_slot(physical);
                    ++prefill_drops;
                } else if (!prefill_hot[logical] && item.hot) {
                    demote_slot(physical);
                }
            }
        }

        if (prefill && prefill_full_layer_enabled && !full_prefill_decided) {
            full_prefill_decided = true;
            full_prefill_active = layer == 0 && ids_tensor->ne[1] >= 512;
            LLAMA_LOG_INFO("%s: full-layer prefill pipeline = %s, tokens = %lld\n",
                    __func__, full_prefill_active ? "active" : "inactive",
                    static_cast<long long>(ids_tensor->ne[1]));
        }
        if (prefill && full_prefill_active) {
            return prepare_full_prefill(layer, ids, count, prefill_keep);
        }

        phase_counters & stats = current_stats();
        std::vector<int32_t> logical_order;
        logical_order.reserve(std::min<size_t>(count, n_expert));
        std::vector<uint8_t> seen(n_expert, 0);
        for (size_t i = 0; i < count; ++i) {
            const int32_t logical = ids[i];
            if (!seen[static_cast<size_t>(logical)]) {
                seen[static_cast<size_t>(logical)] = 1;
                logical_order.push_back(logical);
            }
        }
        if (layered_cache && prefill) {
            std::stable_partition(logical_order.begin(), logical_order.end(), [&](const int32_t logical) {
                return prefill_keep[static_cast<size_t>(logical)] != 0;
            });
        }

        std::vector<pending_load> loads;
        loads.reserve(logical_order.size());

        for (const int32_t logical_i32 : logical_order) {
            if (logical_i32 < 0 || static_cast<uint32_t>(logical_i32) >= n_expert) {
                return fail("router returned an expert outside the model range");
            }
            const uint32_t logical = static_cast<uint32_t>(logical_i32);
            const size_t key = static_cast<size_t>(layer) * n_expert + logical;
            if (direct_io && !prefill) {
                add_request_score(key, routed_scores[logical]);
            }

            int32_t physical = key_to_slot[key];
            if (physical >= 0) {
                ++hits;
                ++stats.hits;
                if (slots[static_cast<uint32_t>(physical)].prefetched) {
                    slots[static_cast<uint32_t>(physical)].prefetched = false;
                    ++decode_prefetch_useful;
                    ++decode_stats.prefetch_useful;
                }
                if (direct_io && !prefill) {
                    uint8_t & frequency = decode_frequency[key];
                    if (frequency != std::numeric_limits<uint8_t>::max()) {
                        ++frequency;
                    }
                    if (!slots[static_cast<uint32_t>(physical)].hot && frequency >= 2) {
                        if (promote_decode(static_cast<uint32_t>(physical), key)) {
                            ++decode_admissions;
                        } else {
                            ++decode_admission_rejections;
                        }
                    }
                }
                if (prefill && prefill_keep[logical]) {
                    slots[static_cast<uint32_t>(physical)].prefill_admitted = true;
                }
                if (!direct_io || (prefill && prefill_hot[logical])) {
                    promote(static_cast<uint32_t>(physical), key);
                }
            } else {
                ++misses;
                ++stats.misses;
                bool admit_decode = false;
                if (direct_io && !prefill) {
                    uint8_t & frequency = decode_frequency[key];
                    if (frequency != std::numeric_limits<uint8_t>::max()) {
                        ++frequency;
                    }
                    admit_decode = frequency >= 2;
                }
                if (layered_cache && prefill) {
                    physical = prefill_keep[logical] && prefill_full_layer_enabled ?
                        find_layer_hot_victim(layer) :
                        prefill_keep[logical] ? find_layer_victim(layer, false) : find_transient_victim();
                } else if (layered_cache) {
                    physical = find_layer_victim(layer, true);
                } else {
                    physical = find_victim(direct_io && !prefill);
                }
                if (physical < 0 && direct_io && !prefill) {
                    // Correctness fallback for batches whose live cold working set exceeds the
                    // reserved scratch area. Normal decode should remain entirely in scratch.
                    physical = layered_cache ? find_layer_victim(layer, false) : find_victim(false);
                    if (physical >= 0) {
                        ++decode_resident_fallbacks;
                    }
                }
                if (physical < 0) {
                    size_t empty = 0;
                    size_t cold = 0;
                    size_t hot = 0;
                    size_t busy = 0;
                    size_t layer_empty = 0;
                    size_t layer_cold = 0;
                    size_t layer_hot = 0;
                    size_t layer_busy = 0;
                    for (size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
                        const slot & item = slots[slot_index];
                        size_t * category = item.refcnt != 0 || item.loading ? &busy :
                            item.key < 0 ? &empty : item.hot ? &hot : &cold;
                        ++*category;
                        if (!layered_cache || slot_belongs_to_layer(slot_index, layer)) {
                            size_t * layer_category = item.refcnt != 0 || item.loading ? &layer_busy :
                                item.key < 0 ? &layer_empty : item.hot ? &layer_hot : &layer_cold;
                            ++*layer_category;
                        }
                    }
                    LLAMA_LOG_ERROR("%s: no slot for layer=%u expert=%u prefill=%d keep=%d; global empty=%zu cold=%zu hot=%zu busy=%zu; layer empty=%zu cold=%zu hot=%zu busy=%zu\n",
                            __func__, layer, logical, prefill ? 1 : 0,
                            prefill && prefill_keep[logical] ? 1 : 0,
                            empty, cold, hot, busy, layer_empty, layer_cold, layer_hot, layer_busy);
                    return fail("edge MoE arena has no evictable slot");
                }

                slot & victim = slots[physical];
                if (victim.key >= 0) {
                    clear_slot(static_cast<uint32_t>(physical));
                    ++evictions;
                    ++stats.evictions;
                }
                victim.key = static_cast<int64_t>(key);
                victim.loading = true;
                victim.transient = prefill && !prefill_keep[logical];
                if (prefill && prefill_keep[logical]) {
                    victim.prefill_admitted = true;
                    if (prefill_hot[logical]) {
                        promote(static_cast<uint32_t>(physical), key);
                    }
                    ++prefill_admissions;
                } else if (admit_decode) {
                    if (promote_decode(static_cast<uint32_t>(physical), key)) {
                        ++decode_admissions;
                    } else {
                        ++decode_admission_rejections;
                    }
                }
                loads.push_back({logical, static_cast<uint32_t>(physical), key});
            }

            slot & selected = slots[physical];
            selected.last_used = ++clock;
            if (!selected.loading) {
                ++selected.refcnt;
                active_slots.push_back(static_cast<uint32_t>(physical));
            }
            remap.emplace(logical_i32, static_cast<uint32_t>(physical));
        }

        if (!load_many(layer, loads)) {
            for (const pending_load & load : loads) {
                clear_slot(load.physical);
            }
            release_active();
            return false;
        }

        for (const pending_load & load : loads) {
            slot & selected = slots[load.physical];
            selected.loading = false;
            selected.key = static_cast<int64_t>(load.key);
            key_to_slot[load.key] = static_cast<int32_t>(load.physical);
            ++selected.refcnt;
            active_slots.push_back(load.physical);
        }

        for (size_t i = 0; i < count; ++i) {
            ids[i] = static_cast<int32_t>(remap.at(ids[i]));
        }
        if (!prefill && decode_prefetch_enabled && layer + 1 < n_layer &&
                !start_decode_prefetch(layer + 1, decode_predictions)) {
            return false;
        }
        return true;
    }
};

llama_edge_moe_arena::llama_edge_moe_arena(
        const llama_model & model,
        ggml_backend * backend_cpu,
        const size_t budget_bytes,
        const bool direct_io,
        const size_t io_depth,
        const bool layered_cache,
        const bool prefill_full_layer,
        const bool decode_prefetch,
        const uint32_t hot_slots_per_layer,
        const std::vector<uint32_t> & hot_slots_by_layer) :
    pimpl(std::make_unique<impl>(model, backend_cpu, budget_bytes, direct_io, io_depth,
            layered_cache, prefill_full_layer, decode_prefetch, hot_slots_per_layer,
            hot_slots_by_layer)) {}

llama_edge_moe_arena::~llama_edge_moe_arena() = default;

ggml_tensor * llama_edge_moe_arena::weight_for(const ggml_tensor * source) const {
    if (source == nullptr) {
        return nullptr;
    }
    const auto it = pimpl->replacements.find(source);
    return it == pimpl->replacements.end() ? nullptr : it->second;
}

void llama_edge_moe_arena::set_router_weights(const uint32_t layer, ggml_tensor * weights) {
    pimpl->set_router_weights(layer, weights);
}

bool llama_edge_moe_arena::callback(ggml_tensor * tensor, const bool ask) {
    uint32_t layer = 0;
    if (!parse_layer(tensor ? tensor->name : nullptr, layer)) {
        return false;
    }
    if (ask) {
        return true;
    }
    return pimpl->prepare(layer, tensor);
}

void llama_edge_moe_arena::begin_batch(const bool prefill) {
    pimpl->begin_batch(prefill);
}

void llama_edge_moe_arena::end_batch() {
    pimpl->end_batch();
}

void llama_edge_moe_arena::begin_compute() {
    pimpl->release_active();
    pimpl->error.clear();
}

bool llama_edge_moe_arena::failed() const {
    return !pimpl->error.empty();
}

size_t llama_edge_moe_arena::size_bytes() const {
    return pimpl->allocated_bytes;
}
