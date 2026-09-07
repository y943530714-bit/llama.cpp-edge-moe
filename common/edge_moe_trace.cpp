#include "edge_moe_trace.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "log.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string_view>
#include <vector>

namespace {

enum class trace_tensor_kind {
    none,
    topk,
    weights_raw,
    weights_normalized,
};

struct tensor_match {
    trace_tensor_kind kind = trace_tensor_kind::none;
    int layer = -1;
};

bool parse_layered_name(const char * name, std::string_view prefix, int & layer) {
    if (name == nullptr) {
        return false;
    }

    const std::string_view value(name);
    if (value.size() <= prefix.size() || value.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }

    const std::string_view suffix = value.substr(prefix.size());
    int parsed = 0;
    for (const char ch : suffix) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        parsed = parsed * 10 + (ch - '0');
    }

    layer = parsed;
    return true;
}

tensor_match match_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return {};
    }

    int layer = -1;
    if (parse_layered_name(tensor->name, "ffn_moe_topk-", layer)) {
        return { trace_tensor_kind::topk, layer };
    }
    if (parse_layered_name(tensor->name, "ffn_moe_weights_norm-", layer)) {
        return { trace_tensor_kind::weights_normalized, layer };
    }
    if (parse_layered_name(tensor->name, "ffn_moe_weights_softmax-", layer)) {
        return { trace_tensor_kind::weights_normalized, layer };
    }
    if (parse_layered_name(tensor->name, "ffn_moe_weights-", layer)) {
        return { trace_tensor_kind::weights_raw, layer };
    }

    return {};
}

bool copy_tensor_data(const ggml_tensor * tensor, std::vector<uint8_t> & data) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return false;
    }

    const size_t size = ggml_nbytes(tensor);
    if (size == 0) {
        return false;
    }

    data.resize(size);
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        if (tensor->data == nullptr) {
            return false;
        }
        std::memcpy(data.data(), tensor->data, size);
    } else {
        ggml_backend_tensor_get(tensor, data.data(), 0, size);
    }

    return true;
}

float read_float(const uint8_t * data, ggml_type type, size_t offset) {
    switch (type) {
        case GGML_TYPE_F64: {
            double value;
            std::memcpy(&value, data + offset, sizeof(value));
            return static_cast<float>(value);
        }
        case GGML_TYPE_F32: {
            float value;
            std::memcpy(&value, data + offset, sizeof(value));
            return value;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return ggml_fp16_to_fp32(value);
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return ggml_bf16_to_fp32(value);
        }
        case GGML_TYPE_I64: {
            int64_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return static_cast<float>(value);
        }
        case GGML_TYPE_I32: {
            int32_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return static_cast<float>(value);
        }
        case GGML_TYPE_I16: {
            int16_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return static_cast<float>(value);
        }
        case GGML_TYPE_I8: {
            int8_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return static_cast<float>(value);
        }
        default:
            return NAN;
    }
}

int32_t read_index(const uint8_t * data, ggml_type type, size_t offset) {
    switch (type) {
        case GGML_TYPE_I64: {
            int64_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return static_cast<int32_t>(value);
        }
        case GGML_TYPE_I32: {
            int32_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return value;
        }
        case GGML_TYPE_I16: {
            int16_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return static_cast<int32_t>(value);
        }
        case GGML_TYPE_I8: {
            int8_t value;
            std::memcpy(&value, data + offset, sizeof(value));
            return static_cast<int32_t>(value);
        }
        default:
            return -1;
    }
}

bool extract_topk(const ggml_tensor * tensor, std::vector<int32_t> & ids, size_t & n_tokens, size_t & n_experts_used) {
    if (tensor == nullptr || tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
        return false;
    }

    std::vector<uint8_t> data;
    if (!copy_tensor_data(tensor, data)) {
        return false;
    }

    n_experts_used = static_cast<size_t>(tensor->ne[0]);
    n_tokens       = static_cast<size_t>(tensor->ne[1]);
    ids.resize(n_tokens * n_experts_used);

    for (size_t token = 0; token < n_tokens; ++token) {
        for (size_t expert = 0; expert < n_experts_used; ++expert) {
            const size_t offset = token * tensor->nb[1] + expert * tensor->nb[0];
            ids[token * n_experts_used + expert] = read_index(data.data(), tensor->type, offset);
        }
    }

    return true;
}

bool extract_weights(
        const ggml_tensor * tensor,
        size_t n_tokens,
        size_t n_experts_used,
        std::vector<float> & weights) {
    if (tensor == nullptr || n_tokens == 0 || n_experts_used == 0) {
        return false;
    }

    std::vector<uint8_t> data;
    if (!copy_tensor_data(tensor, data)) {
        return false;
    }

    size_t tensor_tokens = 0;
    bool packed_first = false;
    if (tensor->ne[0] == 1 && tensor->ne[1] >= static_cast<int64_t>(n_experts_used)) {
        tensor_tokens = static_cast<size_t>(tensor->ne[2]);
        packed_first = true;
    } else if (tensor->ne[0] >= static_cast<int64_t>(n_experts_used) && tensor->ne[1] > 0) {
        tensor_tokens = static_cast<size_t>(tensor->ne[1]);
    } else {
        return false;
    }

    if (tensor_tokens < n_tokens) {
        return false;
    }

    weights.resize(n_tokens * n_experts_used);
    for (size_t token = 0; token < n_tokens; ++token) {
        for (size_t expert = 0; expert < n_experts_used; ++expert) {
            const size_t offset = packed_first
                ? expert * tensor->nb[1] + token * tensor->nb[2]
                : expert * tensor->nb[0] + token * tensor->nb[1];
            weights[token * n_experts_used + expert] = read_float(data.data(), tensor->type, offset);
        }
    }

    return true;
}

struct pending_route {
    std::vector<int32_t> ids;
    std::vector<float> weights_raw;
    std::vector<float> weights_normalized;
    size_t n_tokens = 0;
    size_t n_experts_used = 0;
    bool have_ids = false;
    bool have_raw = false;
    bool have_normalized = false;
    bool emitted = false;
};

} // namespace

struct edge_moe_trace_cb_user_data::impl {
    explicit impl(const std::string & path, uint64_t max_events) : max_events(max_events) {
        stream.open(path, std::ios::out | std::ios::trunc);
        if (!stream) {
            LOG_ERR("edge MoE trace: failed to open '%s'\n", path.c_str());
            return;
        }

        stream << "{\"schema_version\":1,\"event\":\"header\",\"format\":\"edge-moe-route-v1\"}\n";
        stream.flush();
        valid = true;
        LOG_INF("edge MoE trace: writing route events to '%s'\n", path.c_str());
    }

    ~impl() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & entry : pending) {
            emit(entry.first, entry.second, false);
        }
        stream.flush();
    }

    bool interested(const ggml_tensor * tensor) const {
        if (match_tensor(tensor).kind == trace_tensor_kind::none) {
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (!valid || (max_events != 0 && events_written >= max_events)) {
            return false;
        }

        return true;
    }

    void process(ggml_tensor * tensor) {
        const tensor_match match = match_tensor(tensor);
        if (match.kind == trace_tensor_kind::none) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (!valid || (max_events != 0 && events_written >= max_events)) {
            return;
        }

        pending_route & route = pending[match.layer];
        if (match.kind == trace_tensor_kind::topk) {
            if (route.emitted) {
                route = {};
            } else if (route.have_ids && route.have_raw) {
                emit(match.layer, route, false);
                route = {};
            }

            size_t n_tokens = 0;
            size_t n_experts_used = 0;
            if (extract_topk(tensor, route.ids, n_tokens, n_experts_used)) {
                route.n_tokens = n_tokens;
                route.n_experts_used = n_experts_used;
                route.have_ids = true;
                maybe_emit(match.layer, route);
            }
            return;
        }

        if (route.emitted || !route.have_ids) {
            return;
        }

        std::vector<float> weights;
        if (!extract_weights(tensor, route.n_tokens, route.n_experts_used, weights)) {
            return;
        }

        if (match.kind == trace_tensor_kind::weights_normalized) {
            route.weights_normalized = std::move(weights);
            route.have_normalized = true;
        } else {
            route.weights_raw = std::move(weights);
            route.have_raw = true;
        }
        maybe_emit(match.layer, route);
    }

    void maybe_emit(int layer, pending_route & route) {
        if (route.have_ids && route.have_normalized) {
            emit(layer, route, true);
        }
    }

    void emit(int layer, pending_route & route, bool normalized) {
        const std::vector<float> & weights = normalized ? route.weights_normalized : route.weights_raw;
        const size_t expected = route.n_tokens * route.n_experts_used;
        if (!route.have_ids || weights.size() != expected || route.n_tokens == 0 || route.n_experts_used == 0) {
            return;
        }

        stream << "{\"schema_version\":1,\"event\":\"route\",\"event_id\":" << events_written
               << ",\"layer\":" << layer
               << ",\"stage\":\"" << (route.n_tokens == 1 ? "decode" : "prefill") << "\""
               << ",\"n_tokens\":" << route.n_tokens
               << ",\"n_experts_used\":" << route.n_experts_used
               << ",\"weights_normalized\":" << (normalized ? "true" : "false")
               << ",\"expert_ids\":[";

        for (size_t token = 0; token < route.n_tokens; ++token) {
            if (token != 0) {
                stream << ',';
            }
            stream << '[';
            for (size_t expert = 0; expert < route.n_experts_used; ++expert) {
                if (expert != 0) {
                    stream << ',';
                }
                stream << route.ids[token * route.n_experts_used + expert];
            }
            stream << ']';
        }

        stream << "],\"weights\":[" << std::setprecision(9);
        for (size_t token = 0; token < route.n_tokens; ++token) {
            if (token != 0) {
                stream << ',';
            }
            stream << '[';
            for (size_t expert = 0; expert < route.n_experts_used; ++expert) {
                if (expert != 0) {
                    stream << ',';
                }
                const float value = weights[token * route.n_experts_used + expert];
                stream << (std::isfinite(value) ? value : 0.0f);
            }
            stream << ']';
        }
        stream << "]}\n";
        stream.flush();

        ++events_written;
        route.emitted = true;
    }

    std::ofstream stream;
    std::map<int, pending_route> pending;
    mutable std::mutex mutex;
    uint64_t max_events = 0;
    uint64_t events_written = 0;
    bool valid = false;
};

edge_moe_trace_cb_user_data::edge_moe_trace_cb_user_data(const std::string & path, uint64_t max_events) :
    pimpl(std::make_unique<impl>(path, max_events)) {}

edge_moe_trace_cb_user_data::~edge_moe_trace_cb_user_data() = default;

bool edge_moe_trace_cb_user_data::ok() const {
    return pimpl->valid;
}

uint64_t edge_moe_trace_cb_user_data::events_written() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->events_written;
}

bool edge_moe_trace_cb_eval(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * trace = static_cast<edge_moe_trace_cb_user_data *>(user_data);
    if (trace == nullptr || trace->pimpl == nullptr) {
        return false;
    }

    if (ask) {
        return trace->pimpl->interested(tensor);
    }

    if (trace->pimpl->interested(tensor)) {
        trace->pimpl->process(tensor);
        return true;
    }

    return false;
}
