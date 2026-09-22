#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_backend;
struct ggml_tensor;
struct llama_model;

class llama_edge_moe_arena {
public:
    llama_edge_moe_arena(
            const llama_model & model,
            ggml_backend * backend_cpu,
            size_t budget_bytes,
            bool direct_io = false,
            size_t io_depth = 8,
            bool layered_cache = false,
            bool prefill_full_layer = false,
            bool decode_prefetch = false,
            uint32_t hot_slots_per_layer = 0,
            const std::vector<uint32_t> & hot_slots_by_layer = {});
    ~llama_edge_moe_arena();

    llama_edge_moe_arena(const llama_edge_moe_arena &) = delete;
    llama_edge_moe_arena & operator=(const llama_edge_moe_arena &) = delete;

    ggml_tensor * weight_for(const ggml_tensor * source) const;
    void set_router_weights(uint32_t layer, ggml_tensor * weights);

    bool callback(ggml_tensor * tensor, bool ask);
    void begin_batch(bool prefill);
    void end_batch();
    void begin_compute();
    bool failed() const;
    size_t size_bytes() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
