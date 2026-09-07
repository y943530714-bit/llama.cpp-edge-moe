#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct ggml_tensor;

// Captures the tensors that describe MoE routing during graph execution.
// The callback is intentionally independent of a compute backend.
struct edge_moe_trace_cb_user_data {
    edge_moe_trace_cb_user_data(const std::string & path, uint64_t max_events);
    ~edge_moe_trace_cb_user_data();

    edge_moe_trace_cb_user_data(const edge_moe_trace_cb_user_data &) = delete;
    edge_moe_trace_cb_user_data & operator=(const edge_moe_trace_cb_user_data &) = delete;

    bool ok() const;
    uint64_t events_written() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;

    friend bool edge_moe_trace_cb_eval(ggml_tensor * tensor, bool ask, void * user_data);
};

// Intended for use as llama_context_params::cb_eval.
bool edge_moe_trace_cb_eval(ggml_tensor * tensor, bool ask, void * user_data);
