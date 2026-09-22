#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_edge_moe_io_request {
    uint32_t file_index = 0;
    size_t offset = 0;
    size_t size = 0;
    void * destination = nullptr;
};

struct llama_edge_moe_io_stats {
    uint64_t requests = 0;
    uint64_t bytes_requested = 0;
    uint64_t bytes_transferred = 0;
    size_t peak_bounce_bytes = 0;
};

struct llama_edge_moe_process_memory {
    size_t working_set_bytes = 0;
    size_t peak_working_set_bytes = 0;
    size_t private_bytes = 0;
};

class LLAMA_API llama_edge_moe_reader {
public:
    explicit llama_edge_moe_reader(const std::vector<std::string> & paths, size_t max_in_flight = 8);
    ~llama_edge_moe_reader();

    llama_edge_moe_reader(const llama_edge_moe_reader &) = delete;
    llama_edge_moe_reader & operator=(const llama_edge_moe_reader &) = delete;

    bool read(const llama_edge_moe_io_request & request, std::string & error);
    bool read_many(const std::vector<llama_edge_moe_io_request> & requests, std::string & error);

    size_t file_count() const;
    size_t file_alignment(uint32_t file_index) const;
    const llama_edge_moe_io_stats & stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

class llama_edge_moe_resident_set {
public:
    llama_edge_moe_resident_set();
    ~llama_edge_moe_resident_set();

    llama_edge_moe_resident_set(const llama_edge_moe_resident_set &) = delete;
    llama_edge_moe_resident_set & operator=(const llama_edge_moe_resident_set &) = delete;

    bool lock(
            const std::vector<std::pair<void *, size_t>> & ranges,
            size_t process_budget_bytes,
            std::string & error);

    size_t size_bytes() const;
    size_t range_count() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

LLAMA_API bool llama_edge_moe_get_process_memory(llama_edge_moe_process_memory & memory, std::string & error);
