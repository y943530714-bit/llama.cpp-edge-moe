#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class edge_moe_slot_state : uint8_t {
    free,
    loading,
    resident,
    in_use,
    evict_pending,
};

struct edge_moe_slot_meta {
    edge_moe_slot_state state = edge_moe_slot_state::free;
    int32_t logical_expert = -1;
    uint64_t last_used_token = 0;
    float ema_score = 0.0f;
    float admission_score = 0.0f;
    uint32_t refcnt = 0;
    uint32_t generation = 0;
};

class edge_moe_resident_slots {
public:
    edge_moe_resident_slots(uint32_t logical_expert_count, uint32_t slot_count, size_t slot_bytes);

    uint32_t logical_expert_count() const;
    uint32_t slot_count() const;
    size_t slot_bytes() const;

    bool load_blocking(uint32_t logical_expert, const void * data, size_t size, uint64_t token, std::string & error);
    bool resolve(uint32_t logical_expert, uint32_t & physical_slot) const;
    bool begin_use(uint32_t logical_expert, uint64_t token, uint32_t & physical_slot, std::string & error);
    bool end_use(uint32_t logical_expert, std::string & error);
    bool evict(uint32_t logical_expert, std::string & error);

    const edge_moe_slot_meta * slot_meta(uint32_t physical_slot) const;
    const uint8_t * slot_data(uint32_t physical_slot) const;

private:
    std::vector<int32_t> logical_to_slot_;
    std::vector<edge_moe_slot_meta> slots_;
    std::vector<uint8_t> data_;
    size_t slot_bytes_ = 0;

    int find_free_slot() const;
    bool valid_logical(uint32_t logical_expert) const;
    bool valid_slot(uint32_t physical_slot) const;
};
