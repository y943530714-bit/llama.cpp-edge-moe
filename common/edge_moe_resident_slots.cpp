#include "edge_moe_resident_slots.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

uint32_t next_generation(const uint32_t generation) {
    return generation == std::numeric_limits<uint32_t>::max() ? 1 : generation + 1;
}

} // namespace

edge_moe_resident_slots::edge_moe_resident_slots(
        const uint32_t logical_expert_count,
        const uint32_t slot_count,
        const size_t slot_bytes) :
    logical_to_slot_(logical_expert_count, -1),
    slots_(slot_count),
    slot_bytes_(slot_bytes) {
    if (logical_expert_count == 0 || slot_count == 0 || slot_bytes == 0) {
        throw std::invalid_argument("resident slot dimensions must be non-zero");
    }
    if (slot_bytes > std::numeric_limits<size_t>::max() / slot_count) {
        throw std::length_error("resident slot arena size overflows size_t");
    }
    data_.resize(static_cast<size_t>(slot_count) * slot_bytes);
}

uint32_t edge_moe_resident_slots::logical_expert_count() const {
    return static_cast<uint32_t>(logical_to_slot_.size());
}

uint32_t edge_moe_resident_slots::slot_count() const {
    return static_cast<uint32_t>(slots_.size());
}

size_t edge_moe_resident_slots::slot_bytes() const {
    return slot_bytes_;
}

bool edge_moe_resident_slots::valid_logical(const uint32_t logical_expert) const {
    return logical_expert < logical_to_slot_.size();
}

bool edge_moe_resident_slots::valid_slot(const uint32_t physical_slot) const {
    return physical_slot < slots_.size();
}

int edge_moe_resident_slots::find_free_slot() const {
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].state == edge_moe_slot_state::free) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool edge_moe_resident_slots::load_blocking(
        const uint32_t logical_expert,
        const void * data,
        const size_t size,
        const uint64_t token,
        std::string & error) {
    error.clear();
    if (!valid_logical(logical_expert)) {
        error = "logical expert is outside the arena";
        return false;
    }
    if (data == nullptr || size != slot_bytes_) {
        error = "blocking load size does not match the slot size";
        return false;
    }

    int physical_slot = logical_to_slot_[logical_expert];
    if (physical_slot >= 0) {
        if (!valid_slot(static_cast<uint32_t>(physical_slot))) {
            error = "logical-to-physical mapping points outside the arena";
            return false;
        }

        edge_moe_slot_meta & meta = slots_[physical_slot];
        if (meta.logical_expert != static_cast<int32_t>(logical_expert)) {
            error = "logical-to-physical mapping metadata is inconsistent";
            return false;
        }
        if (meta.refcnt != 0 || meta.state == edge_moe_slot_state::in_use) {
            error = "cannot reload an expert that is in use";
            return false;
        }
        if (meta.state != edge_moe_slot_state::resident) {
            error = "resident slot is not ready for blocking reload";
            return false;
        }
    } else {
        physical_slot = find_free_slot();
        if (physical_slot < 0) {
            error = "no free resident slot; evict explicitly before loading";
            return false;
        }

        edge_moe_slot_meta & meta = slots_[physical_slot];
        const uint32_t generation = meta.generation;
        meta = {};
        meta.generation = generation;
        meta.state = edge_moe_slot_state::loading;
        meta.logical_expert = static_cast<int32_t>(logical_expert);
        logical_to_slot_[logical_expert] = physical_slot;
    }

    edge_moe_slot_meta & meta = slots_[physical_slot];
    const uint32_t generation = next_generation(meta.generation);
    meta.state = edge_moe_slot_state::loading;

    // memmove is intentional: callers may reload from a view into this slot.
    std::memmove(
        data_.data() + static_cast<size_t>(physical_slot) * slot_bytes_,
        data,
        slot_bytes_);

    meta.state = edge_moe_slot_state::resident;
    meta.last_used_token = token;
    meta.generation = generation;
    return true;
}

bool edge_moe_resident_slots::resolve(const uint32_t logical_expert, uint32_t & physical_slot) const {
    if (!valid_logical(logical_expert) || logical_to_slot_[logical_expert] < 0) {
        return false;
    }
    physical_slot = static_cast<uint32_t>(logical_to_slot_[logical_expert]);
    return true;
}

bool edge_moe_resident_slots::begin_use(
        const uint32_t logical_expert,
        const uint64_t token,
        uint32_t & physical_slot,
        std::string & error) {
    error.clear();
    if (!resolve(logical_expert, physical_slot)) {
        error = "logical expert is not resident";
        return false;
    }

    edge_moe_slot_meta & meta = slots_[physical_slot];
    if (meta.state != edge_moe_slot_state::resident && meta.state != edge_moe_slot_state::in_use) {
        error = "resident slot is not ready for use";
        return false;
    }
    meta.state = edge_moe_slot_state::in_use;
    meta.last_used_token = token;
    ++meta.refcnt;
    return true;
}

bool edge_moe_resident_slots::end_use(const uint32_t logical_expert, std::string & error) {
    error.clear();
    uint32_t physical_slot = 0;
    if (!resolve(logical_expert, physical_slot)) {
        error = "logical expert is not resident";
        return false;
    }

    edge_moe_slot_meta & meta = slots_[physical_slot];
    if (meta.refcnt == 0) {
        error = "resident slot has no active reference";
        return false;
    }
    --meta.refcnt;
    if (meta.refcnt == 0) {
        meta.state = edge_moe_slot_state::resident;
    }
    return true;
}

bool edge_moe_resident_slots::evict(const uint32_t logical_expert, std::string & error) {
    error.clear();
    uint32_t physical_slot = 0;
    if (!resolve(logical_expert, physical_slot)) {
        error = "logical expert is not resident";
        return false;
    }

    edge_moe_slot_meta & meta = slots_[physical_slot];
    if (meta.state != edge_moe_slot_state::resident || meta.refcnt != 0) {
        error = "cannot evict an active or non-resident slot";
        return false;
    }

    const uint32_t generation = meta.generation;
    logical_to_slot_[logical_expert] = -1;
    meta = {};
    // Keep the generation monotonic across free/rebind cycles.
    meta.generation = generation;
    return true;
}

const edge_moe_slot_meta * edge_moe_resident_slots::slot_meta(const uint32_t physical_slot) const {
    return valid_slot(physical_slot) ? &slots_[physical_slot] : nullptr;
}

const uint8_t * edge_moe_resident_slots::slot_data(const uint32_t physical_slot) const {
    if (!valid_slot(physical_slot)) {
        return nullptr;
    }
    return data_.data() + static_cast<size_t>(physical_slot) * slot_bytes_;
}
