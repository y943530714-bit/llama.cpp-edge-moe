#include "llama-edge-moe-io.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {

#if defined(_WIN32)
std::string win_error(const DWORD code) {
    LPSTR message = nullptr;
    const DWORD size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&message),
        0,
        nullptr);
    if (size == 0) {
        return "Win32 error " + std::to_string(code);
    }
    std::string result(message, size);
    LocalFree(message);
    while (!result.empty() && (result.back() == '\r' || result.back() == '\n')) {
        result.pop_back();
    }
    return result;
}

bool is_power_of_two(const size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool align_down(const size_t value, const size_t alignment, size_t & result) {
    if (!is_power_of_two(alignment)) {
        return false;
    }
    result = value & ~(alignment - 1);
    return true;
}

bool align_up(const size_t value, const size_t alignment, size_t & result) {
    if (!is_power_of_two(alignment) || value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return false;
    }
    result = (value + alignment - 1) & ~(alignment - 1);
    return true;
}
#endif

} // namespace

struct llama_edge_moe_reader::impl {
#if defined(_WIN32)
    struct file {
        HANDLE handle = INVALID_HANDLE_VALUE;
        size_t size = 0;
        size_t alignment = 0;
    };

    struct pending_read {
        const llama_edge_moe_io_request * request = nullptr;
        file * source = nullptr;
        OVERLAPPED overlapped = {};
        HANDLE event = nullptr;
        void * bounce = nullptr;
        size_t aligned_offset = 0;
        size_t aligned_size = 0;
        size_t prefix = 0;
        bool submitted = false;

        pending_read() = default;
        pending_read(const pending_read &) = delete;
        pending_read & operator=(const pending_read &) = delete;

        pending_read(pending_read && other) noexcept {
            *this = std::move(other);
        }

        pending_read & operator=(pending_read && other) noexcept {
            if (this != &other) {
                if (event != nullptr) {
                    CloseHandle(event);
                }
                if (bounce != nullptr) {
                    VirtualFree(bounce, 0, MEM_RELEASE);
                }
                request = other.request;
                source = other.source;
                overlapped = other.overlapped;
                event = other.event;
                bounce = other.bounce;
                aligned_offset = other.aligned_offset;
                aligned_size = other.aligned_size;
                prefix = other.prefix;
                submitted = other.submitted;
                other.event = nullptr;
                other.bounce = nullptr;
                other.submitted = false;
            }
            return *this;
        }

        ~pending_read() {
            if (event != nullptr) {
                CloseHandle(event);
            }
            if (bounce != nullptr) {
                VirtualFree(bounce, 0, MEM_RELEASE);
            }
        }
    };

    std::vector<file> files;
#endif
    llama_edge_moe_io_stats io_stats;
    size_t max_in_flight = 0;

    impl(const std::vector<std::string> & paths, const size_t max_in_flight) : max_in_flight(max_in_flight) {
        if (max_in_flight == 0) {
            throw std::invalid_argument("edge MoE reader queue depth must be non-zero");
        }
#if defined(_WIN32)
        if (paths.empty()) {
            throw std::invalid_argument("edge MoE reader requires at least one file");
        }
        files.reserve(paths.size());
        try {
            for (const std::string & path_utf8 : paths) {
                if (path_utf8.empty()) {
                    throw std::invalid_argument("edge MoE reader cannot open an unnamed file source");
                }

                const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8.c_str(), -1, nullptr, 0);
                if (wide_size == 0) {
                    throw std::runtime_error("failed to convert model path to UTF-16: " + win_error(GetLastError()));
                }
                std::wstring path(static_cast<size_t>(wide_size), L'\0');
                if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8.c_str(), -1, path.data(), wide_size) == 0) {
                    throw std::runtime_error("failed to convert model path to UTF-16: " + win_error(GetLastError()));
                }

                file item;
                item.handle = CreateFileW(
                    path.c_str(),
                    GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr,
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                    nullptr);
                if (item.handle == INVALID_HANDLE_VALUE) {
                    throw std::runtime_error("failed to open expert source with unbuffered I/O: " + win_error(GetLastError()));
                }

                LARGE_INTEGER file_size = {};
                if (!GetFileSizeEx(item.handle, &file_size) || file_size.QuadPart < 0 ||
                    static_cast<uint64_t>(file_size.QuadPart) > std::numeric_limits<size_t>::max()) {
                    const DWORD error = GetLastError();
                    CloseHandle(item.handle);
                    throw std::runtime_error("failed to get expert source size: " + win_error(error));
                }
                item.size = static_cast<size_t>(file_size.QuadPart);

                FILE_STORAGE_INFO storage = {};
                if (!GetFileInformationByHandleEx(item.handle, FileStorageInfo, &storage, sizeof(storage))) {
                    const DWORD error = GetLastError();
                    CloseHandle(item.handle);
                    throw std::runtime_error("failed to query expert source sector size: " + win_error(error));
                }
                FILE_ALIGNMENT_INFO alignment_info = {};
                if (!GetFileInformationByHandleEx(item.handle, FileAlignmentInfo, &alignment_info, sizeof(alignment_info))) {
                    const DWORD error = GetLastError();
                    CloseHandle(item.handle);
                    throw std::runtime_error("failed to query expert source alignment: " + win_error(error));
                }

                item.alignment = std::max<size_t>({
                    4096,
                    storage.LogicalBytesPerSector,
                    storage.PhysicalBytesPerSectorForAtomicity,
                    static_cast<size_t>(alignment_info.AlignmentRequirement) + 1,
                });
                if (!is_power_of_two(item.alignment)) {
                    CloseHandle(item.handle);
                    throw std::runtime_error("expert source alignment is not a power of two");
                }
                files.push_back(item);
            }
        } catch (...) {
            for (file & item : files) {
                CloseHandle(item.handle);
            }
            throw;
        }
#else
        (void) paths;
        throw std::runtime_error("edge MoE unbuffered reader is supported only on Windows");
#endif
    }

    ~impl() {
#if defined(_WIN32)
        for (file & item : files) {
            if (item.handle != INVALID_HANDLE_VALUE) {
                CloseHandle(item.handle);
            }
        }
#endif
    }

#if defined(_WIN32)
    bool prepare(pending_read & pending, const llama_edge_moe_io_request & request, std::string & error) {
        if (request.file_index >= files.size()) {
            error = "expert read file index is outside the source list";
            return false;
        }
        if (request.size == 0) {
            return true;
        }
        if (request.destination == nullptr) {
            error = "expert read destination is null";
            return false;
        }

        file & source = files[request.file_index];
        if (request.offset > source.size || request.size > source.size - request.offset) {
            error = "expert read range is outside the source file";
            return false;
        }

        size_t aligned_offset = 0;
        size_t requested_end = 0;
        size_t aligned_end = 0;
        if (request.size > std::numeric_limits<size_t>::max() - request.offset ||
            !align_down(request.offset, source.alignment, aligned_offset) ||
            !align_up(request.offset + request.size, source.alignment, aligned_end)) {
            error = "expert read range alignment overflows size_t";
            return false;
        }
        requested_end = request.offset + request.size;
        if (requested_end > source.size) {
            error = "expert read range is outside the source file";
            return false;
        }
        const size_t aligned_size = aligned_end - aligned_offset;
        if (aligned_size == 0 || aligned_size > std::numeric_limits<DWORD>::max()) {
            error = "expert read is too large for one Windows request";
            return false;
        }

        pending.request = &request;
        pending.source = &source;
        pending.aligned_offset = aligned_offset;
        pending.aligned_size = aligned_size;
        pending.prefix = request.offset - aligned_offset;
        pending.event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (pending.event == nullptr) {
            error = "failed to create expert read event: " + win_error(GetLastError());
            return false;
        }
        pending.bounce = VirtualAlloc(nullptr, aligned_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (pending.bounce == nullptr) {
            error = "failed to allocate aligned expert read buffer: " + win_error(GetLastError());
            return false;
        }
        pending.overlapped.hEvent = pending.event;
        pending.overlapped.Offset = static_cast<DWORD>(aligned_offset & 0xffffffffu);
        pending.overlapped.OffsetHigh = static_cast<DWORD>((static_cast<uint64_t>(aligned_offset) >> 32) & 0xffffffffu);
        return true;
    }

    bool submit(pending_read & pending, std::string & error) {
        if (pending.request == nullptr || pending.request->size == 0) {
            return true;
        }
        const BOOL result = ReadFile(
            pending.source->handle,
            pending.bounce,
            static_cast<DWORD>(pending.aligned_size),
            nullptr,
            &pending.overlapped);
        if (!result && GetLastError() != ERROR_IO_PENDING) {
            error = "failed to submit unbuffered expert read: " + win_error(GetLastError());
            return false;
        }
        pending.submitted = true;
        return true;
    }

    bool complete(pending_read & pending, std::string & error) {
        if (!pending.submitted) {
            return pending.request == nullptr || pending.request->size == 0;
        }
        DWORD transferred = 0;
        if (!GetOverlappedResult(pending.source->handle, &pending.overlapped, &transferred, TRUE)) {
            error = "unbuffered expert read failed: " + win_error(GetLastError());
            pending.submitted = false;
            return false;
        }
        pending.submitted = false;
        if (static_cast<size_t>(transferred) < pending.prefix + pending.request->size) {
            error = "unbuffered expert read reached the end of file";
            return false;
        }
        std::memcpy(
            pending.request->destination,
            static_cast<const uint8_t *>(pending.bounce) + pending.prefix,
            pending.request->size);
        ++io_stats.requests;
        io_stats.bytes_requested += pending.request->size;
        io_stats.bytes_transferred += transferred;
        return true;
    }

    void cancel_and_drain(std::vector<pending_read> & pending) {
        for (pending_read & item : pending) {
            if (item.submitted) {
                CancelIoEx(item.source->handle, &item.overlapped);
            }
        }
        for (pending_read & item : pending) {
            if (item.submitted) {
                DWORD transferred = 0;
                GetOverlappedResult(item.source->handle, &item.overlapped, &transferred, TRUE);
                item.submitted = false;
            }
        }
    }
#endif

    bool read_many(const std::vector<llama_edge_moe_io_request> & requests, std::string & error) {
        error.clear();
#if defined(_WIN32)
        for (size_t begin = 0; begin < requests.size(); begin += max_in_flight) {
            const size_t count = std::min(max_in_flight, requests.size() - begin);
            std::vector<pending_read> pending(count);
            size_t bounce_bytes = 0;
            for (size_t i = 0; i < count; ++i) {
                if (!prepare(pending[i], requests[begin + i], error)) {
                    return false;
                }
                if (pending[i].aligned_size > std::numeric_limits<size_t>::max() - bounce_bytes) {
                    error = "expert read bounce buffer size overflows size_t";
                    return false;
                }
                bounce_bytes += pending[i].aligned_size;
            }
            io_stats.peak_bounce_bytes = std::max(io_stats.peak_bounce_bytes, bounce_bytes);

            for (pending_read & item : pending) {
                if (!submit(item, error)) {
                    cancel_and_drain(pending);
                    return false;
                }
            }
            for (pending_read & item : pending) {
                if (!complete(item, error)) {
                    cancel_and_drain(pending);
                    return false;
                }
            }
        }
        return true;
#else
        (void) requests;
        error = "edge MoE unbuffered reader is supported only on Windows";
        return false;
#endif
    }
};

llama_edge_moe_reader::llama_edge_moe_reader(const std::vector<std::string> & paths, const size_t max_in_flight) :
    pimpl(std::make_unique<impl>(paths, max_in_flight)) {}

llama_edge_moe_reader::~llama_edge_moe_reader() = default;

bool llama_edge_moe_reader::read(const llama_edge_moe_io_request & request, std::string & error) {
    return pimpl->read_many({request}, error);
}

bool llama_edge_moe_reader::read_many(const std::vector<llama_edge_moe_io_request> & requests, std::string & error) {
    return pimpl->read_many(requests, error);
}

size_t llama_edge_moe_reader::file_count() const {
#if defined(_WIN32)
    return pimpl->files.size();
#else
    return 0;
#endif
}

size_t llama_edge_moe_reader::file_alignment(const uint32_t file_index) const {
#if defined(_WIN32)
    if (file_index >= pimpl->files.size()) {
        throw std::out_of_range("edge MoE reader file index is outside the source list");
    }
    return pimpl->files[file_index].alignment;
#else
    (void) file_index;
    return 0;
#endif
}

const llama_edge_moe_io_stats & llama_edge_moe_reader::stats() const {
    return pimpl->io_stats;
}

struct llama_edge_moe_resident_set::impl {
    std::vector<std::pair<void *, size_t>> locked_ranges;
    size_t locked_bytes = 0;

    ~impl() {
#if defined(_WIN32)
        for (auto it = locked_ranges.rbegin(); it != locked_ranges.rend(); ++it) {
            VirtualUnlock(it->first, it->second);
        }
#endif
    }

    bool lock(
            const std::vector<std::pair<void *, size_t>> & input_ranges,
            const size_t process_budget_bytes,
            std::string & error) {
        error.clear();
        if (!locked_ranges.empty()) {
            error = "resident set is already locked";
            return false;
        }
        if (process_budget_bytes == 0) {
            error = "process memory budget must be non-zero";
            return false;
        }
#if defined(_WIN32)
        SYSTEM_INFO system_info = {};
        GetSystemInfo(&system_info);
        const size_t page_size = system_info.dwPageSize;

        std::vector<std::pair<uintptr_t, uintptr_t>> spans;
        spans.reserve(input_ranges.size());
        for (const auto & range : input_ranges) {
            if (range.first == nullptr || range.second == 0) {
                continue;
            }
            const uintptr_t begin_raw = reinterpret_cast<uintptr_t>(range.first);
            if (range.second > std::numeric_limits<uintptr_t>::max() - begin_raw) {
                error = "resident range overflows the address space";
                return false;
            }
            size_t begin = 0;
            size_t end = 0;
            if (!align_down(static_cast<size_t>(begin_raw), page_size, begin) ||
                !align_up(static_cast<size_t>(begin_raw + range.second), page_size, end)) {
                error = "resident range alignment overflows size_t";
                return false;
            }
            spans.emplace_back(begin, end);
        }
        std::sort(spans.begin(), spans.end());

        std::vector<std::pair<void *, size_t>> merged;
        for (const auto & span : spans) {
            if (!merged.empty()) {
                const uintptr_t previous_begin = reinterpret_cast<uintptr_t>(merged.back().first);
                const uintptr_t previous_end = previous_begin + merged.back().second;
                if (span.first <= previous_end) {
                    merged.back().second = std::max(previous_end, span.second) - previous_begin;
                    continue;
                }
            }
            merged.emplace_back(reinterpret_cast<void *>(span.first), span.second - span.first);
        }

        size_t total = 0;
        for (const auto & range : merged) {
            if (range.second > std::numeric_limits<size_t>::max() - total) {
                error = "resident range total overflows size_t";
                return false;
            }
            total += range.second;
        }
        constexpr size_t lock_overhead = 64ull * 1024 * 1024;
        if (total > process_budget_bytes || process_budget_bytes - total < lock_overhead) {
            error = "process memory budget cannot hold the non-expert resident set and lock overhead";
            return false;
        }

        SIZE_T current_min = 0;
        SIZE_T current_max = 0;
        DWORD current_flags = 0;
        if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &current_min, &current_max, &current_flags)) {
            error = "failed to query process working set limits: " + win_error(GetLastError());
            return false;
        }
        const SIZE_T requested_min = std::max<SIZE_T>(current_min, total + lock_overhead);
        if (!SetProcessWorkingSetSizeEx(
                GetCurrentProcess(),
                requested_min,
                process_budget_bytes,
                (current_flags & ~QUOTA_LIMITS_HARDWS_MAX_DISABLE) | QUOTA_LIMITS_HARDWS_MAX_ENABLE)) {
            error = "failed to set process working set budget: " + win_error(GetLastError());
            return false;
        }

        volatile uint8_t touch = 0;
        for (const auto & range : merged) {
            const uint8_t * data = static_cast<const uint8_t *>(range.first);
            for (size_t offset = 0; offset < range.second; offset += page_size) {
                touch ^= data[offset];
            }
            if (!VirtualLock(range.first, range.second)) {
                error = "failed to lock non-expert weight range: " + win_error(GetLastError());
                for (auto it = locked_ranges.rbegin(); it != locked_ranges.rend(); ++it) {
                    VirtualUnlock(it->first, it->second);
                }
                locked_ranges.clear();
                locked_bytes = 0;
                return false;
            }
            locked_ranges.push_back(range);
            locked_bytes += range.second;
        }
        (void) touch;
        return true;
#else
        (void) input_ranges;
        error = "edge MoE selective residency is supported only on Windows";
        return false;
#endif
    }
};

llama_edge_moe_resident_set::llama_edge_moe_resident_set() : pimpl(std::make_unique<impl>()) {}
llama_edge_moe_resident_set::~llama_edge_moe_resident_set() = default;

bool llama_edge_moe_resident_set::lock(
        const std::vector<std::pair<void *, size_t>> & ranges,
        const size_t process_budget_bytes,
        std::string & error) {
    return pimpl->lock(ranges, process_budget_bytes, error);
}

size_t llama_edge_moe_resident_set::size_bytes() const {
    return pimpl->locked_bytes;
}

size_t llama_edge_moe_resident_set::range_count() const {
    return pimpl->locked_ranges.size();
}

bool llama_edge_moe_get_process_memory(llama_edge_moe_process_memory & memory, std::string & error) {
    memory = {};
    error.clear();
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters), sizeof(counters))) {
        error = "failed to query process memory: " + win_error(GetLastError());
        return false;
    }
    memory.working_set_bytes = counters.WorkingSetSize;
    memory.peak_working_set_bytes = counters.PeakWorkingSetSize;
    memory.private_bytes = counters.PrivateUsage;
    return true;
#else
    error = "edge MoE process memory accounting is supported only on Windows";
    return false;
#endif
}
