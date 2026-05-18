#pragma once

#include <algorithm>
#include <cstring>
#include <vector>
#include <atomic>

template<typename SampleType>
class SharedAudioBuffer {
public:
    SharedAudioBuffer(size_t capacity) 
    : m_buffer(capacity, 0.0f), m_capacity(capacity) {}

    // Called from input callback (producer)
    size_t write(const SampleType* src, size_t count) {
        size_t written = 0;
        while (written < count) {
            size_t w = m_write_pos.load(std::memory_order_relaxed);
            size_t r = m_read_pos.load(std::memory_order_acquire);
            size_t available = m_capacity - 1 - ((w - r + m_capacity) % m_capacity);
            if (available == 0) break; // overflow: drop
            size_t chunk = std::min(count - written, available);
            // write up to the end of the buffer, then wrap
            size_t to_end = m_capacity - (w % m_capacity);
            size_t first  = std::min(chunk, to_end);
            size_t second = chunk - first;
            memcpy(&m_buffer[w % m_capacity], src + written, first * sizeof(SampleType));
            if (second) memcpy(&m_buffer[0], src + written + first, second * sizeof(SampleType));
            m_write_pos.store((w + chunk) % m_capacity, std::memory_order_release);
            written += chunk;
        }
        if (written < count) {
            m_overflow_events.fetch_add(1, std::memory_order_relaxed);
        }
        return written;
    }

    // Called from output callback (consumer)
    size_t read(SampleType* dst, size_t count) {
        size_t r = m_read_pos.load(std::memory_order_relaxed);
        size_t w = m_write_pos.load(std::memory_order_acquire);
        size_t available = (w - r + m_capacity) % m_capacity;
        size_t chunk = std::min(count, available);
        // silence-pad if underrun
        if (chunk < count) memset(dst + chunk, 0, (count - chunk) * sizeof(SampleType));
        size_t to_end  = m_capacity - (r % m_capacity);
        size_t first  = std::min(chunk, to_end);
        size_t second = chunk - first;
        memcpy(dst, &m_buffer[r % m_capacity], first * sizeof(SampleType));
        if (second) memcpy(dst + first, &m_buffer[0], second * sizeof(SampleType));
        m_read_pos.store((r + chunk) % m_capacity, std::memory_order_release);
        if (chunk < count) {
            m_underrun_events.fetch_add(1, std::memory_order_relaxed);
        }
        return chunk;
    }

    size_t available_to_read() const {
        const size_t r = m_read_pos.load(std::memory_order_acquire);
        const size_t w = m_write_pos.load(std::memory_order_acquire);
        return (w - r + m_capacity) % m_capacity;
    }

    uint64_t overflow_events() const {
        return m_overflow_events.load(std::memory_order_relaxed);
    }

    uint64_t underrun_events() const {
        return m_underrun_events.load(std::memory_order_relaxed);
    }

private:
    std::vector<SampleType> m_buffer;
    size_t                  m_capacity;
    std::atomic<size_t>     m_write_pos{0};
    std::atomic<size_t>     m_read_pos{0};
    std::atomic<uint64_t>   m_overflow_events{0};
    std::atomic<uint64_t>   m_underrun_events{0};
};