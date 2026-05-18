#pragma once

#include <oboe/Oboe.h>
#include <AudioParams.hpp>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <vector>
#include <thread>
#include <atomic>
#include <GRUInference.hpp>
#include <BasicSessionHandler.hpp>
#include <npy.hpp>
#include <chrono>
#include <exception>
#ifdef __ANDROID__
#include <android/log.h>
#endif
// ---------------------------------------------------------------------------
// Player
//   Oboe output callback.  Reads raw audio from SharedAudioBuffer, runs it
//   through the GRU filter via GRUBinding (zero hot-path allocations), and
//   writes filtered samples back to Oboe.
// ---------------------------------------------------------------------------
template<IsIRRGRUInfo IRRGRU>
class Player : public oboe::AudioStreamDataCallback {
public:
    bool debug;
    bool profiling;

public:
    Player(
        const IRRGRU&       gru,
        const std::string   model_file,
        const std::string   ep_provider,
        const float         fc_normed,
        int32_t             sample_rate,
        int32_t             channels,
        audio_buffer&       buffer,
        const bool          cpu_only = false,
        const bool          dbg = false,
        const bool          prflg = false
    )
        : m_sample_rate    { sample_rate }
        , m_channels       { channels }
        , m_buffer         { buffer }
        , m_session_handle { model_file, ep_provider, cpu_only, dbg}
        // GRUBinding is constructed after the session so we can pass the session ref
        , m_gru_binding    { m_session_handle.session(), gru, fc_normed }
        , m_expected_frames { static_cast<int32_t>(gru.buffer_size()) }
        , m_expected_samples { m_expected_frames * channels }
        , m_input_to_infer { static_cast<size_t>(m_expected_samples * 8) }
        , m_output_from_infer { static_cast<size_t>(m_expected_samples * 8) }
        , debug { dbg }
        , profiling { prflg }
    {
        printf("normed frequency is %f\n", fc_normed);
        m_infer_thread = std::thread(&Player::inference_loop, this);
    }

    ~Player() override {
        stop_worker();
        printf("Player stats: input_overflow=%llu, output_underrun=%llu\n",
               static_cast<unsigned long long>(m_input_overflow_count.load()),
               static_cast<unsigned long long>(m_output_underrun_count.load()));
    }

    oboe::DataCallbackResult onAudioReady(
        oboe::AudioStream* /*stream*/,
        void*    audio_data,
        int32_t  num_frames) override
    {
        using namespace std::chrono;
        audio_sample_t* out = static_cast<audio_sample_t*>(audio_data);
        const int32_t   num_samples = num_frames * m_channels;

        // Pull dry signal from recorder ring-buffer.
        m_buffer.read(out, num_samples);

        // Guard: GRU binding is sized for a fixed buffer_size; skip inference
        // if Oboe gives us an unexpected frame count to avoid UB.
        if (num_frames != m_expected_frames) {
            fprintf(stderr,
                "Player: unexpected frame count %d (expected %d), passing through\n",
                num_frames, m_expected_frames);
            return oboe::DataCallbackResult::Continue;
        }

        // Queue dry samples for inference thread.
        const size_t queued = m_input_to_infer.write(out, num_samples);
        if (queued < static_cast<size_t>(num_samples)) {
            m_input_overflow_count.fetch_add(1, std::memory_order_relaxed);
        }

        // Try to get processed audio; on underrun keep dry pass-through.
        const size_t got = m_output_from_infer.read(out, num_samples);
        if (got < static_cast<size_t>(num_samples)) {
            m_output_underrun_count.fetch_add(1, std::memory_order_relaxed);
        }
        if(debug && got > 0){
            m_recorded_signal.insert(m_recorded_signal.end(), out, out + num_samples);
        }

        return oboe::DataCallbackResult::Continue;
    }

    void set_normed_fc(const float nfc) { 
        printf("normed frequency is %f\n", nfc);
        m_gru_binding.set_normed_fc(nfc); 
    }

    void set_sample_rate(int32_t sample_rate)
    {
        m_sample_rate = sample_rate;
    }

    void stop_worker()
    {
        const bool was_running = m_running.exchange(false, std::memory_order_acq_rel);
        if (was_running && m_infer_thread.joinable()) {
            m_infer_thread.join();
        }
    }

    uint64_t input_overflow_count() const
    {
        return m_input_overflow_count.load(std::memory_order_relaxed);
    }

    uint64_t output_underrun_count() const
    {
        return m_output_underrun_count.load(std::memory_order_relaxed);
    }

    void dump_realtime_report(const std::string& filename) const
    {
        std::ofstream out(filename, std::ios::trunc);
        if (!out.is_open()) {
            fprintf(stderr, "Failed to open realtime report file: %s\n", filename.c_str());
            return;
        }

        out << "# I-9a Realtime Report\n\n";
        out << "- input_overflow_count: " << input_overflow_count() << "\n";
        out << "- output_underrun_count: " << output_underrun_count() << "\n";
        out << "- input_ring_overflow_events: " << m_input_to_infer.overflow_events() << "\n";
        out << "- input_ring_underrun_events: " << m_input_to_infer.underrun_events() << "\n";
        out << "- output_ring_overflow_events: " << m_output_from_infer.overflow_events() << "\n";
        out << "- output_ring_underrun_events: " << m_output_from_infer.underrun_events() << "\n";

        if (!m_recorded_performances.empty()) {
            const auto minmax = std::minmax_element(
                m_recorded_performances.begin(), m_recorded_performances.end());
            const double sum = std::accumulate(
                m_recorded_performances.begin(), m_recorded_performances.end(), 0.0);
            std::vector<double> sorted = m_recorded_performances;
            std::sort(sorted.begin(), sorted.end());
            const size_t p95_idx = static_cast<size_t>(
                std::min(sorted.size() - 1, static_cast<size_t>(sorted.size() * 0.95)));

            out << "- inference_frames_profiled: " << m_recorded_performances.size() << "\n";
            out << "- inference_ms_min: " << *minmax.first << "\n";
            out << "- inference_ms_mean: " << (sum / m_recorded_performances.size()) << "\n";
            out << "- inference_ms_max: " << *minmax.second << "\n";
            out << "- inference_ms_p95: " << sorted[p95_idx] << "\n";
        } else {
            out << "- inference_frames_profiled: 0\n";
            out << "- inference_ms_min: N/A\n";
            out << "- inference_ms_mean: N/A\n";
            out << "- inference_ms_max: N/A\n";
            out << "- inference_ms_p95: N/A\n";
        }
    }

    void dump_debug(const std::string& filename)
    {
        npy::npy_data<audio_sample_t> d;
        d.data  = m_recorded_signal;
        d.shape = { m_recorded_signal.size() };
        npy::write_npy(filename, d);
    }

    void dump_profiling(const std::string& filename)
    {
        npy::npy_data<double> d;
        d.data  = m_recorded_performances;
        d.shape = { m_recorded_performances.size() };
        npy::write_npy(filename, d);
    }

private:
    void inference_loop() {
        try {
            std::vector<audio_sample_t> dry_block(static_cast<size_t>(m_expected_samples), 0.0f);
            std::vector<audio_sample_t> wet_block(static_cast<size_t>(m_expected_samples), 0.0f);

            while (m_running.load(std::memory_order_acquire)) {
                if (m_input_to_infer.available_to_read() < static_cast<size_t>(m_expected_samples)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                m_input_to_infer.read(dry_block.data(), dry_block.size());

                bool ret = false;
                decltype(std::chrono::high_resolution_clock::now()) start, end;
                if (profiling) {
                    start = std::chrono::high_resolution_clock::now();
                }

                ret = m_gru_binding.run(dry_block.data(), wet_block.data());
                if (!ret) {
                    std::memcpy(wet_block.data(), dry_block.data(),
                                dry_block.size() * sizeof(audio_sample_t));
                }

                if (profiling) {
                    end = std::chrono::high_resolution_clock::now();
                    m_recorded_performances.push_back(
                        duration_cast<std::chrono::milliseconds>(end - start).count());
                }

                m_output_from_infer.write(wet_block.data(), wet_block.size());
            }
        } catch (const std::exception &e) {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "GRUFilter",
                "inference_loop exception: %s", e.what());
#else
            fprintf(stderr, "[Player] inference_loop exception: %s\n", e.what());
#endif
        } catch (...) {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "GRUFilter",
                "inference_loop unknown exception");
#else
            fprintf(stderr, "[Player] inference_loop unknown exception\n");
#endif
        }
    }

    int32_t              m_sample_rate;
    int32_t              m_channels;
    audio_buffer&        m_buffer;

    BasicSessionHandler  m_session_handle;
    GRUBinding<IRRGRU>   m_gru_binding;

    int32_t              m_expected_frames;
    int32_t              m_expected_samples;
    audio_buffer         m_input_to_infer;
    audio_buffer         m_output_from_infer;
    std::atomic<bool>    m_running{true};
    std::atomic<uint64_t> m_input_overflow_count{0};
    std::atomic<uint64_t> m_output_underrun_count{0};
    std::thread          m_infer_thread;

    std::vector<audio_sample_t> m_recorded_signal;
    
    std::vector<double>  m_recorded_performances;
};
