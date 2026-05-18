#pragma once

#include <oboe/Oboe.h>
#include <AudioParams.hpp>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <chrono>

//debug
#include <npy.hpp>
/////

// ---------------------------------------------------------------------------
// Callback: pulls float samples from Oboe, converts to int16, stores them
// ---------------------------------------------------------------------------
class Recorder : public oboe::AudioStreamDataCallback {
public:
    Recorder(
        int32_t sample_rate, 
        int32_t channels, 
        audio_buffer& buffer
    )
        : m_sample_rate { sample_rate }
        , m_channels { channels }
        , m_buffer { buffer }
    {
    }

    oboe::DataCallbackResult onAudioReady(
        oboe::AudioStream* /*stream*/,
        void* audio_data,
        int32_t num_frames) override
    {
        const audio_sample_t* in = static_cast<const audio_sample_t*>(audio_data);
        int32_t num_samples = num_frames * m_channels;

        m_buffer.write(in, num_samples);
        // Instead of the for loop:
        m_recorded_signal.insert(m_recorded_signal.end(), in, in + num_samples);
        return oboe::DataCallbackResult::Continue;
    }

    void dump(const std::string filename)
    {
        npy::npy_data<audio_sample_t> d;
        d.data = m_recorded_signal;
        d.shape = { m_recorded_signal.size() };

        npy::write_npy(filename, d);
    }

    void set_sample_rate(int32_t sample_rate)
    {
        m_sample_rate = sample_rate;
    }

private:
    int32_t m_sample_rate;
    int32_t m_channels;
    audio_buffer& m_buffer;

    std::vector<audio_sample_t> m_recorded_signal;
};