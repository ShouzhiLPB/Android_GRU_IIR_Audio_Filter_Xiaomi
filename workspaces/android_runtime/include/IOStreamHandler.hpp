#pragma once

#include <oboe/Oboe.h>
#include <cstdint>
#include <acquire_audio_stream.hpp>

class IOStreamHandler
{
public:
    IOStreamHandler()
        : m_in_sr(0), m_out_sr(0)
    {
        if (!acquire_audio_stream(
                m_in_builder,
                oboe::Direction::Input,
                oboe::PerformanceMode::LowLatency,
                oboe::SharingMode::Exclusive,
                oboe::AudioFormat::Float,
                oboe::ChannelCount::Mono
            )) {
            fprintf(stderr, "Could not configure input stream builder\n");
        }

        if (!acquire_audio_stream(
                m_out_builder,
                oboe::Direction::Output,
                oboe::PerformanceMode::LowLatency,
                oboe::SharingMode::Exclusive,
                oboe::AudioFormat::Float,
                oboe::ChannelCount::Mono
            )) {
            fprintf(stderr, "Could not configure output stream builder\n");
        }
    }

    const int32_t& get_in_sr() const { return m_in_sr; }
    const int32_t& get_out_sr() const { return m_out_sr; }

    bool create_streams(const int32_t audio_buffer_size)
    {
        auto result = m_in_builder.openManagedStream(m_instream);
        if (result != oboe::Result::OK) {
            fprintf(stderr, "Failed to open input audio stream: %s\n", oboe::convertToText(result));
            return false;
        }

        result = m_out_builder.openManagedStream(m_outstream);
        if (result != oboe::Result::OK) {
            fprintf(stderr, "Failed to open output audio stream: %s\n", oboe::convertToText(result));
            return false;
        }

        // Actual sample rates are only known after streams are opened.
        m_in_sr = m_instream->getSampleRate();
        m_out_sr = m_outstream->getSampleRate();

        auto in_buf_result = m_instream->setBufferSizeInFrames(audio_buffer_size);
        auto out_buf_result = m_outstream->setBufferSizeInFrames(audio_buffer_size);

        if (in_buf_result) {
            printf("Input buffer size: %d frames\n", in_buf_result.value());
        } else {
            fprintf(stderr, "Failed to set input buffer size: %s\n", oboe::convertToText(in_buf_result.error()));
            return false;
        }
        if (out_buf_result) {
            printf("Output buffer size: %d frames\n", out_buf_result.value());
        } else {
            fprintf(stderr, "Failed to set output buffer size: %s\n", oboe::convertToText(out_buf_result.error()));
            return false;
        }

        return true;
    }

    bool start_streams()
    {
        auto result = m_instream->requestStart();
        if (result != oboe::Result::OK) {
            fprintf(stderr, "Failed to start stream: %s\n", oboe::convertToText(result));
            return false;
        }
        result = m_outstream->requestStart();
        if (result != oboe::Result::OK) {
            fprintf(stderr, "Failed to start stream: %s\n", oboe::convertToText(result));
            return false;
        }

        return true;
    }

    void stop_streams()
    {
        m_outstream->requestStop();
        m_outstream->close();
        m_instream->requestStop();
        m_instream->close();
    }
    oboe::AudioStreamBuilder    m_in_builder, 
                                m_out_builder;
private:
    oboe::ManagedStream m_instream, m_outstream;

    int32_t                     m_in_sr, m_out_sr;
};