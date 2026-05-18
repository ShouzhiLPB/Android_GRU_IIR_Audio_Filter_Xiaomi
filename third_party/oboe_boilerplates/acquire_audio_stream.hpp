#pragma once

#include <oboe/Oboe.h>
#include <cstdint>

/**
 * Configure an Oboe AudioStreamBuilder for the requested stream type.
 * Returns a positive preferred sample rate (Hz) for downstream stream open.
 */
int32_t acquire_audio_stream(
    oboe::AudioStreamBuilder& builder,
    oboe::Direction direction,
    oboe::PerformanceMode performanceMode,
    oboe::SharingMode sharingMode,
    oboe::AudioFormat format,
    oboe::ChannelCount channelCount);
