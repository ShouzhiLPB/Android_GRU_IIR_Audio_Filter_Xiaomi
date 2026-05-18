#pragma once
#include <oboe/Oboe.h>

bool acquire_audio_stream(
    oboe::AudioStreamBuilder &builder,
    oboe::Direction direction,
    oboe::PerformanceMode perfMode = oboe::PerformanceMode::LowLatency,
    oboe::SharingMode sharingMode = oboe::SharingMode::Exclusive,
    oboe::AudioFormat format = oboe::AudioFormat::Float,
    oboe::ChannelCount channelCount = oboe::ChannelCount::Mono,
    int sampleRate = 0,
    int deviceId = 0
);