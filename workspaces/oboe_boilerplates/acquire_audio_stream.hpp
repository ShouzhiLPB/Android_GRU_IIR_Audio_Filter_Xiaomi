#pragma once
#include <oboe/Oboe.h>

int acquire_audio_stream(
     oboe::AudioStreamBuilder &builder,
     oboe::Direction direction,
     oboe::PerformanceMode performanceMode,
     oboe::SharingMode sharingMode,
     oboe::AudioFormat format,
     int channelCount,
     int sampleRate = 48000
);
