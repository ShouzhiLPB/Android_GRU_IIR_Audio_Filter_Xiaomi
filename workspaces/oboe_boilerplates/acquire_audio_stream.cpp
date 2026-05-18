#include "acquire_audio_stream.hpp"

int acquire_audio_stream(
     oboe::AudioStreamBuilder &builder,
     oboe::Direction direction,
     oboe::PerformanceMode performanceMode,
     oboe::SharingMode sharingMode,
     oboe::AudioFormat format,
     int channelCount,
     int sampleRate
) {
     builder.setDirection(direction)
            ->setSampleRate(sampleRate)
            ->setChannelCount(channelCount)
            ->setFormat(format)
            ->setPerformanceMode(performanceMode)
            ->setSharingMode(sharingMode);
     return sampleRate;
}
