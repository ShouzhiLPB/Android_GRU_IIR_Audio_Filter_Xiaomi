#include "acquire_audio_stream.hpp"

/**
 * Apply common low-latency defaults used by the RT filter app and return a sample rate.
 * The device may still resample when the stream is opened; this value seeds the builder.
 */
int32_t acquire_audio_stream(
    oboe::AudioStreamBuilder& builder,
    oboe::Direction direction,
    oboe::PerformanceMode performanceMode,
    oboe::SharingMode sharingMode,
    oboe::AudioFormat format,
    oboe::ChannelCount channelCount)
{
    builder.setDirection(direction)
        ->setPerformanceMode(performanceMode)
        ->setSharingMode(sharingMode)
        ->setFormat(format)
        ->setChannelCount(static_cast<int32_t>(channelCount));

    constexpr int32_t kPreferredSampleRate = 48000;
    builder.setSampleRate(kPreferredSampleRate);
    return kPreferredSampleRate;
}
