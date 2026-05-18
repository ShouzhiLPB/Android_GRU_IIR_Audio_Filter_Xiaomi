#pragma once

#include <AudioParams.hpp>
#include <Recorder.hpp>
#include <Player.hpp>
#include <acquire_audio_stream.hpp>
#include <signal.h>
#include <IIRGRUInfo.hpp>
#include <IIRGRUUtils.hpp>
#include <onnxruntime_cxx_api.h>
#include <cxxopts.hpp>
#include <chrono>
#include <memory>
#include <IOStreamHandler.hpp>

static constexpr int32_t batch_size = 1;
static constexpr int32_t dsp_audio_buffer_size = 256;
static constexpr int32_t algo_audio_buffer_size = 256;
static constexpr int32_t input_size = 2;
static constexpr int32_t hidden_size = 64;
static constexpr int32_t num_layers = 2;

class App
{
public:
    App(const cxxopts::ParseResult& args, std::atomic<bool>& running);
    ~App(); // does the dumping job

    void run();

private:
    void parse_options(const cxxopts::ParseResult&);

private:
    std::atomic<bool>&          m_running;

    audio_buffer                m_audio_buffer;
    IOStreamHandler             m_stream_handler;
    IRRGRUInfo< batch_size, 
                algo_audio_buffer_size, 
                input_size, 
                hidden_size, 
                num_layers>     m_gru;

    std::unique_ptr<Recorder>                    m_recorder;
    std::unique_ptr<Player<decltype(m_gru)>>     m_player;
    bool                        m_debug;
    bool                        m_profiling;
    int                         m_run_duration;
};
