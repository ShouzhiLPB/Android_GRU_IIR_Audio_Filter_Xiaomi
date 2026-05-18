#include <App.hpp>
#include <IIRGRUUtils.hpp>
#include <filesystem>
#include <stdexcept>

App::App(const cxxopts::ParseResult &args, std::atomic<bool> &running)
    : m_running{running}, m_audio_buffer{4096}, m_stream_handler{},
      m_recorder{nullptr}, m_player{nullptr}, m_debug{false},
      m_profiling{false}, m_run_duration{-1} {

  parse_options(args);

  // Build callbacks before opening streams so builders carry callbacks at open time.
  m_recorder = std::make_unique<Recorder>(
      0, 1, m_audio_buffer);
  m_player = std::make_unique<Player<decltype(m_gru)>>(
      m_gru, args["model"].as<std::string>(), args["ep"].as<std::string>(),
      normalize_frequency((float)args["fc"].as<int32_t>(),
                          48000.0f),
      0, 1, m_audio_buffer,
      args["cpu_only"].as<bool>());
  m_player->debug = m_debug;
  m_player->profiling = m_profiling;

  m_stream_handler.m_in_builder.setDataCallback(m_recorder.get())
      ->setSampleRate(m_stream_handler.get_in_sr())
      ->setFramesPerCallback(dsp_audio_buffer_size);

  m_stream_handler.m_out_builder.setDataCallback(m_player.get())
      ->setFramesPerCallback(dsp_audio_buffer_size);

  if (!m_stream_handler.create_streams(dsp_audio_buffer_size)) {
    throw std::runtime_error("Failed to create audio streams");
  }

  // After stream open, lock all runtime objects to actual device sample rates.
  m_recorder->set_sample_rate(m_stream_handler.get_in_sr());
  m_player->set_sample_rate(m_stream_handler.get_out_sr());
  m_player->set_normed_fc(
      normalize_frequency((float)args["fc"].as<int32_t>(),
                          (float)m_stream_handler.get_out_sr()));
}

App::~App() {
  if (m_recorder) {
    m_recorder->dump("input.npy");
  }

  if (m_player && m_player->debug)
    m_player->dump_debug("output.npy");
  if (m_player && m_player->profiling)
    m_player->dump_profiling("latency.npy");

  printf("Done.\n");
}

void App::run() {
  m_stream_handler.start_streams();
  printf("Audio passing through ...\n");

  {
    using namespace std::chrono;
    auto last_timestamp = steady_clock::now();
    int delta = 0;
    while (m_running) {
      delta = duration_cast<seconds>(steady_clock::now() - last_timestamp).count();
      if (m_run_duration > 0 && delta > m_run_duration)
        m_running = false;
      std::this_thread::sleep_for(150ms);
    }
  }
  m_stream_handler.stop_streams();
  if (m_player) {
    m_player->stop_worker();
    std::filesystem::create_directories("artifacts/test_results/realtime");
    m_player->dump_realtime_report("artifacts/test_results/realtime/i9a_runtime_report.md");
  }
}

void App::parse_options(const cxxopts::ParseResult &args) {
  bool profiling = false;
  if (args.count("profiling") > 0) {
    profiling = args["profiling"].as<bool>();
  }
  printf("Profiling active : %s\n", profiling ? "yes" : "no");

  bool debug = false;
  if (args.count("debug") > 0) {
    debug = args["debug"].as<bool>();
  }
  printf("Debug active : %s\n", debug ? "yes" : "no");

  if (args.count("run_duration") > 0) {
    m_run_duration = args["run_duration"].as<int>();
    printf("Run duration : %d seconds\n", m_run_duration);
  }

  bool cpu_only = false;
  if (args.count("cpu_only") > 0) {
    cpu_only = args["cpu_only"].as<int>();
  }
  printf("CPU only : %s \n", cpu_only ? "yes" : "no");

  m_debug = debug;
  m_profiling = profiling;
}