#include <App.hpp>
#include <BasicSessionHandler.hpp>
#include <GRUInference.hpp>
#include <IIRGRUInfo.hpp>
#include <IIRGRUUtils.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

std::atomic<bool> run = true;
void sigint_handler(int arg)
{
    run = false;
}

namespace {
struct WavData {
    int32_t sample_rate = 0;
    int16_t bits_per_sample = 0;
    int16_t channels = 0;
    std::vector<float> samples;
};

uint32_t read_u32(std::ifstream& in) {
    uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

uint16_t read_u16(std::ifstream& in) {
    uint16_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

WavData read_wav_mono(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open input wav: " + path);
    }

    char riff[4], wave[4];
    in.read(riff, 4);
    const uint32_t file_size = read_u32(in);
    (void)file_size;
    in.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("Invalid WAV header");
    }

    WavData wav{};
    std::vector<char> raw_data;
    while (in.good() && !in.eof()) {
        char chunk_id[4];
        in.read(chunk_id, 4);
        if (in.gcount() != 4) {
            break;
        }
        const uint32_t chunk_size = read_u32(in);
        const std::string id(chunk_id, 4);

        if (id == "fmt ") {
            const uint16_t audio_format = read_u16(in);
            wav.channels = static_cast<int16_t>(read_u16(in));
            wav.sample_rate = static_cast<int32_t>(read_u32(in));
            const uint32_t byte_rate = read_u32(in);
            (void)byte_rate;
            const uint16_t block_align = read_u16(in);
            (void)block_align;
            wav.bits_per_sample = static_cast<int16_t>(read_u16(in));
            if (chunk_size > 16) {
                in.seekg(chunk_size - 16, std::ios::cur);
            }
            if (audio_format != 3 && audio_format != 1) {
                throw std::runtime_error("Only PCM16 or float32 WAV are supported");
            }
        } else if (id == "data") {
            raw_data.resize(chunk_size);
            in.read(raw_data.data(), static_cast<std::streamsize>(chunk_size));
        } else {
            in.seekg(chunk_size, std::ios::cur);
        }
    }

    if (wav.channels != 1) {
        throw std::runtime_error("Only mono wav is supported in file mode");
    }
    if (raw_data.empty()) {
        throw std::runtime_error("WAV data chunk is empty");
    }

    if (wav.bits_per_sample == 32) {
        const size_t count = raw_data.size() / sizeof(float);
        wav.samples.resize(count);
        std::memcpy(wav.samples.data(), raw_data.data(), count * sizeof(float));
    } else if (wav.bits_per_sample == 16) {
        const size_t count = raw_data.size() / sizeof(int16_t);
        wav.samples.resize(count);
        const auto* pcm = reinterpret_cast<const int16_t*>(raw_data.data());
        for (size_t i = 0; i < count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
        }
    } else {
        throw std::runtime_error("Unsupported bits_per_sample for input wav");
    }

    return wav;
}

void write_wav_float32_mono(const std::string& path, const std::vector<float>& samples, int32_t sample_rate) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output wav: " + path);
    }

    const uint16_t audio_format = 3; // IEEE float
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 32;
    const uint16_t block_align = channels * (bits_per_sample / 8);
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(float));
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4);

    const uint32_t fmt_chunk_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    out.write(reinterpret_cast<const char*>(samples.data()), static_cast<std::streamsize>(data_size));
}

void run_file_mode(const cxxopts::ParseResult& args) {
    static constexpr int32_t batch_size = 1;
    static constexpr int32_t buffer_size = 256;
    static constexpr int32_t input_size = 2;
    static constexpr int32_t hidden_size = 64;
    static constexpr int32_t num_layers = 2;
    using GruInfo = IRRGRUInfo<batch_size, buffer_size, input_size, hidden_size, num_layers>;

    const std::string input_path = args["input"].as<std::string>();
    const std::string output_path = args["output"].as<std::string>();
    const std::string model_path = args["model"].as<std::string>();
    const std::string ep = args["ep"].as<std::string>();
    const bool cpu_only = args["cpu_only"].as<bool>();
    const bool debug = args["debug"].as<bool>();

    const WavData input_wav = read_wav_mono(input_path);
    const float fc_hz = static_cast<float>(args.count("fc") ? args["fc"].as<int32_t>() : 1000);
    const float fc_norm = normalize_frequency(fc_hz, static_cast<float>(input_wav.sample_rate));
    std::cerr << "[FILE MODE] normed frequency is " << fc_norm << std::endl;

    BasicSessionHandler session_handler(model_path, ep, cpu_only, debug);
    GruInfo gru_info;
    GRUBinding<GruInfo> binding(session_handler.session(), gru_info, fc_norm);

    std::vector<float> output(input_wav.samples.size(), 0.0f);
    std::vector<float> in_block(buffer_size, 0.0f);
    std::vector<float> out_block(buffer_size, 0.0f);
    std::vector<double> lat_ms;
    lat_ms.reserve((input_wav.samples.size() + buffer_size - 1) / buffer_size);

    for (size_t offset = 0; offset < input_wav.samples.size(); offset += buffer_size) {
        const size_t n = std::min(static_cast<size_t>(buffer_size), input_wav.samples.size() - offset);
        std::fill(in_block.begin(), in_block.end(), 0.0f);
        std::memcpy(in_block.data(), input_wav.samples.data() + offset, n * sizeof(float));

        const auto t0 = std::chrono::high_resolution_clock::now();
        binding.run(in_block.data(), out_block.data());
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
        lat_ms.push_back(ms);
        std::cout << "[BLOCK] offset=" << offset << " samples=" << n << " infer_ms=" << ms << std::endl;

        std::memcpy(output.data() + offset, out_block.data(), n * sizeof(float));
    }

    write_wav_float32_mono(output_path, output, input_wav.sample_rate);

    if (!lat_ms.empty()) {
        auto sorted = lat_ms;
        std::sort(sorted.begin(), sorted.end());
        const auto [min_it, max_it] = std::minmax_element(lat_ms.begin(), lat_ms.end());
        const double mean = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) / static_cast<double>(lat_ms.size());
        const size_t p95_idx = static_cast<size_t>(std::min(sorted.size() - 1, static_cast<size_t>(sorted.size() * 0.95)));
        std::cout << "[SUMMARY] blocks=" << lat_ms.size()
                  << " min_ms=" << *min_it
                  << " mean_ms=" << mean
                  << " max_ms=" << *max_it
                  << " p95_ms=" << sorted[p95_idx]
                  << std::endl;
    }
    std::cout << "[FILE MODE] Wrote output wav: " << output_path << std::endl;
}
} // namespace

int main(int argc, char ** argv)
{
    signal(SIGINT, sigint_handler);
    cxxopts::Options options { "FilterProgram", "Audio passing through filter program" };

    options.add_options()
    ("h,help", "Print usage")
    ("m,model",        "File containing the model to load (expected .onnx file)", cxxopts::value<std::string>()->default_value("./lowpass_rnn.onnx"))
    ("f,fc",        "Cutoff frequency (Hz)", cxxopts::value<int32_t>())
    ("p,profiling", "Profiling mode : get information about session perfomance (boolean)", cxxopts::value<bool>()->default_value("false"))
    ("r,run_duration", "Run duration (seconds): indicate of much time to run the program (if not specified, the program runs until stopped with Ctrl+C)", cxxopts::value<int>())
    ("d,debug", "Debug mode : get session input and output signals (boolean)", cxxopts::value<bool>()->default_value("false"))
    ("e,ep", "Execution Provider preference: nnapi | xnnpack | cpu", 
            cxxopts::value<std::string>()->default_value("nnapi"))
    ("i,input", "Input wav path for offline file mode", cxxopts::value<std::string>())
    ("o,output", "Output wav path for offline file mode", cxxopts::value<std::string>()->default_value("output.wav"))
    ("c,cpu_only", "CPU only mode : NNAPI will not try to run inference on GPU/NPU (boolean)", cxxopts::value<bool>()->default_value("false"))
    ;

    auto args = options.parse(argc, argv);
    
    if (args.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    if (args.count("input")) {
      run_file_mode(args);
      return EXIT_SUCCESS;
    }

    App app(args, run);
    app.run();
    return  EXIT_SUCCESS;
}
