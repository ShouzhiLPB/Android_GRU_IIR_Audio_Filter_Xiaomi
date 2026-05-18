#pragma once

#include <cpu_provider_factory.h>
#include <nnapi_provider_factory.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <thread>
#include <vector>

class BasicSessionHandler {
public:
  BasicSessionHandler(const std::string model_filename, const std::string ep_name, const bool cpu_only = false, const bool debug = false)
      : m_env(debug ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_FATAL, "lowpass_rnn\n") {
    std::vector<std::string> providers = Ort::GetAvailableProviders();
    for (const auto &p : providers) {
      std::cout << "  - " << p << std::endl;
    }

    const std::string requested_ep = normalize_ep_name(ep_name);
    const std::vector<std::string> ep_chain = make_ep_chain(requested_ep);

    const unsigned hc = std::thread::hardware_concurrency();
    const int threads = std::max(1, static_cast<int>(hc > 0 ? hc / 2 : 1));
    m_session_options.SetIntraOpNumThreads(threads);
    m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    m_session_options.SetLogSeverityLevel(0);
    m_session_options.SetLogId("ort_session");

    bool ep_selected = false;
    for (const auto &candidate : ep_chain) {
      if (candidate == "nnapi") {
        uint32_t nnapi_flags = cpu_only ? NNAPI_FLAG_CPU_ONLY : NNAPI_FLAG_CPU_DISABLED;
        OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_Nnapi(m_session_options, nnapi_flags);
        if (status != nullptr) {
          std::string err = Ort::GetApi().GetErrorMessage(status);
          Ort::GetApi().ReleaseStatus(status);
          std::cerr << "[WARN] Failed to enable NNAPI EP: " << err << std::endl;
          continue;
        }
        m_selected_ep_name = "NnapiExecutionProvider";
        ep_selected = true;
        break;
      }

      if (candidate == "xnnpack") {
        try {
          m_session_options.AppendExecutionProvider("XnnpackExecutionProvider");
          m_selected_ep_name = "XnnpackExecutionProvider";
          ep_selected = true;
          break;
        } catch (const Ort::Exception &e) {
          std::cerr << "[WARN] Failed to enable XNNPACK EP: " << e.what() << std::endl;
          continue;
        }
      }

      if (candidate == "cpu") {
        m_session_options.AppendExecutionProvider_CPU(threads);
        m_selected_ep_name = "CPUExecutionProvider";
        ep_selected = true;
        break;
      }
    }

    if (!ep_selected) {
      m_session_options.AppendExecutionProvider_CPU(threads);
      m_selected_ep_name = "CPUExecutionProvider";
    }

    std::cerr << "[INFO] Selected EP: " << m_selected_ep_name << std::endl;

    m_session = Ort::Session(m_env, model_filename.c_str(), m_session_options);
  }

  Ort::Session &session() { return m_session.value(); }

private:
  static std::string normalize_ep_name(const std::string &raw) {
    std::string value = raw;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
  }

  static std::vector<std::string> make_ep_chain(const std::string &requested) {
    if (requested == "cpu") {
      return {"cpu"};
    }
    if (requested == "xnnpack") {
      return {"xnnpack", "cpu"};
    }
    return {"nnapi", "xnnpack", "cpu"};
  }

  Ort::Env m_env;
  Ort::SessionOptions m_session_options;
  std::optional<Ort::Session> m_session;
  std::string m_selected_ep_name;
};