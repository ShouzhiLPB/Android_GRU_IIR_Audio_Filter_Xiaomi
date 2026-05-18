// JNI glue: Kotlin FilterEngine -> Oboe I/O + GRUBinding (ONNX Runtime).
#include <jni.h>

#include <mutex>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <AudioParams.hpp>
#include <BasicSessionHandler.hpp>
#include <GRUInference.hpp>
#include <IIRGRUInfo.hpp>
#include <IIRGRUUtils.hpp>
#include <IOStreamHandler.hpp>
#include <Player.hpp>
#include <Recorder.hpp>

namespace {
constexpr int32_t kBatchSize = 1;
constexpr int32_t kDspBufferSize = 256;
constexpr int32_t kAlgoBufferSize = 256;
constexpr int32_t kInputSize = 2;
constexpr int32_t kHiddenSize = 64;
constexpr int32_t kNumLayers = 2;

using GruInfo = IRRGRUInfo<kBatchSize, kAlgoBufferSize, kInputSize, kHiddenSize, kNumLayers>;

struct NativeEngine {
  explicit NativeEngine(const std::string &model_path, const std::string &ep, float fc, bool cpu_only)
      : audio_buffer(4096),
        stream_handler(),
        recorder(nullptr),
        player(nullptr),
        started(false) {
    recorder = std::make_unique<Recorder>(0, 1, audio_buffer);
    player = std::make_unique<Player<GruInfo>>(
        gru, model_path, ep, normalize_frequency(fc, 48000.0f), 0, 1, audio_buffer, cpu_only);

    stream_handler.m_in_builder.setDataCallback(recorder.get())
        ->setFramesPerCallback(kDspBufferSize);
    stream_handler.m_out_builder.setDataCallback(player.get())
        ->setFramesPerCallback(kDspBufferSize);

    if (!stream_handler.create_streams(kDspBufferSize)) {
      throw std::runtime_error("Failed to create Oboe streams");
    }

    recorder->set_sample_rate(stream_handler.get_in_sr());
    player->set_sample_rate(stream_handler.get_out_sr());
    player->set_normed_fc(normalize_frequency(fc, static_cast<float>(stream_handler.get_out_sr())));
  }

  ~NativeEngine() {
    stop();
  }

  bool start() {
    if (started) return true;
    started = stream_handler.start_streams();
    return started;
  }

  void stop() {
    if (!started) return;
    stream_handler.stop_streams();
    player->stop_worker();
    started = false;
  }

  void set_fc(float fc) {
    if (!player) {
      return;
    }
    player->set_normed_fc(
        normalize_frequency(fc, static_cast<float>(stream_handler.get_out_sr())));
  }

  std::string stats() const {
    std::ostringstream oss;
    oss << "input_sr=" << stream_handler.get_in_sr()
        << ", output_sr=" << stream_handler.get_out_sr()
        << ", input_overflow=" << player->input_overflow_count()
        << ", output_underrun=" << player->output_underrun_count();
    return oss.str();
  }

  audio_buffer audio_buffer;
  IOStreamHandler stream_handler;
  GruInfo gru;
  std::unique_ptr<Recorder> recorder;
  std::unique_ptr<Player<GruInfo>> player;
  bool started;
};

std::mutex g_mutex;
std::unique_ptr<NativeEngine> g_engine;

// Last successful nativeInit parameters — used to rebuild after nativeStop destroys the engine.
std::string g_last_model_path;
std::string g_last_ep;
float g_last_fc = 1000.f;
bool g_last_cpu_only = false;
bool g_have_last_params = false;

std::string JStringToStd(JNIEnv *env, jstring s) {
  if (!s) return {};
  const char *chars = env->GetStringUTFChars(s, nullptr);
  std::string value(chars ? chars : "");
  if (chars) env->ReleaseStringUTFChars(s, chars);
  return value;
}

static void throw_java_runtime(JNIEnv *env, const char *msg) {
  jclass cls = env->FindClass("java/lang/RuntimeException");
  if (cls != nullptr) {
    env->ThrowNew(cls, msg);
  } else {
    env->ExceptionClear();
  }
}
}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gru_filter_FilterEngine_nativeInit(JNIEnv *env, jobject /*thiz*/, jstring model_path,
                                             jfloat fc, jstring ep, jboolean cpu_only) {
  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_engine.reset();
    g_last_model_path = JStringToStd(env, model_path);
    g_last_ep = JStringToStd(env, ep);
    g_last_fc = fc;
    g_last_cpu_only = (cpu_only == JNI_TRUE);
    g_engine = std::make_unique<NativeEngine>(g_last_model_path, g_last_ep, g_last_fc, g_last_cpu_only);
    g_have_last_params = true;
    return JNI_TRUE;
  } catch (const std::exception &e) {
    g_engine.reset();
    g_have_last_params = false;
    throw_java_runtime(env, e.what());
    return JNI_FALSE;
  } catch (...) {
    g_engine.reset();
    g_have_last_params = false;
    throw_java_runtime(env, "Unknown native error");
    return JNI_FALSE;
  }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gru_filter_FilterEngine_nativeStart(JNIEnv *env, jobject /*thiz*/) {
  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) {
      if (!g_have_last_params) {
        throw_java_runtime(env, "Engine not initialized");
        return JNI_FALSE;
      }
      g_engine = std::make_unique<NativeEngine>(g_last_model_path, g_last_ep, g_last_fc, g_last_cpu_only);
    }
    if (!g_engine->start()) {
      throw_java_runtime(env, "Failed to start Oboe streams");
      return JNI_FALSE;
    }
    return JNI_TRUE;
  } catch (const std::exception &e) {
    throw_java_runtime(env, e.what());
    return JNI_FALSE;
  } catch (...) {
    throw_java_runtime(env, "Unknown native error");
    return JNI_FALSE;
  }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gru_filter_FilterEngine_nativeStop(JNIEnv *env, jobject /*thiz*/) {
  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) {
      return JNI_TRUE;
    }
    g_engine->stop();
    g_engine.reset();
    return JNI_TRUE;
  } catch (const std::exception &e) {
    throw_java_runtime(env, e.what());
    return JNI_FALSE;
  } catch (...) {
    throw_java_runtime(env, "Unknown native error");
    return JNI_FALSE;
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_gru_filter_FilterEngine_nativeGetStats(JNIEnv *env, jobject /*thiz*/) {
  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::string text = g_engine ? g_engine->stats() : "engine_not_initialized";
    return env->NewStringUTF(text.c_str());
  } catch (const std::exception &e) {
    throw_java_runtime(env, e.what());
    return nullptr;
  } catch (...) {
    throw_java_runtime(env, "Unknown native error");
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_gru_filter_FilterEngine_nativeSetFc(JNIEnv *env, jobject /*thiz*/, jfloat fc) {
  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_fc = fc;
    if (g_engine) {
      g_engine->set_fc(fc);
    }
  } catch (const std::exception &e) {
    throw_java_runtime(env, e.what());
  } catch (...) {
    throw_java_runtime(env, "Unknown native error");
  }
}
