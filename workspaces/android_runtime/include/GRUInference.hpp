#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <IIRGRUInfo.hpp>

// ---------------------------------------------------------------------------
// GRUBinding
//   Owns all input/output tensor memory and the IoBinding object.
//   Construct once; call run() every audio callback — zero heap allocations
//   in the hot path.
//
//   Memory layout
//   ─────────────
//   m_x_data      : interleaved (audio, fc_norm) pairs  → shape (1, B, 2)
//   m_hidden[0/1] : double-buffered hidden state        → shape (L, 1, H)
//   m_output      : filtered audio samples              → shape (1, B, 1)
//                   (or (1, B) depending on your model export — adjust if needed)
//
//   After run(), filtered samples are in output_ptr() / output_size().
// ---------------------------------------------------------------------------
template<IsIRRGRUInfo IRRGRU>
class GRUBinding {
public:
    GRUBinding(Ort::Session& session, const IRRGRU& gru, const float fc_normed)
        : m_session   { session }
        , m_gru       { gru }
        , m_fc_normed { fc_normed }
        , m_memory_info { Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) }
        , m_binding   { session }
    {
        const auto B = gru.buffer_size();
        const auto H = gru.hidden_size();
        const auto L = gru.num_layers();

        // ── Allocate owned buffers ───────────────────────────────────────────
        m_x_data.resize(B * gru.input_size());
        // Pre-fill the fc channel (odd indices); audio channel filled per-call
        for (size_t i = 0; i < static_cast<size_t>(B); ++i) {
            m_x_data[i * 2]     = 0.f;        // audio — overwritten each call
            m_x_data[i * 2 + 1] = fc_normed;  // fc    — constant
        }

        const size_t hidden_elems = static_cast<size_t>(L * gru.batch_size() * H);
        m_hidden[0].assign(hidden_elems, 0.f);
        m_hidden[1].assign(hidden_elems, 0.f);
        m_hidden_ping = 0; // index of current hidden_in buffer

        m_output.resize(static_cast<size_t>(B * gru.batch_size()));

        // ── Shapes ──────────────────────────────────────────────────────────
        m_x_shape      = { gru.batch_size(), B, gru.input_size() };
        m_hidden_shape  = { L, gru.batch_size(), H };
        m_output_shape  = { gru.batch_size(), B, 1 }; // adjust if your model exports (1,B)

        // ── Bind inputs ─────────────────────────────────────────────────────
        // Input "x": we update m_x_data in-place before each Run, so the
        // binding pointer stays valid — ORT reads the latest data automatically.
        Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_x_data.data(), m_x_data.size(),
            m_x_shape.data(), m_x_shape.size()
        );
        m_binding.BindInput("x", x_tensor);

        // Input "hidden_in": will be re-bound each call (ping-pong swap),
        // so we do the initial bind here and update in run().
        Ort::Value h_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_hidden[m_hidden_ping].data(), hidden_elems,
            m_hidden_shape.data(), m_hidden_shape.size()
        );
        m_binding.BindInput("hidden_in", h_tensor);

        // ── Bind outputs ─────────────────────────────────────────────────────
        // Output "output": ORT writes directly into m_output — no allocation.
        Ort::Value out_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_output.data(), m_output.size(),
            m_output_shape.data(), m_output_shape.size()
        );
        m_binding.BindOutput("output", out_tensor);

        // Output "hidden_out": bound to the pong buffer; swapped each call.
        const int pong = 1 - m_hidden_ping;
        Ort::Value h_out_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_hidden[pong].data(), hidden_elems,
            m_hidden_shape.data(), m_hidden_shape.size()
        );
        m_binding.BindOutput("hidden_out", h_out_tensor);
    }

    // -------------------------------------------------------------------------
    // run()  —  call from the audio callback, hot path, zero allocations
    //
    //   audio_in   : pointer to `buffer_size` raw input samples (from Oboe)
    //   audio_out  : pointer to `buffer_size` floats — filled with filtered output
    // -------------------------------------------------------------------------
    bool run(const float* audio_in, float* audio_out)
    {
        static const size_t B            = static_cast<size_t>(m_gru.buffer_size());
        static const size_t hidden_elems = m_hidden[0].size();

        // 1. Fill audio channel of x with new input values
        for (size_t i = 0; i < B; ++i)
            m_x_data[i * 2] = audio_in[i];

        // 2. Swap hidden ping-pong:
        //    hidden_in  ← current ping buffer (carries state from last call)
        //    hidden_out ← pong buffer (ORT will overwrite it)
        const int pong = 1 - m_hidden_ping;

        Ort::Value h_in_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_hidden[m_hidden_ping].data(), hidden_elems,
            m_hidden_shape.data(), m_hidden_shape.size()
        );
        m_binding.BindInput("hidden_in", h_in_tensor);

        Ort::Value h_out_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_hidden[pong].data(), hidden_elems,
            m_hidden_shape.data(), m_hidden_shape.size()
        );
        m_binding.BindOutput("hidden_out", h_out_tensor);

        // 3. Run — ORT writes output directly into m_output and m_hidden[pong]
        m_session.Run(Ort::RunOptions{nullptr}, m_binding);

        // 4. Advance ping-pong
        m_hidden_ping = pong;

        // 5. Copy filtered audio to caller's output buffer
        std::memcpy(audio_out, m_output.data(), B * sizeof(float));

        return true;
    }

    void set_normed_fc(const float nfc) { 
        m_fc_normed = nfc; 
        const auto B = m_gru.buffer_size();
        for (size_t i = 0; i < static_cast<size_t>(B); ++i) {
            m_x_data[i * 2 + 1] = nfc;  // fc    — constant
        }
    }
    // Convenience accessors (e.g. for recording)
    const float* output_ptr()  const { return m_output.data(); }
    size_t        output_size() const { return m_output.size(); }

private:
    Ort::Session&              m_session;
    IRRGRU                     m_gru;
    float                      m_fc_normed;

    Ort::MemoryInfo            m_memory_info;
    Ort::IoBinding             m_binding;

    // Owned tensor data — never reallocated after construction
    std::vector<float>         m_x_data;
    std::vector<float>         m_hidden[2];   // double buffer for hidden state
    int                        m_hidden_ping; // which buffer is currently hidden_in
    std::vector<float>         m_output;

    // Shapes (stored as members so tensor creation in run() has stable pointers)
    std::array<int64_t, 3>     m_x_shape;
    std::array<int64_t, 3>     m_hidden_shape;
    std::array<int64_t, 3>     m_output_shape;
};