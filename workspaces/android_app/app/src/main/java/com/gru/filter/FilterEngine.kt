package com.gru.filter

/** JNI bridge to the native Oboe + ONNX GRU filter engine. */
object FilterEngine {
    init {
        System.loadLibrary("gru_filter")
    }

    /** Load ONNX model and wire Oboe streams; does not start audio yet. */
    external fun nativeInit(modelPath: String, fc: Float, ep: String, cpuOnly: Boolean): Boolean
    /** Start capture/playback and the inference worker thread. */
    external fun nativeStart(): Boolean
    /** Stop streams and release the native engine. */
    external fun nativeStop(): Boolean
    /** Sample rates and xrun counters for the UI. */
    external fun nativeGetStats(): String
    /** Update cutoff frequency while running (Hz). */
    external fun nativeSetFc(fc: Float)
}
