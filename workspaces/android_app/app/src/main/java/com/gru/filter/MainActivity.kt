package com.gru.filter

/** Simple UI: permission, start/stop realtime filter, seek-bar cutoff. */
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView
    private lateinit var statsText: TextView
    private lateinit var fcValueText: TextView
    private lateinit var fcSeekBar: SeekBar
    private lateinit var controlButton: Button

    private var started = false

    companion object {
        private const val FC_MIN_HZ = 200
        private const val FC_MAX_HZ = 8000
    }

    private fun hzFromSeekProgress(progress: Int): Float =
        (FC_MIN_HZ + progress).toFloat()

    private fun updateFcLabel(hz: Float) {
        fcValueText.text = "fc: ${hz.toInt()} Hz"
    }
    private val uiHandler = Handler(Looper.getMainLooper())
    private val statsUpdater = object : Runnable {
        override fun run() {
            if (started) {
                statsText.text = "Stats: ${FilterEngine.nativeGetStats()}"
                uiHandler.postDelayed(this, 1000)
            }
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            statusText.text = "Microphone permission granted."
        } else {
            statusText.text = "Microphone permission denied."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        statsText = findViewById(R.id.statsText)
        fcValueText = findViewById(R.id.fcValueText)
        fcSeekBar = findViewById(R.id.fcSeekBar)
        controlButton = findViewById(R.id.controlButton)

        fcSeekBar.max = FC_MAX_HZ - FC_MIN_HZ
        fcSeekBar.progress = 1000 - FC_MIN_HZ
        updateFcLabel(hzFromSeekProgress(fcSeekBar.progress))

        fcSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val hz = hzFromSeekProgress(progress)
                updateFcLabel(hz)
                FilterEngine.nativeSetFc(hz)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        ensureMicPermission()
        initializeEngine()

        controlButton.setOnClickListener {
            if (started) {
                stopEngine()
            } else {
                startEngine()
            }
        }
    }

    override fun onStop() {
        super.onStop()
        if (started) {
            stopEngine()
        }
    }

    private fun ensureMicPermission() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun initializeEngine() {
        val modelFile = copyModelFromAssets("lowpass_rnn.onnx")
        if (modelFile == null) {
            statusText.text = "Model asset lowpass_rnn.onnx not found. Add it to assets."
            controlButton.isEnabled = false
            return
        }

        val ok = FilterEngine.nativeInit(
            modelFile.absolutePath,
            hzFromSeekProgress(fcSeekBar.progress),
            "cpu",
            true
        )
        if (ok) {
            statusText.text = "Engine initialized. Ready to start."
        } else {
            statusText.text = "Engine init failed."
            controlButton.isEnabled = false
        }
    }

    private fun startEngine() {
        val ok = FilterEngine.nativeStart()
        if (!ok) {
            statusText.text = "Failed to start audio engine."
            return
        }
        started = true
        controlButton.text = "Stop"
        statusText.text = "Running..."
        uiHandler.post(statsUpdater)
    }

    private fun stopEngine() {
        FilterEngine.nativeStop()
        started = false
        controlButton.text = "Start"
        statusText.text = "Stopped."
        uiHandler.removeCallbacks(statsUpdater)
        statsText.text = "Stats: ${FilterEngine.nativeGetStats()}"
    }

    private fun copyModelFromAssets(assetName: String): File? {
        return try {
            val outFile = File(filesDir, assetName)
            assets.open(assetName).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            outFile
        } catch (_: Exception) {
            null
        }
    }
}
