package ai.neral.offline

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.tensai.llamakt.ChatMessage
import com.tensai.llamakt.LlamaEngine
import com.tensai.llamakt.SamplingParams
import com.tensai.llamakt.TokenCallback
import java.io.File
import java.util.concurrent.Executors

/**
 * Real, usable Android View chat. It does not rely on Compose.
 * Native llama.cpp is only loaded after the user requests model initialization.
 */
class NativeChatActivity : Activity() {
    private val ui = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private var engine: LlamaEngine? = null
    private var busy = false
    private var generating = false
    private val turns = mutableListOf<ChatTurn>()
    private lateinit var store: ChatStore
    private lateinit var model: File
    private lateinit var status: TextView
    private lateinit var startButton: Button
    private lateinit var sendButton: Button
    private lateinit var cancelButton: Button
    private lateinit var input: EditText
    private lateinit var messages: LinearLayout
    private lateinit var scrolling: ScrollView

    private fun postUpdate(block: () -> Unit) {
        ui.post { if (!isFinishing && !isDestroyed) block() }
    }

    private fun text(value: String, size: Float = 14f, color: Int = Color.WHITE): TextView =
        TextView(this).apply {
            this.text = value
            textSize = size
            setTextColor(color)
            setPadding(8, 8, 8, 8)
        }

    private fun button(label: String, click: () -> Unit): Button = Button(this).apply {
        text = label
        setOnClickListener { click() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ChatStore(filesDir)
        model = File(filesDir, "models/${OfflineModel.FILE_NAME}")
        turns.addAll(store.load())
        val background = Color.rgb(16, 18, 23)
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(12, 12, 12, 12)
            setBackgroundColor(background)
        }
        setContentView(column)

        column.addView(text("NERAL Offline · Einfacher Modus", 21f))
        status = text("Kostenlos · CPU · nur lokal. KI einmalig herunterladen.", 12f, Color.rgb(156, 201, 242))
        column.addView(status)
        startButton = button("KI-Modell laden / starten") { startModel() }
        column.addView(startButton)

        scrolling = ScrollView(this)
        messages = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        scrolling.addView(messages)
        column.addView(scrolling, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        actions.addView(button("Neuer Chat") {
            if (!busy) {
                turns.clear()
                store.clear()
                messages.removeAllViews()
            }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        actions.addView(button("Antwort kopieren") {
            val reply = turns.lastOrNull { it.role == "assistant" }?.content.orEmpty()
            if (reply.isNotBlank()) {
                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                clipboard.setPrimaryClip(ClipData.newPlainText("NERAL Offline Antwort", reply))
                status.text = "Antwort kopiert"
            }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        column.addView(actions)

        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.BOTTOM }
        input = EditText(this).apply {
            hint = "Nachricht an NERAL"
            setTextColor(Color.WHITE)
            setHintTextColor(Color.LTGRAY)
            maxLines = 4
        }
        row.addView(input, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        sendButton = button("Senden") { sendMessage() }
        row.addView(sendButton)
        column.addView(row)

        cancelButton = button("Antwort abbrechen") { engine?.interrupt() }
        column.addView(cancelButton)
        turns.forEach { bubble(it) }
        updateControls()
    }

    private fun bubble(turn: ChatTurn): TextView {
        val view = text(
            (if (turn.role == "user") "Du" else "NERAL · lokal") + "\n" + turn.content,
            15f
        )
        view.setBackgroundColor(
            if (turn.role == "user") Color.rgb(37, 58, 78) else Color.rgb(32, 36, 44)
        )
        val margin = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = 8 }
        messages.addView(view, margin)
        scrolling.post { scrolling.fullScroll(android.view.View.FOCUS_DOWN) }
        return view
    }

    private fun updateControls() {
        startButton.isEnabled = !busy && engine == null
        sendButton.isEnabled = !busy && engine != null
        cancelButton.isEnabled = generating
    }

    private fun startModel() {
        if (busy || engine != null) return
        busy = true
        status.text = "Pruefe oder lade Modell …"
        updateControls()
        worker.execute {
            var created: LlamaEngine? = null
            try {
                if (!OfflineModel.isVerified(model)) {
                    ModelDownloader.download(model) { current, length ->
                        if (length > 0) {
                            val percentage = current * 100 / length
                            postUpdate { status.text = "Download: $percentage %" }
                        }
                    }
                }
                if (!OfflineModel.isVerified(model)) error("Modellpruefsumme ungueltig")
                postUpdate { status.text = "Modell wird lokal gestartet …" }
                created = LlamaEngine()
                created.load(path = model.absolutePath, nGpuLayers = 0, nCtx = 2048)
                val ready = created
                postUpdate {
                    engine = ready
                    busy = false
                    status.text = "Offline bereit · Qwen3 0,6B"
                    updateControls()
                }
            } catch (failure: Throwable) {
                CrashLog.save(filesDir, failure)
                try { created?.free() } catch (_: Throwable) { }
                postUpdate {
                    busy = false
                    status.text = "Modellstart fehlgeschlagen: ${failure.message ?: failure.javaClass.simpleName}. Diagnose in Startansicht kopieren."
                    updateControls()
                }
            }
        }
    }

    private fun sendMessage() {
        val native = engine ?: return
        val question = input.text.toString().trim().take(1200)
        if (question.isEmpty() || busy) return
        input.setText("")
        busy = true
        generating = true
        turns.add(ChatTurn("user", question))
        bubble(turns.last())
        val promptHistory = turns.toList()
        turns.add(ChatTurn("assistant", ""))
        val assistantIndex = turns.lastIndex
        val assistantView = bubble(turns.last())
        try { store.save(turns.filter { it.content.isNotBlank() }) }
        catch (_: Exception) { status.text = "Verlauf konnte nicht gespeichert werden" }
        status.text = "Antwort wird auf dem Handy berechnet …"
        updateControls()

        worker.execute {
            try {
                val finished = OfflineGenerator.generate(
                    native, promptHistory,
                    onText = { reply ->
                        postUpdate {
                            turns[assistantIndex] = ChatTurn("assistant", reply)
                            assistantView.text = "NERAL · lokal\n$reply"
                        }
                    },
                    onRetry = {
                        postUpdate { status.text = "Wiederholte Antwort erkannt – neuer Versuch …" }
                    }
                )
                postUpdate {
                    turns[assistantIndex] = ChatTurn("assistant", finished)
                    assistantView.text = "NERAL · lokal\n$finished"
                    try { store.save(turns.toList()) } catch (_: Exception) { }
                    busy = false
                    generating = false
                    status.text = "Offline bereit"
                    updateControls()
                }
            } catch (failure: Throwable) {
                CrashLog.save(filesDir, failure)
                postUpdate {
                    turns[assistantIndex] = ChatTurn("assistant", "Fehler: ${failure.message ?: failure.javaClass.simpleName}")
                    assistantView.text = turns[assistantIndex].content
                    try { store.save(turns.toList()) } catch (_: Exception) { }
                    busy = false
                    generating = false
                    status.text = "Fehler protokolliert. Diagnose in Startansicht kopieren."
                    updateControls()
                }
            }
        }
    }

    override fun onDestroy() {
        engine?.interrupt()
        worker.execute { try { engine?.free() } catch (_: Throwable) { } }
        worker.shutdown()
        super.onDestroy()
    }
}
