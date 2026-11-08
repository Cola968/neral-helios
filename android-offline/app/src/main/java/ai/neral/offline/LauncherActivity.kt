package ai.neral.offline

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast

/**
 * System-UI-only recovery entrypoint. No Compose or llama.cpp is loaded
 * until the user explicitly opens the chat.
 */
class LauncherActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(28, 36, 28, 26)
            setBackgroundColor(Color.rgb(16, 18, 23))
        }
        val layout = ScrollView(this).apply { addView(root) }
        setContentView(layout)

        fun paragraph(value: String, size: Float = 16f, textColor: Int = Color.WHITE): TextView {
            return TextView(this).apply {
                text = value
                textSize = size
                setTextColor(textColor)
                setPadding(0, 4, 0, 20)
                root.addView(this)
            }
        }

        paragraph("NERAL", 30f)
        paragraph("Offline auf Android · kostenlos · ohne PC", 14f, Color.rgb(156, 201, 242))
        paragraph("Der Chat läuft lokal. Beim ersten Start muss das Sprachmodell einmal heruntergeladen werden.", 15f)

        val report = CrashLog.read(filesDir)
        if (report.isNotBlank()) {
            paragraph("Ein Absturz wurde erkannt. Du kannst die Diagnose kopieren und damit den Fehler untersuchen lassen.", 14f, Color.rgb(255, 191, 126))
            Button(this).apply {
                text = "Fehlerdiagnose kopieren"
                setOnClickListener {
                    val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    clipboard.setPrimaryClip(ClipData.newPlainText("NERAL Android Fehlerdiagnose", report))
                    Toast.makeText(this@LauncherActivity, "Fehlerdiagnose kopiert", Toast.LENGTH_SHORT).show()
                }
                root.addView(this)
            }
        }

        Button(this).apply {
            text = "NERAL Chat öffnen"
            setOnClickListener {
                try { startActivity(Intent(this@LauncherActivity, MainActivity::class.java)) }
                catch (error: Throwable) {
                    CrashLog.save(filesDir, error)
                    Toast.makeText(this@LauncherActivity, "Start fehlgeschlagen. App erneut öffnen und Diagnose kopieren.", Toast.LENGTH_LONG).show()
                }
            }
            root.addView(this)
        }
        paragraph("Bei einem Absturz: App erneut öffnen, Diagnose kopieren und den Text hier in ChatGPT einfügen. Es werden keine Daten automatisch übertragen.", 12f, Color.LTGRAY)
    }
}
