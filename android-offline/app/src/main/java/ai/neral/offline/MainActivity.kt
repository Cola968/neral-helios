package ai.neral.offline

import android.os.Bundle
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.tensai.llamakt.ChatMessage
import com.tensai.llamakt.LlamaEngine
import com.tensai.llamakt.SamplingParams
import com.tensai.llamakt.TokenCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { OfflineScreen() }
    }
}

private val Background = Color(0xFF101217)
private val Surface = Color(0xFF20242C)
private val Accent = Color(0xFF9CC9F2)

@Composable
private fun OfflineScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val handler = remember { Handler(Looper.getMainLooper()) }
    val model = remember { File(context.filesDir, "models/${OfflineModel.FILE_NAME}") }
    val store = remember { ChatStore(context.filesDir) }
    val turns = remember { mutableStateListOf<ChatTurn>().apply { addAll(store.load()) } }
    val listState = rememberLazyListState()
    var installed by remember { mutableStateOf(OfflineModel.isReady(model)) }
    var engine by remember { mutableStateOf<LlamaEngine?>(null) }
    var input by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var generating by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf(if(installed) "Modell bereit zum Laden" else "Einmaliger Modelldownload erforderlich") }

    val initialize: (Boolean) -> Unit = { download ->
        if (!busy && engine == null) {
            busy = true
            status = if (download) "Lade Qwen3 0,6B herunter …" else "Initialisiere das Modell …"
            scope.launch {
                try {
                    val loaded = withContext(Dispatchers.IO) {
                        if (download) ModelDownloader.download(model) { read, total ->
                            if (total > 0) {
                                val percent = ((read * 100) / total).coerceIn(0, 100)
                                handler.post { status = "Download: $percent %" }
                            }
                        }
                        handler.post { status = "Pruefe Modellintegritaet …" }
                        if (!OfflineModel.isVerified(model)) error("Modell unvollstaendig oder Pruefsumme falsch; bitte erneut laden")
                        handler.post { status = "Lade Modell in den Handyspeicher …" }
                        LlamaEngine().apply { load(path = model.absolutePath, nGpuLayers = 0, nCtx = 2048) }
                    }
                    installed = true
                    engine = loaded
                    status = "Offline bereit · CPU · Qwen3 0,6B"
                } catch (e: Exception) {
                    status = "Fehler: ${e.message ?: "Modell konnte nicht gestartet werden"}"
                    installed = OfflineModel.isReady(model)
                } finally { busy = false }
            }
        }
    }

    LaunchedEffect(Unit) {
        if (installed) initialize(false)
    }
    LaunchedEffect(turns.size) {
        if (turns.isNotEmpty()) listState.animateScrollToItem(turns.lastIndex)
    }

    MaterialTheme(colorScheme = darkColorScheme(primary = Accent, background = Background, surface = Surface)) {
        Column(
            Modifier.fillMaxSize().background(Background)
                .windowInsetsPadding(WindowInsets.safeDrawing).imePadding().padding(14.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Text("NERAL", fontSize = 25.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
                Spacer(Modifier.weight(1f))
                TextButton(
                    onClick = {
                        store.clear()
                        turns.clear()
                        input = ""
                        status = if (engine != null) "Neuer Chat · offline bereit" else "Neuer Chat"
                    },
                    enabled = !busy && turns.isNotEmpty()
                ) { Text("Neuer Chat") }
            }
            Text("Offline · kostenlos · ohne PC · Verlauf nur auf diesem Handy", fontSize = 12.sp, color = Accent)
            Spacer(Modifier.height(8.dp))
            Text(status, fontSize = 12.sp, color = Color(0xFFB0B9C8))
            Spacer(Modifier.height(12.dp))

            if (engine == null) {
                Button(onClick = { initialize(!installed) }, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                    Text(if (busy) "Wird vorbereitet …" else if (installed) "Lokales Modell starten" else "Modell laden · ca. 640 MB")
                }
                Text(
                    "Einmal mit Internet laden, danach ohne Internet und ohne eingeschalteten PC chatten. " +
                    "Das Modell ist Qwen3 0,6B, kein trainierter NERAL-LoRA-Adapter.",
                    fontSize = 12.sp, color = Color(0xFFB0B9C8)
                )
            }

            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (turns.isEmpty()) item {
                    Text(
                        "Dein lokaler Chat. Keine Cloud-Inferenz, keine Anmeldung.",
                        color = Color(0xFF9DA9BB), modifier = Modifier.padding(vertical = 28.dp)
                    )
                }
                itemsIndexed(turns) { _, message ->
                    Column(
                        Modifier.fillMaxWidth().background(
                            if (message.role == "user") Color(0xFF253A4E) else Surface,
                            RoundedCornerShape(14.dp)
                        ).padding(12.dp)
                    ) {
                        Text(
                            if (message.role == "user") "Du" else "NERAL · Qwen3 0,6B",
                            fontSize = 11.sp, color = Accent, fontWeight = FontWeight.Bold
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(message.content.ifEmpty { "…" }, color = Color.White, fontSize = 15.sp)
                        if (message.role == "assistant" && message.content.isNotBlank()) {
                            TextButton(onClick = {
                                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                clipboard.setPrimaryClip(ClipData.newPlainText("NERAL-Antwort", message.content))
                                Toast.makeText(context, "Antwort kopiert", Toast.LENGTH_SHORT).show()
                            }) { Text("Kopieren", fontSize = 12.sp) }
                        }
                    }
                }
            }

            if (generating) {
                TextButton(onClick = { engine?.interrupt(); status = "Abbruch angefordert …" }) {
                    Text("Antwort abbrechen")
                }
            }

            Row(verticalAlignment = Alignment.Bottom, modifier = Modifier.fillMaxWidth()) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it.take(1200) },
                    placeholder = { Text("Nachricht …") },
                    modifier = Modifier.weight(1f),
                    enabled = engine != null && !busy,
                    maxLines = 5
                )
                Spacer(Modifier.width(8.dp))
                Button(
                    enabled = engine != null && !busy && input.isNotBlank(),
                    onClick = {
                        val localEngine = engine ?: return@Button
                        val question = input.trim()
                        input = ""
                        turns.add(ChatTurn("user", question))
                        val previous = turns.toList()
                        turns.add(ChatTurn("assistant", ""))
                        try { store.save(turns.filter { it.content.isNotBlank() }) }
                        catch (_: Exception) { status = "Verlauf konnte nicht gespeichert werden" }
                        val responseIndex = turns.lastIndex
                        generating = true
                        busy = true
                        status = "Antwort wird lokal berechnet …"
                        scope.launch(Dispatchers.IO) {
                            val answer = StringBuilder()
                            try {
                                val conversation = OfflineModel.window(previous).map {
                                    ChatMessage(it.role, it.content)
                                }
                                localEngine.resetInterrupt()
                                val prompt = localEngine.formatChat(conversation, enableThinking = false)
                                val cap = 384
                                val count = localEngine.completion(
                                    prompt,
                                    SamplingParams(nPredict = cap, temperature = 0.65f),
                                    TokenCallback { token ->
                                        answer.append(token)
                                        val current = answer.toString()
                                        handler.post {
                                            if (responseIndex < turns.size)
                                                turns[responseIndex] = ChatTurn("assistant", current)
                                        }
                                    }
                                )
                                val result = when {
                                    count < 0 -> "Fehler bei der lokalen Modellberechnung."
                                    count == 0 && answer.isEmpty() -> "Keine Antwort erzeugt."
                                    count >= cap -> answer.toString() + "\n\n[Antwort durch Tokenlimit abgeschnitten]"
                                    else -> answer.toString()
                                }
                                handler.post {
                                    if (responseIndex < turns.size)
                                        turns[responseIndex] = ChatTurn("assistant", result)
                                    status = try {
                                        store.save(turns.toList())
                                        "Offline bereit · CPU · Qwen3 0,6B"
                                    } catch (_: Exception) { "Antwort da, Verlauf konnte nicht gespeichert werden" }
                                }
                            } catch (e: Exception) {
                                handler.post {
                                    if (responseIndex < turns.size)
                                        turns[responseIndex] = ChatTurn("assistant", "Fehler: ${e.message}")
                                    try { store.save(turns.toList()) } catch (_: Exception) { }
                                    status = "Modellfehler · erneut versuchen"
                                }
                            } finally {
                                handler.post { generating = false; busy = false }
                            }
                        }
                    }
                ) { Text("Senden") }
            }
        }
    }
}
