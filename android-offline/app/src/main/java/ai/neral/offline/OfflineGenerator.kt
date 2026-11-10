package ai.neral.offline

import com.tensai.llamakt.ChatMessage
import com.tensai.llamakt.LlamaEngine
import com.tensai.llamakt.SamplingParams
import com.tensai.llamakt.TokenCallback

/**
 * Shared code path used by both the Compose UI and the simpler Android-Views UI.
 * Refuses stale answer reuse; second attempt drops earlier assistant/user context.
 */
object OfflineGenerator {
    private const val TOKEN_LIMIT = 384

    fun generate(
        engine: LlamaEngine,
        history: List<ChatTurn>,
        onText: (String) -> Unit,
        onRetry: () -> Unit = {},
    ): String {
        val current = history.lastOrNull { it.role == "user" }?.content?.trim().orEmpty()
        require(current.isNotEmpty()) { "Die letzte Nutzerfrage fehlt" }

        fun attempt(messages: List<ChatTurn>, temperature: Float): String {
            val prompt = engine.formatChat(
                messages.map { ChatMessage(it.role, it.content) }, enableThinking = false
            )
            // Detect broken templates instead of silently generating from an old question.
            check(prompt.isNotBlank() && prompt.contains(current)) {
                "Die aktuelle Frage fehlt im Prompt der lokalen KI"
            }
            val output = StringBuilder()
            engine.resetInterrupt()
            val count = engine.completion(
                prompt,
                SamplingParams(nPredict = TOKEN_LIMIT, temperature = temperature),
                TokenCallback { token ->
                    output.append(token)
                    onText(output.toString())
                }
            )
            if (count < 0) error("Lokale Modellberechnung fehlgeschlagen")
            val response = output.toString().trim()
            if (count >= TOKEN_LIMIT) return response + "\n\n[Antwort durch Tokenlimit abgeschnitten]"
            return response
        }

        val first = attempt(OfflineModel.window(history), 0.5f)
        if (!ReplySafety.needsRetry(history, first)) return first
        onRetry()
        onText("")
        val second = attempt(ReplySafety.isolatedMessages(history), 0.35f)
        // Repeated broken output should never be displayed as a confident answer.
        if (ReplySafety.needsRetry(history, second)) {
            return "Die lokale KI hat die Frage wiederholt oder eine alte Antwort geliefert. " +
                "Bitte versuche einen neuen Chat. Das kleine Qwen3-0.6B ist noch nicht zuverlaessig genug."
        }
        return second
    }
}
