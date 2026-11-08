package ai.neral.offline

import java.util.Locale

/**
 * Detects the concrete stale-answer/echo failure observed on Android.
 * Does not decide factual correctness; a real device smoke test is still needed.
 */
object ReplySafety {
    private fun normalized(text: String) =
        text.trim().lowercase(Locale.ROOT).replace(Regex("\\s+"), " ")

    fun needsRetry(history: List<ChatTurn>, reply: String): Boolean {
        val answer = normalized(reply)
        if (answer.isBlank()) return true
        val latestQuestion = history.lastOrNull { it.role == "user" }?.content?.let(::normalized)
            ?: return false
        val earlierAssistant = history.dropLast(1).lastOrNull { it.role == "assistant" }?.content?.let(::normalized)
        val earlierQuestion = history.dropLast(1).lastOrNull { it.role == "user" }?.content?.let(::normalized)
        // An identical long answer to unrelated questions is a stale-generation symptom.
        if (earlierAssistant != null && earlierQuestion != null &&
            latestQuestion != earlierQuestion && answer.length >= 25 && answer == earlierAssistant) return true
        // Short greeting + verbatim re-asking the user is not an answer.
        if (latestQuestion.length >= 8 && answer.length <= latestQuestion.length + 85 &&
            answer.contains(latestQuestion) &&
            (answer.startsWith("hallo") || answer == latestQuestion)) return true
        return false
    }

    fun isolatedMessages(history: List<ChatTurn>): List<ChatTurn> {
        val lastUser = history.lastOrNull { it.role == "user" }
            ?: error("Keine Nutzerfrage fuer einen frischen Versuch")
        return listOf(
            ChatTurn("system", "Du bist NERAL Offline (Qwen3-0.6B). Antworte auf die letzte Nutzerfrage direkt und konkret auf Deutsch. " +
                "Keine Begruessung, keine Frage einfach wiederholen, keine erfundenen Quellen. " +
                "Wenn du etwas nicht weisst, sage es."),
            ChatTurn("user", lastUser.content)
        )
    }
}
