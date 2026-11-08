package ai.neral.offline

import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/** This app runs an unmodified Qwen base model. It does NOT claim NERAL LoRA weights. */
data class ChatTurn(val role: String, val content: String)

object OfflineModel {
    const val FILE_NAME = "Qwen3-0.6B-Q8_0.gguf"
    const val URL = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
    const val MIN_BYTES = 450_000_000L
    const val MAX_BYTES = 900_000_000L
    // SHA256 published by Qwen for the official Q8_0 GGUF (Hugging Face file details).
    const val EXPECTED_SHA256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"

    fun isGguf(header: ByteArray): Boolean =
        header.size >= 4 && header[0] == 0x47.toByte() && header[1] == 0x47.toByte() &&
            header[2] == 0x55.toByte() && header[3] == 0x46.toByte()

    fun isPlausibleSize(bytes: Long): Boolean = bytes in MIN_BYTES..MAX_BYTES

    fun isReady(file: File): Boolean {
        if (!file.isFile || !isPlausibleSize(file.length())) return false
        return try {
            FileInputStream(file).use { stream ->
                val header = ByteArray(4)
                stream.read(header) == 4 && isGguf(header)
            }
        } catch (_: Exception) { false }
    }

    /** Full integrity check before native code parses the downloaded GGUF. */
    fun isVerified(file: File): Boolean {
        if (!isReady(file)) return false
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    digest.update(buffer, 0, n)
                }
            }
            digest.digest().joinToString("") { "%02x".format(it) } == EXPECTED_SHA256
        } catch (_: Exception) { false }
    }

    /** Keep prompt compact enough for a low-memory CPU-only mobile context. */
    fun window(history: List<ChatTurn>): List<ChatTurn> =
        listOf(ChatTurn("system", "Du bist NERAL Offline, ein lokaler Chat-Assistent. " +
            "Du nutzt das unveraenderte Qwen3-0.6B-Modell. Antworte auf Deutsch, " +
            "nenne Unsicherheit und erfinde keine Quellen.")) +
        history.takeLast(6).map { turn ->
            ChatTurn(if (turn.role == "assistant") "assistant" else "user", turn.content.takeLast(600))
        }
}
