package ai.neral.offline

import java.io.File
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.util.Base64

/** Private on-device history; no account, internet, or telemetry. */
class ChatStore(directory: File) {
    val file = File(directory, "neral_offline_history_v1.txt")

    fun load(): List<ChatTurn> {
        if (!file.isFile || file.length() > 700_000) return emptyList()
        return try {
            val lines = file.readLines(Charsets.UTF_8)
            if (lines.firstOrNull() != "NERAL_OFFLINE_V1") return emptyList()
            lines.drop(1).takeLast(100).mapNotNull { line ->
                val parts = line.split('\t', limit = 2)
                if (parts.size != 2 || parts[0] !in listOf("user", "assistant")) return@mapNotNull null
                val bytes = Base64.getDecoder().decode(parts[1])
                if (bytes.size > 16_000) return@mapNotNull null
                ChatTurn(parts[0], String(bytes, StandardCharsets.UTF_8).take(4000))
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun save(turns: List<ChatTurn>) {
        file.parentFile?.mkdirs()
        val tmp = File(file.parentFile, file.name + ".tmp")
        val data = buildString {
            append("NERAL_OFFLINE_V1\n")
            turns.takeLast(100).forEach { turn ->
                if (turn.role == "user" || turn.role == "assistant") {
                    append(turn.role)
                    append('\t')
                    append(Base64.getEncoder().encodeToString(turn.content.take(4000).toByteArray(Charsets.UTF_8)))
                    append('\n')
                }
            }
        }
        try {
            FileOutputStream(tmp).use { output ->
                output.write(data.toByteArray(Charsets.UTF_8))
                output.fd.sync()
            }
            if (!tmp.renameTo(file)) error("Lokaler Chatverlauf konnte nicht gespeichert werden")
        } finally {
            if (tmp.exists()) tmp.delete()
        }
    }

    fun clear() {
        if (file.exists() && !file.delete()) error("Lokaler Chatverlauf konnte nicht geloescht werden")
    }
}
