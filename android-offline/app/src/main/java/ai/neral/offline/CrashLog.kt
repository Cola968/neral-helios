package ai.neral.offline

import java.io.File
import java.io.PrintWriter
import java.io.StringWriter

/** Private, bounded crash report. Never uploaded or sent over the network. */
object CrashLog {
    fun file(directory: File): File = File(directory, "neral_android_last_crash.txt")

    fun save(directory: File, cause: Throwable) {
        try {
            val writer = StringWriter()
            cause.printStackTrace(PrintWriter(writer))
            file(directory).writeText(writer.toString().take(12000))
        } catch (_: Throwable) {
            // Never crash while recording an already-failing process.
        }
    }

    fun read(directory: File): String {
        return try {
            val log = file(directory)
            if (log.isFile) log.bufferedReader().use { it.readText().take(12000) } else ""
        } catch (_: Throwable) { "" }
    }

    fun clear(directory: File) {
        try { file(directory).delete() } catch (_: Throwable) { }
    }
}
