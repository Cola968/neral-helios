package ai.neral.offline

import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/** Only invoked after the user taps Download. No inference traffic leaves Android. */
object ModelDownloader {
    fun download(destination: File, progress: (Long, Long) -> Unit): File {
        destination.parentFile?.mkdirs()
        val partial = File(destination.parentFile, destination.name + ".part")
        partial.delete()
        val connection = (URL(OfflineModel.URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 45_000
            instanceFollowRedirects = true
            requestMethod = "GET"
            setRequestProperty("User-Agent", "NERAL-Offline-Android/0.1")
        }
        try {
            if (connection.responseCode != 200) {
                error("Modell-Download fehlgeschlagen: HTTP ${connection.responseCode}")
            }
            val expected = connection.contentLengthLong
            if (expected > 0 && !OfflineModel.isPlausibleSize(expected)) {
                error("Unerwartete Modellgroesse: $expected Bytes")
            }
            var received = 0L
            val buffer = ByteArray(64 * 1024)
            connection.inputStream.use { input ->
                FileOutputStream(partial).use { out ->
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        out.write(buffer, 0, n)
                        received += n
                        if (received > OfflineModel.MAX_BYTES) error("Modell zu gross")
                        progress(received, expected)
                    }
                    out.fd.sync()
                }
            }
            if (expected > 0 && received != expected) error("Download unvollstaendig")
            if (!OfflineModel.isReady(partial)) error("Ungueltiger GGUF-Download")
            // Only a fully downloaded GGUF replaces an existing file.
            if (destination.exists() && !destination.delete()) error("Zieldatei blockiert")
            if (!partial.renameTo(destination)) error("Modell konnte nicht gespeichert werden")
            return destination
        } catch (ex: Exception) {
            partial.delete()
            throw ex
        } finally {
            connection.disconnect()
        }
    }
}
