package ai.neral.offline

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class CrashLogTest {
    @Test fun `crashes are recorded locally with bounded size`() {
        val dir = Files.createTempDirectory("neral-crash").toFile()
        try {
            val failure = IllegalStateException("unavailable model " + "x".repeat(50000))
            CrashLog.save(dir, failure)
            val report = CrashLog.read(dir)
            assertTrue(report.contains("IllegalStateException"))
            assertTrue(report.contains("unavailable model"))
            assertTrue(report.length <= 12000)
            CrashLog.clear(dir)
            assertEquals("", CrashLog.read(dir))
        } finally { dir.deleteRecursively() }
    }

    @Test fun `empty and damaged log do not throw`() {
        val dir = Files.createTempDirectory("neral-crash-empty").toFile()
        try {
            assertEquals("", CrashLog.read(dir))
            CrashLog.file(dir).writeText("z".repeat(60000))
            assertTrue(CrashLog.read(dir).length <= 12000)
        } finally { dir.deleteRecursively() }
    }

    @Test fun `android manifest uses safe launcher instead of compose for first startup`() {
        val manifest = java.io.File("src/main/AndroidManifest.xml").readText()
        assertTrue(manifest.contains("android:name=\".LauncherActivity\""))
        assertTrue(manifest.contains("android:name=\".MainActivity\""))
        assertTrue(manifest.contains("android:name=\".NeralApplication\""))
        assertTrue(manifest.contains("android:name=\"libvulkan.so\""))
    }
}
