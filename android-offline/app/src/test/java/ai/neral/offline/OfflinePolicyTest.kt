package ai.neral.offline

import org.junit.Assert.*
import org.junit.Test

class OfflinePolicyTest {
    @Test fun `model URL is official HTTPS and no cloud inference host`() {
        assertEquals("https", java.net.URI(OfflineModel.URL).scheme)
        assertEquals("huggingface.co", java.net.URI(OfflineModel.URL).host)
        assertTrue(OfflineModel.URL.contains("/Qwen/Qwen3-0.6B-GGUF/"))
    }

    @Test fun `model header must be GGUF`() {
        assertTrue(OfflineModel.isGguf(byteArrayOf(0x47, 0x47, 0x55, 0x46)))
        assertFalse(OfflineModel.isGguf(byteArrayOf(0x50, 0x4b, 0x03, 0x04)))
        assertFalse(OfflineModel.isGguf(byteArrayOf(0x47)))
    }

    @Test fun `invalid and partial downloads are rejected`() {
        assertFalse(OfflineModel.isPlausibleSize(0))
        assertFalse(OfflineModel.isPlausibleSize(32_000_000))
        assertTrue(OfflineModel.isPlausibleSize(639_000_000))
        assertFalse(OfflineModel.isPlausibleSize(3_000_000_000))
    }

    @Test fun `official model checksum is pinned and corrupt GGUF is rejected`() {
        assertEquals(64, OfflineModel.EXPECTED_SHA256.length)
        val f = java.io.File.createTempFile("neral-corrupt-", ".gguf")
        try {
            f.writeBytes(byteArrayOf(0x47, 0x47, 0x55, 0x46, 0x00))
            assertFalse(OfflineModel.isVerified(f))
        } finally { f.delete() }
    }

    @Test fun `conversation window keeps latest exchanges and bounds input`() {
        val turns=(1..25).map { ChatTurn(if(it%2==0) "assistant" else "user", "x".repeat(1000)+it) }
        val clipped=OfflineModel.window(turns)
        assertTrue(clipped.size<=7)
        assertEquals("system", clipped.first().role)
        assertEquals("user", clipped.last().role)
        assertTrue(clipped.drop(1).all{ it.content.length<=600 })
        assertTrue(clipped.last().content.endsWith("25"))
    }
}
