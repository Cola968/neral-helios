package ai.neral.offline

import org.junit.Assert.*
import org.junit.Test

class ReplySafetyTest {
    @Test fun `repeated greeting on unrelated capital question is rejected`() {
        val history = listOf(
            ChatTurn("user", "welches Modell bist du"),
            ChatTurn("assistant", "Hallo! Ich freue mich auf dich! Welches Modell bist du? 😊"),
            ChatTurn("user", "Hauptstadt von Deutschland?")
        )
        assertTrue(ReplySafety.needsRetry(history, "Hallo! Ich freue mich auf dich! Welches Modell bist du? 😊"))
    }

    @Test fun `question echo is not accepted as useful reply`() {
        val history = listOf(ChatTurn("user", "welches Modell bist du"))
        assertTrue(ReplySafety.needsRetry(history, "Hallo! Welches Modell bist du?"))
    }

    @Test fun `valid concise answer and legitimate same answer are kept`() {
        val history = listOf(ChatTurn("user", "Hauptstadt von Deutschland?"))
        assertFalse(ReplySafety.needsRetry(history, "Berlin."))
        val other = listOf(
            ChatTurn("user", "Nenne einen Kontinent"),
            ChatTurn("assistant", "Europa"),
            ChatTurn("user", "Zu welchem Kontinent gehört Deutschland?")
        )
        assertFalse(ReplySafety.needsRetry(other, "Europa"))
    }

    @Test fun `isolated retry keeps ONLY current question and a system instruction`() {
        val history = listOf(
            ChatTurn("user", "welches Modell bist du"),
            ChatTurn("assistant", "Hallo! Ich freue mich auf dich! Welches Modell bist du? 😊"),
            ChatTurn("user", "Hauptstadt von Deutschland?")
        )
        val isolated = ReplySafety.isolatedMessages(history)
        assertEquals(listOf("system", "user"), isolated.map { it.role })
        assertEquals("Hauptstadt von Deutschland?", isolated.last().content)
        assertFalse(isolated.any { it.content.contains("Hallo! Ich freue") })
    }

    @Test fun `blank completion and model echo trigger retry`() {
        val history = listOf(ChatTurn("user", "Was ist 2+2?"))
        assertTrue(ReplySafety.needsRetry(history, "  "))
        assertTrue(ReplySafety.needsRetry(history, "Was ist 2+2?"))
    }
}
