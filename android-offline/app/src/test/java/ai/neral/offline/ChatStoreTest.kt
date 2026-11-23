package ai.neral.offline

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class ChatStoreTest {
    @Test fun `new store contains no chats`() {
        val directory = Files.createTempDirectory("neral-chat-empty").toFile()
        try {
            assertTrue(ChatStore(directory).load().isEmpty())
        } finally { directory.deleteRecursively() }
    }

    @Test fun `conversation survives reopening and preserves unicode and multiline text`() {
        val directory = Files.createTempDirectory("neral-chat-roundtrip").toFile()
        try {
            val expected = listOf(ChatTurn("user", "Wie geht's? ✨\nHallo"), ChatTurn("assistant", "Gut.\nErgebnis: 42"))
            ChatStore(directory).save(expected)
            assertEquals(expected, ChatStore(directory).load())
        } finally { directory.deleteRecursively() }
    }

    @Test fun `clear deletes saved conversation immediately`() {
        val directory = Files.createTempDirectory("neral-chat-clear").toFile()
        try {
            val store = ChatStore(directory)
            store.save(listOf(ChatTurn("user", "privat")))
            store.clear()
            assertTrue(ChatStore(directory).load().isEmpty())
        } finally { directory.deleteRecursively() }
    }

    @Test fun `oversized and corrupted history does not crash or exhaust memory`() {
        val directory = Files.createTempDirectory("neral-chat-bad").toFile()
        try {
            val store = ChatStore(directory)
            store.save((1..300).map { ChatTurn("user", "x".repeat(20_000)) })
            val read = store.load()
            assertTrue(read.size <= 100)
            assertTrue(read.all { it.content.length <= 4000 })
            store.file.writeText("bad\tbase64\n")
            assertTrue(store.load().isEmpty())
        } finally { directory.deleteRecursively() }
    }
}
