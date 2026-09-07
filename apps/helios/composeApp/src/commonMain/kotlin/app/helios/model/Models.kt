package app.helios.model

enum class HeliosSection(val label: String) {
    HOME("Home"),
    CHAT("Chat"),
    SOLVE("Solve"),
    PROJECTS("Projects"),
    LIBRARY("Library"),
    SETTINGS("Settings")
}

enum class AssistantMode(val label: String, val shortLabel: String) {
    CHAT("Chat", "Chat"),
    SOLVE("Solve", "Solve"),
    VERIFY("Verify", "Verify"),
    BUILD("Build", "Build")
}

data class ChatMessage(
    val id: String,
    val role: Role,
    val text: String,
    val meta: String? = null
) {
    enum class Role { USER, ASSISTANT }
}

data class Conversation(
    val id: String,
    val title: String,
    val preview: String,
    val active: Boolean = false
)

data class ProjectCard(
    val id: String,
    val title: String,
    val subtitle: String,
    val progress: String
)

object DemoContent {
    val conversations = listOf(
        Conversation("1", "Neral 50M Roadmap", "I1 paused after B3a checkpoint", true),
        Conversation("2", "BER network ideas", "Condor focus city concept"),
        Conversation("3", "Physics homework", "Circular motion and gravity"),
        Conversation("4", "Helios app", "Adaptive Android + desktop design")
    )

    val messages = listOf(
        ChatMessage("m1", ChatMessage.Role.USER, "Wie weit ist Neral aktuell?"),
        ChatMessage(
            "m2",
            ChatMessage.Role.ASSISTANT,
            "Der aktuelle 50M-Checkpoint ist lokal verfügbar. I1 ist pausiert und kann aus dem letzten B3a-Checkpoint fortgesetzt werden.",
            "Neral 50M · local"
        )
    )

    val projects = listOf(
        ProjectCard("p1", "Neral", "Model training, benchmarks and releases", "I1 · paused"),
        ProjectCard("p2", "Helios", "Cross-platform assistant app", "v0.1 · building"),
        ProjectCard("p3", "BahnConnections", "Rail connection explorer", "prototype")
    )
}
