package app.helios.runtime

import app.helios.model.AssistantMode

enum class RuntimeHealth { OFFLINE, READY, BUSY }

data class RuntimeSnapshot(
    val health: RuntimeHealth,
    val modelName: String,
    val parameterCount: String,
    val execution: String
)

data class GenerationRequest(
    val prompt: String,
    val mode: AssistantMode
)

interface NeralClient {
    fun snapshot(): RuntimeSnapshot
    fun generate(request: GenerationRequest): String
}

/**
 * UI-only runtime used while native Neral inference is being wired in.
 * Keeping the transport behind this interface means Android and Desktop can
 * later use different inference backends without changing the shared UI.
 */
class DemoNeralClient : NeralClient {
    override fun snapshot() = RuntimeSnapshot(
        health = RuntimeHealth.READY,
        modelName = "Neral 50M",
        parameterCount = "50.0M",
        execution = "Local preview"
    )

    override fun generate(request: GenerationRequest): String = when (request.mode) {
        AssistantMode.SOLVE -> "Solve mode is ready for the Neral runtime. I would break the problem into steps and verify the result."
        AssistantMode.VERIFY -> "Verify mode will check the answer against the available context and show uncertainty explicitly."
        AssistantMode.BUILD -> "Build mode will turn a request into an editable plan or project artifact before applying changes."
        AssistantMode.CHAT -> "This is the Helios UI shell. Connect the native Neral runtime next and this message will be generated locally."
    }
}
