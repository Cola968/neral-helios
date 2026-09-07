package app.helios.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.helios.model.*
import app.helios.runtime.DemoNeralClient
import app.helios.runtime.GenerationRequest
import app.helios.runtime.RuntimeHealth

private data class NavItem(
    val section: HeliosSection,
    val icon: ImageVector
)

private val primaryNav = listOf(
    NavItem(HeliosSection.HOME, Icons.Rounded.Home),
    NavItem(HeliosSection.CHAT, Icons.Rounded.ChatBubbleOutline),
    NavItem(HeliosSection.SOLVE, Icons.Rounded.Calculate),
    NavItem(HeliosSection.PROJECTS, Icons.Rounded.FolderOpen),
    NavItem(HeliosSection.LIBRARY, Icons.Rounded.CollectionsBookmark)
)

@Composable
fun HeliosShell() {
    var section by remember { mutableStateOf(HeliosSection.HOME) }
    var mode by remember { mutableStateOf(AssistantMode.CHAT) }
    val client = remember { DemoNeralClient() }
    val runtime = client.snapshot()
    var messages by remember { mutableStateOf(DemoContent.messages) }

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        val compact = maxWidth < 720.dp
        val expandedNav = maxWidth >= 1220.dp
        val showConversationPane = maxWidth >= 1020.dp
        val showInspector = maxWidth >= 1500.dp

        if (compact) {
            Column(Modifier.fillMaxSize()) {
                CompactTopBar(runtime.modelName)
                Box(Modifier.weight(1f).fillMaxWidth()) {
                    ContentRouter(
                        section = section,
                        mode = mode,
                        onModeChange = { mode = it },
                        messages = messages,
                        onSend = { text ->
                            val user = ChatMessage("u${messages.size}", ChatMessage.Role.USER, text)
                            val reply = ChatMessage(
                                "a${messages.size + 1}",
                                ChatMessage.Role.ASSISTANT,
                                client.generate(GenerationRequest(text, mode)),
                                "${runtime.modelName} · ${runtime.execution}"
                            )
                            messages = messages + user + reply
                        },
                        showConversationPane = false,
                        showInspector = false,
                        onOpenChat = { section = HeliosSection.CHAT }
                    )
                }
                CompactNavigation(section, onSelect = { section = it })
            }
        } else {
            Row(Modifier.fillMaxSize()) {
                DesktopNavigation(
                    section = section,
                    expanded = expandedNav,
                    onSelect = { section = it }
                )
                Box(Modifier.weight(1f).fillMaxHeight()) {
                    ContentRouter(
                        section = section,
                        mode = mode,
                        onModeChange = { mode = it },
                        messages = messages,
                        onSend = { text ->
                            val user = ChatMessage("u${messages.size}", ChatMessage.Role.USER, text)
                            val reply = ChatMessage(
                                "a${messages.size + 1}",
                                ChatMessage.Role.ASSISTANT,
                                client.generate(GenerationRequest(text, mode)),
                                "${runtime.modelName} · ${runtime.execution}"
                            )
                            messages = messages + user + reply
                        },
                        showConversationPane = showConversationPane,
                        showInspector = showInspector,
                        onOpenChat = { section = HeliosSection.CHAT }
                    )
                }
            }
        }
    }
}

@Composable
private fun CompactTopBar(modelName: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp)
            .padding(horizontal = 18.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        HeliosMark(36.dp)
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text("Helios", fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
            Text(modelName, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        IconButton(onClick = {}) {
            Icon(Icons.Rounded.Search, contentDescription = "Search")
        }
        IconButton(onClick = {}) {
            Icon(Icons.Rounded.MoreHoriz, contentDescription = "More")
        }
    }
}

@Composable
private fun HeliosMark(size: androidx.compose.ui.unit.Dp) {
    Box(
        modifier = Modifier
            .size(size)
            .clip(RoundedCornerShape(size / 3))
            .background(MaterialTheme.colorScheme.primaryContainer),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            Icons.Rounded.WbSunny,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(size * 0.58f)
        )
    }
}

@Composable
private fun DesktopNavigation(
    section: HeliosSection,
    expanded: Boolean,
    onSelect: (HeliosSection) -> Unit
) {
    val width = if (expanded) 238.dp else 86.dp
    Surface(
        modifier = Modifier.width(width).fillMaxHeight(),
        color = MaterialTheme.colorScheme.surface
    ) {
        Column(Modifier.fillMaxSize().padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth().height(56.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = if (expanded) Arrangement.Start else Arrangement.Center
            ) {
                HeliosMark(38.dp)
                if (expanded) {
                    Spacer(Modifier.width(12.dp))
                    Column {
                        Text("Helios", fontWeight = FontWeight.Bold, fontSize = 18.sp)
                        Text("Neral workspace", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
            Spacer(Modifier.height(14.dp))

            FilledTonalButton(
                onClick = { onSelect(HeliosSection.CHAT) },
                modifier = Modifier.fillMaxWidth().height(48.dp),
                contentPadding = PaddingValues(horizontal = if (expanded) 14.dp else 0.dp)
            ) {
                Icon(Icons.Rounded.Add, contentDescription = null)
                if (expanded) {
                    Spacer(Modifier.width(10.dp))
                    Text("New chat")
                }
            }

            Spacer(Modifier.height(16.dp))
            primaryNav.forEach { item ->
                DesktopNavItem(
                    item = item,
                    selected = section == item.section,
                    expanded = expanded,
                    onClick = { onSelect(item.section) }
                )
            }
            Spacer(Modifier.weight(1f))
            DesktopNavItem(
                item = NavItem(HeliosSection.SETTINGS, Icons.Rounded.Settings),
                selected = section == HeliosSection.SETTINGS,
                expanded = expanded,
                onClick = { onSelect(HeliosSection.SETTINGS) }
            )
            if (expanded) {
                Spacer(Modifier.height(12.dp))
                RuntimeMiniCard()
            }
        }
    }
}

@Composable
private fun DesktopNavItem(
    item: NavItem,
    selected: Boolean,
    expanded: Boolean,
    onClick: () -> Unit
) {
    val container = if (selected) MaterialTheme.colorScheme.surfaceVariant else MaterialTheme.colorScheme.surface
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(48.dp)
            .clip(RoundedCornerShape(15.dp))
            .background(container)
            .clickable(onClick = onClick)
            .padding(horizontal = if (expanded) 14.dp else 0.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = if (expanded) Arrangement.Start else Arrangement.Center
    ) {
        Icon(
            item.icon,
            contentDescription = item.section.label,
            tint = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
        )
        if (expanded) {
            Spacer(Modifier.width(13.dp))
            Text(item.section.label, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal)
        }
    }
    Spacer(Modifier.height(4.dp))
}

@Composable
private fun RuntimeMiniCard() {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = MaterialTheme.colorScheme.surfaceVariant
    ) {
        Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(9.dp).clip(CircleShape).background(MaterialTheme.colorScheme.secondary))
            Spacer(Modifier.width(9.dp))
            Column(Modifier.weight(1f)) {
                Text("Neral 50M", style = MaterialTheme.typography.labelLarge)
                Text("Local · ready", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun CompactNavigation(section: HeliosSection, onSelect: (HeliosSection) -> Unit) {
    NavigationBar(containerColor = MaterialTheme.colorScheme.surface) {
        primaryNav.take(5).forEach { item ->
            NavigationBarItem(
                selected = section == item.section,
                onClick = { onSelect(item.section) },
                icon = { Icon(item.icon, contentDescription = item.section.label) },
                label = { Text(item.section.label, maxLines = 1) }
            )
        }
    }
}

@Composable
private fun ContentRouter(
    section: HeliosSection,
    mode: AssistantMode,
    onModeChange: (AssistantMode) -> Unit,
    messages: List<ChatMessage>,
    onSend: (String) -> Unit,
    showConversationPane: Boolean,
    showInspector: Boolean,
    onOpenChat: () -> Unit
) {
    when (section) {
        HeliosSection.HOME -> HomeScreen(onOpenChat)
        HeliosSection.CHAT -> ChatWorkspace(
            mode = mode,
            onModeChange = onModeChange,
            messages = messages,
            onSend = onSend,
            showConversationPane = showConversationPane,
            showInspector = showInspector
        )
        HeliosSection.SOLVE -> SolveScreen(onOpenChat)
        HeliosSection.PROJECTS -> ProjectsScreen()
        HeliosSection.LIBRARY -> LibraryScreen()
        HeliosSection.SETTINGS -> SettingsScreen()
    }
}

@Composable
private fun HomeScreen(onOpenChat: () -> Unit) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 24.dp, vertical = 28.dp),
        verticalArrangement = Arrangement.spacedBy(22.dp)
    ) {
        item {
            Column(Modifier.fillMaxWidth()) {
                Text(
                    "Good evening.",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.primary
                )
                Spacer(Modifier.height(6.dp))
                Text(
                    "What do you want to build with Helios?",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    "Chat, solve, verify or continue a project. Neral stays separate from the interface so the app can evolve with every model release.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        }

        item {
            Surface(
                modifier = Modifier.fillMaxWidth().clickable(onClick = onOpenChat),
                shape = RoundedCornerShape(24.dp),
                color = MaterialTheme.colorScheme.surfaceVariant
            ) {
                Row(
                    Modifier.fillMaxWidth().padding(18.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Rounded.AutoAwesome, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.width(12.dp))
                    Text("Ask Helios anything…", color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
                    FilledIconButton(onClick = onOpenChat) {
                        Icon(Icons.Rounded.ArrowUpward, contentDescription = "Open chat")
                    }
                }
            }
        }

        item {
            Text("Modes", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(12.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                QuickActionCard("Chat", "Think and talk", Icons.Rounded.ChatBubbleOutline, Modifier.weight(1f), onOpenChat)
                QuickActionCard("Solve", "Math & physics", Icons.Rounded.Calculate, Modifier.weight(1f), onOpenChat)
            }
            Spacer(Modifier.height(12.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                QuickActionCard("Verify", "Check an answer", Icons.Rounded.FactCheck, Modifier.weight(1f), onOpenChat)
                QuickActionCard("Build", "Create a project", Icons.Rounded.Build, Modifier.weight(1f), onOpenChat)
            }
        }

        item {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("Recent projects", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                TextButton(onClick = {}) { Text("See all") }
            }
            Spacer(Modifier.height(8.dp))
        }

        items(DemoContent.projects) { project ->
            ProjectRow(project)
        }
    }
}

@Composable
private fun QuickActionCard(
    title: String,
    subtitle: String,
    icon: ImageVector,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Surface(
        modifier = modifier.height(116.dp).clickable(onClick = onClick),
        shape = RoundedCornerShape(22.dp),
        color = MaterialTheme.colorScheme.surface
    ) {
        Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.SpaceBetween) {
            Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.secondary)
            Column {
                Text(title, fontWeight = FontWeight.SemiBold)
                Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun ProjectRow(project: ProjectCard) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surface
    ) {
        Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(44.dp).clip(RoundedCornerShape(13.dp)).background(MaterialTheme.colorScheme.primaryContainer),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Rounded.FolderOpen, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(project.title, fontWeight = FontWeight.SemiBold)
                Text(project.subtitle, maxLines = 1, overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
            }
            AssistChip(onClick = {}, label = { Text(project.progress) })
        }
    }
}

@Composable
private fun ChatWorkspace(
    mode: AssistantMode,
    onModeChange: (AssistantMode) -> Unit,
    messages: List<ChatMessage>,
    onSend: (String) -> Unit,
    showConversationPane: Boolean,
    showInspector: Boolean
) {
    Row(Modifier.fillMaxSize()) {
        if (showConversationPane) {
            ConversationPane(Modifier.width(286.dp).fillMaxHeight())
            VerticalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f))
        }
        ChatPane(
            mode = mode,
            onModeChange = onModeChange,
            messages = messages,
            onSend = onSend,
            modifier = Modifier.weight(1f).fillMaxHeight()
        )
        if (showInspector) {
            VerticalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f))
            InspectorPane(Modifier.width(300.dp).fillMaxHeight())
        }
    }
}

@Composable
private fun ConversationPane(modifier: Modifier = Modifier) {
    Column(modifier.background(MaterialTheme.colorScheme.surface).padding(14.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Chats", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            IconButton(onClick = {}) { Icon(Icons.Rounded.EditSquare, contentDescription = "New chat") }
        }
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = "",
            onValueChange = {},
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("Search") },
            leadingIcon = { Icon(Icons.Rounded.Search, contentDescription = null) },
            singleLine = true,
            shape = RoundedCornerShape(16.dp)
        )
        Spacer(Modifier.height(12.dp))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            items(DemoContent.conversations) { conversation ->
                val bg = if (conversation.active) MaterialTheme.colorScheme.surfaceVariant else MaterialTheme.colorScheme.surface
                Column(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(15.dp)).background(bg).clickable {}.padding(12.dp)
                ) {
                    Text(conversation.title, maxLines = 1, overflow = TextOverflow.Ellipsis, fontWeight = if (conversation.active) FontWeight.SemiBold else FontWeight.Normal)
                    Spacer(Modifier.height(3.dp))
                    Text(conversation.preview, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun ChatPane(
    mode: AssistantMode,
    onModeChange: (AssistantMode) -> Unit,
    messages: List<ChatMessage>,
    onSend: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    var input by remember { mutableStateOf("") }

    Column(modifier) {
        Row(
            Modifier.fillMaxWidth().height(68.dp).padding(horizontal = 20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(Modifier.weight(1f)) {
                Text("Neral 50M Roadmap", fontWeight = FontWeight.SemiBold)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(7.dp).clip(CircleShape).background(MaterialTheme.colorScheme.secondary))
                    Spacer(Modifier.width(7.dp))
                    Text("Local runtime ready", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            AssistChip(
                onClick = {},
                label = { Text("Neral 50M") },
                leadingIcon = { Icon(Icons.Rounded.Memory, contentDescription = null, modifier = Modifier.size(18.dp)) }
            )
            Spacer(Modifier.width(6.dp))
            IconButton(onClick = {}) { Icon(Icons.Rounded.MoreHoriz, contentDescription = "More") }
        }
        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.25f))

        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 24.dp, vertical = 28.dp),
            verticalArrangement = Arrangement.spacedBy(22.dp)
        ) {
            items(messages) { message -> MessageBubble(message) }
        }

        Column(Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 14.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AssistantMode.entries.forEach { item ->
                    FilterChip(
                        selected = mode == item,
                        onClick = { onModeChange(item) },
                        label = { Text(item.shortLabel) },
                        leadingIcon = if (mode == item) {
                            { Icon(modeIcon(item), contentDescription = null, modifier = Modifier.size(17.dp)) }
                        } else null
                    )
                }
            }
            Spacer(Modifier.height(10.dp))
            Surface(
                shape = RoundedCornerShape(24.dp),
                color = MaterialTheme.colorScheme.surfaceVariant
            ) {
                Column(Modifier.fillMaxWidth().padding(10.dp)) {
                    OutlinedTextField(
                        value = input,
                        onValueChange = { input = it },
                        modifier = Modifier.fillMaxWidth(),
                        placeholder = { Text("Message Helios…") },
                        minLines = 2,
                        maxLines = 6,
                        shape = RoundedCornerShape(18.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            unfocusedBorderColor = MaterialTheme.colorScheme.surfaceVariant,
                            focusedBorderColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    )
                    Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
                        IconButton(onClick = {}) { Icon(Icons.Rounded.Add, contentDescription = "Attach") }
                        IconButton(onClick = {}) { Icon(Icons.Rounded.Image, contentDescription = "Image") }
                        Text("Local", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
                        FilledIconButton(
                            onClick = {
                                val text = input.trim()
                                if (text.isNotEmpty()) {
                                    onSend(text)
                                    input = ""
                                }
                            },
                            enabled = input.isNotBlank()
                        ) {
                            Icon(Icons.Rounded.ArrowUpward, contentDescription = "Send")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(message: ChatMessage) {
    val assistant = message.role == ChatMessage.Role.ASSISTANT
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (assistant) Arrangement.Start else Arrangement.End
    ) {
        if (assistant) {
            HeliosMark(34.dp)
            Spacer(Modifier.width(10.dp))
        }
        Surface(
            modifier = Modifier.widthIn(max = 760.dp),
            shape = RoundedCornerShape(20.dp),
            color = if (assistant) MaterialTheme.colorScheme.surface else MaterialTheme.colorScheme.primaryContainer
        ) {
            Column(Modifier.padding(horizontal = 16.dp, vertical = 13.dp)) {
                Text(message.text, style = MaterialTheme.typography.bodyLarge)
                if (message.meta != null) {
                    Spacer(Modifier.height(7.dp))
                    Text(message.meta, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun InspectorPane(modifier: Modifier = Modifier) {
    Column(modifier.background(MaterialTheme.colorScheme.surface).padding(18.dp)) {
        Text("Context", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(16.dp))
        InspectorCard("Model", "Neral 50M", Icons.Rounded.Memory)
        Spacer(Modifier.height(10.dp))
        InspectorCard("Project", "Neral", Icons.Rounded.FolderOpen)
        Spacer(Modifier.height(10.dp))
        InspectorCard("Context", "2 messages · local", Icons.Rounded.History)
        Spacer(Modifier.height(22.dp))
        Text("Helios actions", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(8.dp))
        ActionRow("Verify answer", Icons.Rounded.FactCheck)
        ActionRow("Show dependencies", Icons.Rounded.AccountTree)
        ActionRow("Create project", Icons.Rounded.CreateNewFolder)
        Spacer(Modifier.weight(1f))
        Text("Nothing leaves this device in local mode.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun InspectorCard(title: String, value: String, icon: ImageVector) {
    Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.secondary)
            Spacer(Modifier.width(10.dp))
            Column {
                Text(title, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(value, fontWeight = FontWeight.Medium)
            }
        }
    }
}

@Composable
private fun ActionRow(label: String, icon: ImageVector) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).clickable {}.padding(vertical = 10.dp, horizontal = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(10.dp))
        Text(label, modifier = Modifier.weight(1f))
        Icon(Icons.Rounded.ChevronRight, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun SolveScreen(onOpenChat: () -> Unit) {
    FeatureScreen(
        eyebrow = "SOLVE",
        title = "Work through difficult problems, not just answers.",
        description = "Math and physics get a dedicated workspace with steps, assumptions, verification and a clean final result.",
        icon = Icons.Rounded.Calculate,
        button = "Open Solve chat",
        onClick = onOpenChat
    )
}

@Composable
private fun ProjectsScreen() {
    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = PaddingValues(28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Text("Projects", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
            Text("Persistent workspaces for chats, files and decisions.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(12.dp))
        }
        items(DemoContent.projects) { ProjectRow(it) }
    }
}

@Composable
private fun LibraryScreen() {
    FeatureScreen(
        eyebrow = "LIBRARY",
        title = "One place for chats, files and model outputs.",
        description = "Library will index your local Helios content without mixing it into model training.",
        icon = Icons.Rounded.CollectionsBookmark,
        button = "Add file",
        onClick = {}
    )
}

@Composable
private fun SettingsScreen() {
    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = PaddingValues(28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Text("Settings", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
            Text("Model, runtime, privacy and interface.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(18.dp))
        }
        item { SettingsRow("Model", "Neral 50M · local", Icons.Rounded.Memory) }
        item { SettingsRow("Runtime", "Ready", Icons.Rounded.Bolt) }
        item { SettingsRow("Privacy", "Local-first", Icons.Rounded.Security) }
        item { SettingsRow("Appearance", "System", Icons.Rounded.Palette) }
    }
}

@Composable
private fun SettingsRow(title: String, subtitle: String, icon: ImageVector) {
    Surface(shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surface) {
        Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(title, fontWeight = FontWeight.Medium)
                Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Icon(Icons.Rounded.ChevronRight, contentDescription = null)
        }
    }
}

@Composable
private fun FeatureScreen(
    eyebrow: String,
    title: String,
    description: String,
    icon: ImageVector,
    button: String,
    onClick: () -> Unit
) {
    Box(Modifier.fillMaxSize().padding(28.dp), contentAlignment = Alignment.Center) {
        Column(Modifier.widthIn(max = 720.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                Modifier.size(72.dp).clip(RoundedCornerShape(24.dp)).background(MaterialTheme.colorScheme.primaryContainer),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(36.dp))
            }
            Spacer(Modifier.height(20.dp))
            Text(eyebrow, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(6.dp))
            Text(title, style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(12.dp))
            Text(description, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyLarge)
            Spacer(Modifier.height(22.dp))
            Button(onClick = onClick) { Text(button) }
        }
    }
}

private fun modeIcon(mode: AssistantMode): ImageVector = when (mode) {
    AssistantMode.CHAT -> Icons.Rounded.ChatBubbleOutline
    AssistantMode.SOLVE -> Icons.Rounded.Calculate
    AssistantMode.VERIFY -> Icons.Rounded.FactCheck
    AssistantMode.BUILD -> Icons.Rounded.Build
}
