package app.helios.design

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val HeliosDark = darkColorScheme(
    primary = Color(0xFFFFB454),
    onPrimary = Color(0xFF2D1600),
    primaryContainer = Color(0xFF3A2917),
    onPrimaryContainer = Color(0xFFFFDCB3),
    secondary = Color(0xFF76D6E8),
    onSecondary = Color(0xFF00363E),
    secondaryContainer = Color(0xFF12343A),
    onSecondaryContainer = Color(0xFFB8F0FA),
    background = Color(0xFF0C0E11),
    onBackground = Color(0xFFE7E9EE),
    surface = Color(0xFF111419),
    onSurface = Color(0xFFE7E9EE),
    surfaceVariant = Color(0xFF1A1E24),
    onSurfaceVariant = Color(0xFFB9C0CA),
    outline = Color(0xFF3A4049),
    error = Color(0xFFFFB4AB)
)

private val HeliosLight = lightColorScheme(
    primary = Color(0xFF7C4D00),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFFFDDB8),
    onPrimaryContainer = Color(0xFF2A1700),
    secondary = Color(0xFF006978),
    onSecondary = Color.White,
    background = Color(0xFFF7F8FB),
    onBackground = Color(0xFF17191D),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF17191D),
    surfaceVariant = Color(0xFFE9EDF2),
    onSurfaceVariant = Color(0xFF444A53),
    outline = Color(0xFF747A83)
)

@Composable
fun HeliosTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = if (darkTheme) HeliosDark else HeliosLight,
        typography = Typography(),
        content = content
    )
}
