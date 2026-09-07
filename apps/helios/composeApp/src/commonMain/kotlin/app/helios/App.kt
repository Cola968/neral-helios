package app.helios

import androidx.compose.runtime.Composable
import app.helios.design.HeliosTheme
import app.helios.ui.HeliosShell

@Composable
fun HeliosApp() {
    HeliosTheme {
        HeliosShell()
    }
}
