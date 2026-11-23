plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}
android {
    namespace = "ai.neral.offline"
    compileSdk = 35
    defaultConfig {
        applicationId = "ai.neral.offline.replyfix"
        minSdk = 29
        targetSdk = 35
        versionCode = 4
        versionName = "0.4.0-replyfix"
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }
    buildFeatures { compose = true }
    testOptions { unitTests.isReturnDefaultValues = true }
    packaging { jniLibs { useLegacyPackaging = true } }
}
kotlin { jvmToolchain(21) }
dependencies {
    implementation(project(":llama-kt"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    val composeBom = platform("androidx.compose:compose-bom:2024.09.02")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    testImplementation("junit:junit:4.13.2")
}
