pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS); repositories { google(); mavenCentral() } }
rootProject.name = "NERALOffline"
include(":app", ":llama-kt")
project(":llama-kt").projectDir = file("libs/llama.kt/llama-kt")
