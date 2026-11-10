package ai.neral.offline

import android.app.Application

/** Keep a local diagnostic when the UI or native engine causes a Java exception. */
class NeralApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        val original = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            CrashLog.save(filesDir, error)
            if (original != null) original.uncaughtException(thread, error)
            else android.os.Process.killProcess(android.os.Process.myPid())
        }
    }
}
