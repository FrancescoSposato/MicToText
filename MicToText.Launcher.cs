// Avviatore di MicToText: unico file da cliccare.
//
// Fa da solo tutto quello che serve prima di aprire l'app:
//   1. controlla che l'ambiente virtuale esista;
//   2. se il server e' gia' acceso apre solo il browser (un secondo server fallirebbe
//      sulla porta occupata);
//   3. se Ollama non risponde lo avvia, altrimenti i menu dei modelli restano vuoti;
//   4. apre un terminale con l'ambiente attivo e avvia il server, che apre il browser.
//
// Dopo Ctrl+C la finestra resta aperta con l'ambiente attivo, pronta per altri comandi:
// e' il motivo di "cmd /k" invece di "cmd /c".
//
// Ricompilare dopo una modifica:
//   C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /target:winexe ^
//     /win32icon:MicToText.ico /out:MicToText.exe /r:System.Windows.Forms.dll ^
//     MicToText.Launcher.cs

using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.NetworkInformation;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

static class Launcher
{
    const int AppPort = 8765;
    const string AppUrl = "http://127.0.0.1:8765/";
    const string OllamaUrl = "http://127.0.0.1:11434/api/tags";

    [STAThread]
    static int Main()
    {
        string dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        string python = Path.Combine(dir, @".venv\Scripts\python.exe");

        if (!File.Exists(python))
        {
            Fail("Ambiente virtuale non trovato in:\n\n" + Path.Combine(dir, ".venv") +
                 "\n\nCrealo con:\n" +
                 "    py -3.12 -m venv .venv\n" +
                 "    .venv\\Scripts\\pip install -r requirements-gpu.txt");
            return 1;
        }

        if (IsPortInUse(AppPort))
        {
            OpenBrowser();
            return 0;
        }

        if (!IsOllamaUp())
            StartOllama();

        try
        {
            // Il server apre il browser da solo un secondo dopo l'avvio.
            var psi = new ProcessStartInfo("cmd.exe",
                "/k \"call .venv\\Scripts\\activate.bat && python -m mictotext.web\"");
            psi.WorkingDirectory = dir;
            psi.UseShellExecute = true;
            Process.Start(psi);
            return 0;
        }
        catch (Exception ex)
        {
            Fail("Avvio non riuscito:\n\n" + ex.Message);
            return 1;
        }
    }

    static bool IsPortInUse(int port)
    {
        try
        {
            foreach (var endpoint in IPGlobalProperties.GetIPGlobalProperties().GetActiveTcpListeners())
                if (endpoint.Port == port)
                    return true;
        }
        catch { /* in dubbio, si prosegue con l'avvio normale */ }
        return false;
    }

    static bool IsOllamaUp()
    {
        try
        {
            var request = (HttpWebRequest)WebRequest.Create(OllamaUrl);
            request.Timeout = 3000;
            using ((HttpWebResponse)request.GetResponse()) { return true; }
        }
        catch { return false; }
    }

    static void StartOllama()
    {
        try
        {
            var psi = new ProcessStartInfo("ollama", "serve");
            psi.UseShellExecute = true;
            psi.WindowStyle = ProcessWindowStyle.Minimized;
            Process.Start(psi);
            for (int i = 0; i < 10 && !IsOllamaUp(); i++)   // fino a ~5 secondi
                Thread.Sleep(500);
        }
        catch { /* l'app parte comunque: mostrera' i menu dei modelli vuoti */ }
    }

    static void OpenBrowser()
    {
        try
        {
            var psi = new ProcessStartInfo(AppUrl);
            psi.UseShellExecute = true;
            Process.Start(psi);
        }
        catch (Exception ex) { Fail("Non riesco ad aprire il browser:\n\n" + ex.Message); }
    }

    static void Fail(string message)
    {
        MessageBox.Show(message, "MicToText", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }
}
