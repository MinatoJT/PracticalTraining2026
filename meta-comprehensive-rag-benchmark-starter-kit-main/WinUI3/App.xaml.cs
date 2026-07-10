using Microsoft.UI.Xaml;
using System;
using System.IO;

namespace CRAGMM.WinUI;

public partial class App : Application
{
    private Window? _window;

    public App()
    {
        InitializeComponent();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            WriteStartupLog("OnLaunched begin");
            _window = new MainWindow();
            WriteStartupLog("MainWindow created");
            _window.Activate();
            WriteStartupLog("MainWindow activated");
        }
        catch (Exception ex)
        {
            WriteStartupLog(ex.ToString());
            throw;
        }
    }

    private static void WriteStartupLog(string message)
    {
        try
        {
            var logPath = Path.Combine(AppContext.BaseDirectory, "cragmm_winui_startup.log");
            File.AppendAllText(logPath, DateTime.Now.ToString("s") + " " + message + Environment.NewLine);
        }
        catch
        {
        }
    }
}
