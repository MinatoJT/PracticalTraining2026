using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace CRAGMM.WinUI;

public sealed partial class MainWindow : Window
{
    private readonly string _projectRoot;
    private Process? _process;

    public MainWindow()
    {
        InitializeComponent();
        _projectRoot = FindProjectRoot();

        ModeCombo.SelectionChanged += (_, _) => SyncMode();
        TaskCombo.SelectionChanged += (_, _) => SyncTaskDefaultAgent();
        AgentCombo.SelectionChanged += (_, _) => UpdateCommandPreview();
        ConversationCount.ValueChanged += (_, _) => UpdateCommandPreview();
        DisplayCount.ValueChanged += (_, _) => UpdateCommandPreview();
        EvalModelCombo.SelectionChanged += (_, _) => UpdateCommandPreview();
        ImagePathBox.TextChanged += (_, _) => UpdateCommandPreview();
        QuestionBox.TextChanged += (_, _) => UpdateCommandPreview();
        ApiKeyBox.PasswordChanged += (_, _) => UpdateCommandPreview();
        DeepSeekModelBox.TextChanged += (_, _) => UpdateCommandPreview();
        RevisionBox.TextChanged += (_, _) => UpdateCommandPreview();
        DisableProgressBox.Checked += (_, _) => UpdateCommandPreview();
        DisableProgressBox.Unchecked += (_, _) => UpdateCommandPreview();
        BrowseButton.Click += BrowseButton_Click;
        RunButton.Click += RunButton_Click;
        StopButton.Click += StopButton_Click;

        SyncMode();
        SyncTaskDefaultAgent();
        UpdateCommandPreview();
    }

    private static string FindProjectRoot()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "UI", "run_eval.py")))
            {
                return current.FullName;
            }
            current = current.Parent;
        }
        throw new DirectoryNotFoundException("未找到包含 UI\\run_eval.py 的项目根目录。");
    }

    private string SelectedTag(ComboBox combo)
    {
        return (combo.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "";
    }

    private void SyncMode()
    {
        var isCustom = SelectedTag(ModeCombo) == "custom";
        TaskCombo.IsEnabled = !isCustom;
        AgentCombo.IsEnabled = !isCustom;
        ConversationCount.IsEnabled = !isCustom;
        DisplayCount.IsEnabled = !isCustom;
        EvalModelCombo.IsEnabled = !isCustom;
        ImagePathBox.IsEnabled = isCustom;
        BrowseButton.IsEnabled = isCustom;
        QuestionBox.IsEnabled = isCustom;
        UpdateCommandPreview();
    }

    private void SyncTaskDefaultAgent()
    {
        var task = SelectedTag(TaskCombo);
        if (task == "task1")
        {
            AgentCombo.SelectedIndex = 0;
        }
        else if (task == "task2")
        {
            AgentCombo.SelectedIndex = 1;
        }
        else
        {
            AgentCombo.SelectedIndex = 2;
        }
        UpdateCommandPreview();
    }

    private async void BrowseButton_Click(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".png");
        picker.FileTypeFilter.Add(".jpg");
        picker.FileTypeFilter.Add(".jpeg");
        picker.FileTypeFilter.Add(".bmp");
        picker.FileTypeFilter.Add(".webp");

        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));
        var file = await picker.PickSingleFileAsync();
        if (file is not null)
        {
            ImagePathBox.Text = file.Path;
        }
    }

    private void RunButton_Click(object sender, RoutedEventArgs e)
    {
        if (_process is not null && !_process.HasExited)
        {
            return;
        }

        var command = BuildCommand();
        if (command is null)
        {
            return;
        }

        OutputBox.Text = "";
        AppendOutput("运行命令: " + FormatCommand(command.Value.Executable, command.Value.Arguments) + Environment.NewLine);

        var psi = new ProcessStartInfo
        {
            FileName = command.Value.Executable,
            WorkingDirectory = _projectRoot,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        foreach (var arg in command.Value.Arguments)
        {
            psi.ArgumentList.Add(arg);
        }

        foreach (var item in BuildEnvironment())
        {
            psi.Environment[item.Key] = item.Value;
        }

        _process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _process.OutputDataReceived += (_, args) => AppendProcessLine(args.Data);
        _process.ErrorDataReceived += (_, args) => AppendProcessLine(args.Data);
        _process.Exited += (_, _) =>
        {
            DispatcherQueue.TryEnqueue(() =>
            {
                AppendOutput(Environment.NewLine + $"进程结束，退出码 {_process?.ExitCode}" + Environment.NewLine);
                RunButton.IsEnabled = true;
                StopButton.IsEnabled = false;
            });
        };

        try
        {
            _process.Start();
            _process.BeginOutputReadLine();
            _process.BeginErrorReadLine();
            RunButton.IsEnabled = false;
            StopButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            AppendOutput("启动失败: " + ex.Message + Environment.NewLine);
            RunButton.IsEnabled = true;
            StopButton.IsEnabled = false;
        }
    }

    private void StopButton_Click(object sender, RoutedEventArgs e)
    {
        if (_process is not null && !_process.HasExited)
        {
            _process.Kill(entireProcessTree: true);
        }
    }

    private void AppendProcessLine(string? line)
    {
        if (line is null)
        {
            return;
        }

        DispatcherQueue.TryEnqueue(() => AppendOutput(line + Environment.NewLine));
    }

    private void AppendOutput(string text)
    {
        OutputBox.Text += text;
        OutputBox.SelectionStart = OutputBox.Text.Length;
    }

    private (string Executable, List<string> Arguments)? BuildCommand()
    {
        var python = ResolvePython();
        if (python is null)
        {
            AppendOutput("未找到 Python。请安装 Python，或设置 CRAGMM_PYTHON 环境变量。" + Environment.NewLine);
            return null;
        }

        var args = new List<string>(python.Value.PrefixArguments);
        var isCustom = SelectedTag(ModeCombo) == "custom";

        if (isCustom)
        {
            if (string.IsNullOrWhiteSpace(ImagePathBox.Text) || string.IsNullOrWhiteSpace(QuestionBox.Text))
            {
                AppendOutput("自定义模式需要同时填写图片路径和问题。" + Environment.NewLine);
                return null;
            }

            args.Add(Path.Combine(_projectRoot, "UI", "custom_task1.py"));
            args.Add("--image");
            args.Add(ImagePathBox.Text.Trim());
            args.Add("--question");
            args.Add(QuestionBox.Text.Trim());
        }
        else
        {
            args.Add(Path.Combine(_projectRoot, "UI", "run_eval.py"));
            args.Add("--task");
            args.Add(SelectedTag(TaskCombo));
            args.Add("--agent");
            args.Add(SelectedTag(AgentCombo));
            args.Add("--num-conversations");
            args.Add(((int)Math.Round(ConversationCount.Value)).ToString());
            args.Add("--display-conversations");
            args.Add(((int)Math.Round(DisplayCount.Value)).ToString());
            args.Add("--eval-model");
            args.Add(SelectedTag(EvalModelCombo));
            args.Add("--revision");
            args.Add(RevisionBox.Text.Trim());

            if (DisableProgressBox.IsChecked == true)
            {
                args.Add("--no-progress");
            }
        }

        return (python.Value.Executable, args);
    }

    private Dictionary<string, string> BuildEnvironment()
    {
        var dataset = Path.Combine(_projectRoot, "Dataset");
        Directory.CreateDirectory(dataset);

        var env = new Dictionary<string, string>
        {
            ["PYTHONDONTWRITEBYTECODE"] = "1",
            ["PYTHONUTF8"] = "1",
            ["PANDAS_USE_NUMEXPR"] = "0",
            ["PANDAS_USE_BOTTLENECK"] = "0",
            ["HF_HOME"] = Path.Combine(dataset, "hf_home"),
            ["HF_DATASETS_CACHE"] = Path.Combine(dataset, "hf_datasets"),
            ["HUGGINGFACE_HUB_CACHE"] = Path.Combine(dataset, "hf_hub"),
            ["HF_XET_CACHE"] = Path.Combine(dataset, "hf_xet"),
            ["TRANSFORMERS_CACHE"] = Path.Combine(dataset, "transformers"),
            ["SENTENCE_TRANSFORMERS_HOME"] = Path.Combine(dataset, "sentence_transformers"),
            ["CRAG_CACHE_DIR"] = Path.Combine(dataset, "crag_images"),
            ["CRAG_WEBSEARCH_CACHE_DIR"] = Path.Combine(dataset, "crag_web_search"),
            ["TASK1_DEBUG_PATH"] = Path.Combine(_projectRoot, "UI", "outputs", "task1", "debug.jsonl"),
            ["TASK2_DEBUG_PATH"] = Path.Combine(_projectRoot, "UI", "outputs", "task2", "debug.jsonl"),
            ["TASK3_DEBUG_PATH"] = Path.Combine(_projectRoot, "UI", "outputs", "task3", "debug.jsonl"),
        };

        if (!string.IsNullOrWhiteSpace(ApiKeyBox.Password))
        {
            env["DEEPSEEK_API_KEY"] = ApiKeyBox.Password.Trim();
        }

        if (!string.IsNullOrWhiteSpace(DeepSeekModelBox.Text))
        {
            env["DEEPSEEK_MODEL"] = DeepSeekModelBox.Text.Trim();
        }

        return env;
    }

    private (string Executable, List<string> PrefixArguments)? ResolvePython()
    {
        var configured = Environment.GetEnvironmentVariable("CRAGMM_PYTHON");
        if (!string.IsNullOrWhiteSpace(configured)
            && File.Exists(configured)
            && PythonHasBackendDependencies(configured, Array.Empty<string>()))
        {
            return (configured, new List<string> { "-B" });
        }

        var venvPython = Path.Combine(_projectRoot, ".venv", "Scripts", "python.exe");
        if (File.Exists(venvPython) && PythonHasBackendDependencies(venvPython, Array.Empty<string>()))
        {
            return (venvPython, new List<string> { "-B" });
        }

        var condaPrefix = Environment.GetEnvironmentVariable("CONDA_PREFIX");
        if (!string.IsNullOrWhiteSpace(condaPrefix))
        {
            var condaPython = Path.Combine(condaPrefix, "python.exe");
            if (File.Exists(condaPython) && PythonHasBackendDependencies(condaPython, Array.Empty<string>()))
            {
                return (condaPython, new List<string> { "-B" });
            }
        }

        // Double-clicking the WinUI launcher does not always inherit Conda's PATH.
        var commonAnacondaPython = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.System).Substring(0, 3),
            "anaconda",
            "python.exe");
        if (File.Exists(commonAnacondaPython)
            && PythonHasBackendDependencies(commonAnacondaPython, Array.Empty<string>()))
        {
            return (commonAnacondaPython, new List<string> { "-B" });
        }

        if (CommandExists("python") && PythonHasBackendDependencies("python", Array.Empty<string>()))
        {
            return ("python", new List<string> { "-B" });
        }

        if (CommandExists("py") && PythonHasBackendDependencies("py", new[] { "-3" }))
        {
            return ("py", new List<string> { "-3", "-B" });
        }

        return null;
    }

    private static bool PythonHasBackendDependencies(string executable, IReadOnlyList<string> prefixArguments)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = executable,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            foreach (var argument in prefixArguments)
            {
                psi.ArgumentList.Add(argument);
            }
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add("import datasets");

            using var process = Process.Start(psi);
            if (process is null || !process.WaitForExit(10000))
            {
                process?.Kill(true);
                return false;
            }
            return process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private static bool CommandExists(string command)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "where",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add(command);

            using var process = Process.Start(psi);
            process?.WaitForExit(1500);
            return process?.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private void UpdateCommandPreview()
    {
        var command = BuildPreviewCommand();
        CommandPreviewBox.Text = command;
    }

    private string BuildPreviewCommand()
    {
        var python = ResolvePython();
        if (python is null)
        {
            return "未找到 Python。可设置 CRAGMM_PYTHON，或在项目根目录创建 .venv。";
        }

        var command = BuildCommandWithoutValidation(python.Value.Executable, python.Value.PrefixArguments);
        return FormatCommand(command.Executable, command.Arguments);
    }

    private (string Executable, List<string> Arguments) BuildCommandWithoutValidation(string executable, List<string> prefixArguments)
    {
        var args = new List<string>(prefixArguments);
        var isCustom = SelectedTag(ModeCombo) == "custom";
        if (isCustom)
        {
            args.Add(Path.Combine(_projectRoot, "UI", "custom_task1.py"));
            args.Add("--image");
            args.Add(ImagePathBox.Text.Trim());
            args.Add("--question");
            args.Add(QuestionBox.Text.Trim());
        }
        else
        {
            args.Add(Path.Combine(_projectRoot, "UI", "run_eval.py"));
            args.Add("--task");
            args.Add(SelectedTag(TaskCombo));
            args.Add("--agent");
            args.Add(SelectedTag(AgentCombo));
            args.Add("--num-conversations");
            args.Add(((int)Math.Round(ConversationCount.Value)).ToString());
            args.Add("--display-conversations");
            args.Add(((int)Math.Round(DisplayCount.Value)).ToString());
            args.Add("--eval-model");
            args.Add(SelectedTag(EvalModelCombo));
            args.Add("--revision");
            args.Add(RevisionBox.Text.Trim());
            if (DisableProgressBox.IsChecked == true)
            {
                args.Add("--no-progress");
            }
        }

        return (executable, args);
    }

    private static string FormatCommand(string executable, IReadOnlyList<string> args)
    {
        var parts = new List<string> { Quote(executable) };
        foreach (var arg in args)
        {
            parts.Add(Quote(arg));
        }
        return string.Join(" ", parts);
    }

    private static string Quote(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return "\"\"";
        }
        return value.Contains(' ') ? $"\"{value}\"" : value;
    }
}
