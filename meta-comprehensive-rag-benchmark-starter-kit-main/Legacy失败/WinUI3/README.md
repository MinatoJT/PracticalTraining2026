# CRAG-MM WinUI 3 前端

这是 Windows 原生适配版本 UI，保留原 Qt UI，不改动原项目骨架。

## 运行条件

1. Windows 10 19041 及以上或 Windows 11。
2. 安装 .NET 8 SDK。
3. 如用 Visual Studio，安装“使用 C++ 的桌面开发”或 Windows App SDK 相关组件。
4. 先按项目根目录 `requirements.txt` 配好 Python 环境。

## 启动

在本目录双击：

```bat
run_winui3.bat
```

或在项目根目录执行：

```bat
dotnet run --project WinUI3\CRAGMM.WinUI.csproj --configuration Release
```

UI 会自动查找项目根目录、优先使用项目 `.venv`，也支持通过 `CRAGMM_PYTHON` 指定 Python。


## 构建组件提示

如果 `dotnet build` 报错缺少 `Microsoft.Build.Packaging.Pri.Tasks.dll`，说明当前电脑只有 .NET SDK，但没有安装 WinUI 3 需要的 Visual Studio/Windows App SDK 构建组件。安装 Visual Studio 2022 或 Build Tools 2022，并勾选 Windows 应用开发、Windows App SDK、Windows 10/11 SDK 后再运行即可。


已验证可用的 VS 组件 ID：`Microsoft.VisualStudio.ComponentGroup.UWP.BuildTools`、`Microsoft.VisualStudio.Component.Windows10SDK.19041`、`Microsoft.VisualStudio.Component.Windows11SDK.26100`。
## 运行时说明

当前项目已启用 `WindowsAppSDKSelfContained=true`，构建输出会携带 Windows App SDK 运行时文件，减少目标机器缺少 `Microsoft.WindowsAppRuntime.1.6` 用户级注册时的启动失败概率。仍建议安装 Visual Studio Build Tools 的 UWP BuildTools 组件用于本地构建。
