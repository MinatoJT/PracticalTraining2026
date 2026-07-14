# CRAG-MM Qt UI

最简 Qt 前端，用于选择 Task1/Task2/Task3 并启动本地评测。

## 启动

```powershell
cd "D:\USER DOC\JERRY\Desktop\CRAGMM\meta-comprehensive-rag-benchmark-starter-kit-main"
C:\anaconda\python.exe UI\app.py
```

也可以双击 `UI\run_ui.bat`。

## 说明

- Task1 默认使用 `Task1KGAgent`，并禁用 web search。
- Task2/Task3 默认使用项目里的 `agents.user_config.UserAgent`，方便后续替换你的多源/多轮 agent。
- 运行结果会保存到 `UI\outputs\<task>`。

## Custom Task1 question

在 UI 的 Mode 中选择 Custom Task1 question，选择本地图片并输入问题，即可用 Task1 图像检索 API + Task1KGAgent 做单条问答。


## Dataset 缓存目录

UI 已将 HuggingFace 数据集、模型、Xet 缓存，以及 CRAG 图片/网页缓存统一设置到项目根目录下的 Dataset/。如果出现旧缓存损坏，可以优先删除 Dataset/hf_* 后重新运行。

