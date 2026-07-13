import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_EXE = os.environ.get("CRAGMM_PYTHON") or sys.executable
LIVE_EVENT_PREFIX = "__CRAGMM_LIVE_EVENT__"


class ConversationView(QWidget):
    """单个测试会话的聊天视图；Task3 的多轮消息会持续追加到同一页。"""

    def __init__(self):
        super().__init__()
        self.image_shown = False
        self.seen_turns = set()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.content = QWidget()
        self.messages = QVBoxLayout(self.content)
        self.messages.setContentsMargins(12, 12, 12, 12)
        self.messages.setSpacing(10)
        self.messages.addStretch(1)
        self.scroll.setWidget(self.content)
        root_layout.addWidget(self.scroll)

    def add_event(self, event):
        turn = int(event.get("turn", 0))
        if turn in self.seen_turns:
            return
        self.seen_turns.add(turn)

        image_path = str(event.get("image_path", ""))
        if image_path and not self.image_shown:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                image_label = QLabel()
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setPixmap(
                    pixmap.scaled(460, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                image_label.setStyleSheet("background: #151515; padding: 8px; border: 1px solid #3a3a3a;")
                self.messages.insertWidget(self.messages.count() - 1, image_label)
                self.image_shown = True

        self._add_bubble(f"Query · Turn {turn + 1}", str(event.get("query", "")), is_user=True)
        self._add_bubble("Agent Response", str(event.get("response", "")), is_user=False)
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _add_bubble(self, title, text, is_user):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        bubble = QFrame()
        bubble.setMaximumWidth(500)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 9, 12, 10)
        bubble_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 11px; font-weight: 700; color: #bfc7d5;")
        body = QLabel(text or "(empty response)")
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setStyleSheet("font-size: 13px; color: #f2f2f2;")
        bubble_layout.addWidget(title_label)
        bubble_layout.addWidget(body)

        if is_user:
            bubble.setStyleSheet("QFrame { background: #174a73; border: 1px solid #286a9c; border-radius: 6px; }")
            row.addStretch(1)
            row.addWidget(bubble)
        else:
            bubble.setStyleSheet("QFrame { background: #303030; border: 1px solid #484848; border-radius: 6px; }")
            row.addWidget(bubble)
            row.addStretch(1)
        self.messages.insertLayout(self.messages.count() - 1, row)

    def _scroll_to_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())


class EvalWorker(QThread):
    output = Signal(str)
    live_event = Signal(object)
    finished_with_code = Signal(int)

    def __init__(self, command, env):
        super().__init__()
        self.command = command
        self.env = env
        self.process = None

    def run(self):
        self.output.emit("运行命令: " + " ".join(self.command) + "\n")
        self.process = subprocess.Popen(
            self.command,
            cwd=str(ROOT_DIR),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert self.process.stdout is not None
        for line in self.process.stdout:
            marker = line.find(LIVE_EVENT_PREFIX)
            if marker < 0:
                self.output.emit(line)
                continue
            if marker > 0:
                self.output.emit(line[:marker])
            payload_text = line[marker + len(LIVE_EVENT_PREFIX):].strip()
            try:
                payload, _ = json.JSONDecoder().raw_decode(payload_text)
                self.live_event.emit(payload)
            except Exception:
                self.output.emit(line)
        self.finished_with_code.emit(self.process.wait())

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.conversation_views = {}
        self.conversation_numbers = {}
        self.setWindowTitle("CRAG-MM 智能实训工具")
        self.resize(1320, 860)
        self._build_ui()
        self._sync_task_defaults()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        header = QLabel("CRAG-MM 任务运行器")
        header.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(header)

        form_box = QGroupBox("运行设置")
        form = QFormLayout(form_box)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("数据集评测", "eval")
        self.mode_combo.addItem("自定义 Task1 问答", "custom")
        self.mode_combo.currentIndexChanged.connect(self._sync_mode)
        form.addRow("运行模式", self.mode_combo)

        self.task_combo = QComboBox()
        self.task_combo.addItem("Task 1 - 单源增强", "task1")
        self.task_combo.addItem("Task 2 - 多源增强", "task2")
        self.task_combo.addItem("Task 3 - 多轮问答", "task3")
        self.task_combo.currentIndexChanged.connect(self._sync_task_defaults)
        form.addRow("任务", self.task_combo)

        self.agent_combo = QComboBox()
        self.agent_combo.addItem("Task1KGAgent（Task1 知识图谱）", "task1kg")
        self.agent_combo.addItem("Task2Agent（Task2 多源增强）", "task2agent")
        self.agent_combo.addItem("Task3Agent（Task3 上下文优化）", "task3agent")
        self.agent_combo.addItem("项目 user_config.UserAgent", "user_config")
        form.addRow("智能体", self.agent_combo)

        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 5000)
        self.num_spin.setValue(20)
        form.addRow("评测样本数", self.num_spin)

        self.display_spin = QSpinBox()
        self.display_spin.setRange(0, 50)
        self.display_spin.setValue(5)
        form.addRow("展示样例数", self.display_spin)

        self.eval_combo = QComboBox()
        self.eval_combo.addItem("不使用语义评测（仅 exact match）", "None")
        self.eval_combo.addItem("gpt-4o-mini - 语义评测", "gpt-4o-mini")
        self.eval_combo.addItem("deepseek-v4-flash - 语义评测", "deepseek-v4-flash")
        form.addRow("评测模型", self.eval_combo)

        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText("选择一张图片用于自定义 Task1 问答")
        self.browse_button = QPushButton("选择图片")
        self.browse_button.clicked.connect(self._browse_image)
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(self.browse_button)
        form.addRow("自定义图片", image_row)

        self.question_edit = QLineEdit()
        self.question_edit.setPlaceholderText("示例：What is this building called?")
        form.addRow("自定义问题", self.question_edit)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setText(os.environ.get("DEEPSEEK_API_KEY", ""))
        self.api_key_edit.setPlaceholderText("使用 DeepSeek 生成或语义评测时填写")
        form.addRow("DeepSeek API Key", self.api_key_edit)

        self.model_edit = QLineEdit(os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
        form.addRow("DeepSeek 模型", self.model_edit)

        self.vision_enabled_check = QCheckBox("启用 Qwen3-VL 视觉增强")
        self.vision_enabled_check.setChecked(os.environ.get("VISION_ENABLED", "0") == "1")
        form.addRow("视觉模型", self.vision_enabled_check)

        self.qwen_key_edit = QLineEdit()
        self.qwen_key_edit.setEchoMode(QLineEdit.Password)
        self.qwen_key_edit.setText(os.environ.get("QWEN_VL_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")))
        self.qwen_key_edit.setPlaceholderText("百炼 Qwen VL API Key（不会写入命令预览）")
        form.addRow("Qwen VL API Key", self.qwen_key_edit)

        legacy_qwen_model = os.environ.get("QWEN_VL_MODEL", "").strip()
        self.qwen_anchor_model_edit = QLineEdit(os.environ.get(
            "QWEN_VL_ANCHOR_MODEL", legacy_qwen_model or "qwen3.5-omni-plus"
        ))
        form.addRow("视觉锚点模型", self.qwen_anchor_model_edit)

        self.qwen_rerank_model_edit = QLineEdit(os.environ.get(
            "QWEN_VL_RERANK_MODEL", legacy_qwen_model or "qwen3.5-omni-flash"
        ))
        form.addRow("候选重排模型", self.qwen_rerank_model_edit)

        self.qwen_base_url_edit = QLineEdit(os.environ.get(
            "QWEN_VL_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ))
        form.addRow("Qwen VL Base URL", self.qwen_base_url_edit)

        self.revision_edit = QLineEdit("v0.1.2")
        form.addRow("数据集版本", self.revision_edit)

        self.no_progress_check = QCheckBox("禁用进度条")
        self.no_progress_check.setChecked(False)
        form.addRow("选项", self.no_progress_check)

        layout.addWidget(form_box)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("运行")
        self.run_button.clicked.connect(self._run_eval)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self._stop_eval)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(94)
        layout.addWidget(QLabel("命令预览"))
        layout.addWidget(self.command_preview)

        results_splitter = QSplitter(Qt.Horizontal)

        output_panel = QWidget()
        output_layout = QVBoxLayout(output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(QLabel("输出"))
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        output_layout.addWidget(self.output, 1)

        live_group = QGroupBox("实时测试对话")
        live_layout = QVBoxLayout(live_group)
        self.live_tabs = QTabWidget()
        self.live_tabs.setDocumentMode(True)
        live_layout.addWidget(self.live_tabs)

        results_splitter.addWidget(output_panel)
        results_splitter.addWidget(live_group)
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 1)
        results_splitter.setSizes([600, 700])
        layout.addWidget(results_splitter, 1)

        self.setCentralWidget(root)
        self._sync_mode()

        for widget in [
            self.mode_combo,
            self.task_combo,
            self.agent_combo,
            self.num_spin,
            self.display_spin,
            self.eval_combo,
            self.image_edit,
            self.question_edit,
            self.model_edit,
            self.vision_enabled_check,
            self.qwen_anchor_model_edit,
            self.qwen_rerank_model_edit,
            self.qwen_base_url_edit,
            self.revision_edit,
            self.no_progress_check,
        ]:
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._update_command_preview)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._update_command_preview)
            if hasattr(widget, "textChanged"):
                widget.textChanged.connect(self._update_command_preview)
            if hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._update_command_preview)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(ROOT_DIR),
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*.*)",
        )
        if path:
            self.image_edit.setText(path)

    def _sync_mode(self):
        custom = self.mode_combo.currentData() == "custom"
        self.task_combo.setEnabled(not custom)
        self.agent_combo.setEnabled(not custom)
        self.num_spin.setEnabled(not custom)
        self.display_spin.setEnabled(not custom)
        self.eval_combo.setEnabled(not custom)
        self.revision_edit.setEnabled(not custom)
        self.no_progress_check.setEnabled(not custom)
        self.image_edit.setEnabled(custom)
        self.browse_button.setEnabled(custom)
        self.question_edit.setEnabled(custom)
        if custom:
            self.task_combo.setCurrentIndex(0)
            self.agent_combo.setCurrentIndex(0)
        self._update_command_preview()

    def _sync_task_defaults(self):
        task = self.task_combo.currentData()
        if task == "task1":
            self.agent_combo.setCurrentIndex(0)
        elif task == "task2":
            self.agent_combo.setCurrentIndex(1)
        else:
            self.agent_combo.setCurrentIndex(2)
        self._update_command_preview()

    def _build_command(self):
        if self.mode_combo.currentData() == "custom":
            return [
                PYTHON_EXE,
                str(ROOT_DIR / "UI" / "custom_task1.py"),
                "--image",
                self.image_edit.text().strip(),
                "--question",
                self.question_edit.text().strip(),
            ]

        command = [
            PYTHON_EXE,
            str(ROOT_DIR / "UI" / "run_eval.py"),
            "--task",
            self.task_combo.currentData(),
            "--agent",
            self.agent_combo.currentData(),
            "--num-conversations",
            str(self.num_spin.value()),
            "--display-conversations",
            str(self.display_spin.value()),
            "--eval-model",
            self.eval_combo.currentData(),
            "--revision",
            self.revision_edit.text().strip() or "v0.1.2",
        ]
        if self.no_progress_check.isChecked():
            command.append("--no-progress")
        return command

    def _build_env(self):
        env = os.environ.copy()
        key = self.api_key_edit.text().strip()
        if key:
            env["DEEPSEEK_API_KEY"] = key
        model = self.model_edit.text().strip()
        if model:
            env["DEEPSEEK_MODEL"] = model
        env["VISION_ENABLED"] = "1" if self.vision_enabled_check.isChecked() else "0"
        qwen_key = self.qwen_key_edit.text().strip()
        if qwen_key:
            env["QWEN_VL_API_KEY"] = qwen_key
        qwen_anchor_model = self.qwen_anchor_model_edit.text().strip()
        qwen_rerank_model = self.qwen_rerank_model_edit.text().strip()
        if "-realtime" in qwen_anchor_model.lower() or "-realtime" in qwen_rerank_model.lower():
            raise ValueError("realtime_model_not_supported")
        if qwen_anchor_model:
            env["QWEN_VL_ANCHOR_MODEL"] = qwen_anchor_model
        if qwen_rerank_model:
            env["QWEN_VL_RERANK_MODEL"] = qwen_rerank_model
        env["QWEN_VL_FALLBACK_MODEL"] = os.environ.get("QWEN_VL_FALLBACK_MODEL", "qwen3-vl-flash")
        qwen_base_url = self.qwen_base_url_edit.text().strip()
        if qwen_base_url:
            env["QWEN_VL_BASE_URL"] = qwen_base_url
        env["QWEN_VL_PROVIDER"] = "dashscope"
        env["QWEN_VL_ENABLE_THINKING"] = "0"
        dataset_dir = ROOT_DIR / "Dataset"
        dataset_dir.mkdir(exist_ok=True)
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PANDAS_USE_NUMEXPR"] = "0"
        env["PANDAS_USE_BOTTLENECK"] = "0"
        env["HF_HOME"] = str(dataset_dir / "hf_home")
        env["HF_DATASETS_CACHE"] = str(dataset_dir / "hf_datasets")
        env["HUGGINGFACE_HUB_CACHE"] = str(dataset_dir / "hf_hub")
        env["HF_XET_CACHE"] = str(dataset_dir / "hf_xet")
        env["TRANSFORMERS_CACHE"] = str(dataset_dir / "transformers")
        env["SENTENCE_TRANSFORMERS_HOME"] = str(dataset_dir / "sentence_transformers")
        env["CRAG_CACHE_DIR"] = str(dataset_dir / "crag_images")
        env["CRAG_WEBSEARCH_CACHE_DIR"] = str(dataset_dir / "crag_web_search")
        env["TASK1_DEBUG_PATH"] = str(ROOT_DIR / "UI" / "outputs" / "task1" / "debug.jsonl")
        env["TASK2_DEBUG_PATH"] = str(ROOT_DIR / "UI" / "outputs" / "task2" / "debug.jsonl")
        env["TASK3_DEBUG_PATH"] = str(ROOT_DIR / "UI" / "outputs" / "task3" / "debug.jsonl")
        env["CRAGMM_LIVE_EVENTS"] = "1"
        return env

    def _update_command_preview(self):
        self.command_preview.setPlainText(" ".join(self._build_command()))

    def _run_eval(self):
        if self.mode_combo.currentData() == "custom":
            if not self.image_edit.text().strip() or not self.question_edit.text().strip():
                self.output.setPlainText("自定义 Task1 模式需要同时选择图片并填写问题。\n")
                return
        self.output.clear()
        self._clear_live_conversations()
        try:
            child_env = self._build_env()
        except ValueError as exc:
            self.output.setPlainText(str(exc) + "\n")
            return
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = EvalWorker(self._build_command(), child_env)
        self.worker.output.connect(self.output.insertPlainText)
        self.worker.live_event.connect(self._append_live_event)
        self.worker.finished_with_code.connect(self._finished)
        self.worker.start()

    def _clear_live_conversations(self):
        self.conversation_views.clear()
        self.conversation_numbers.clear()
        while self.live_tabs.count():
            widget = self.live_tabs.widget(0)
            self.live_tabs.removeTab(0)
            widget.deleteLater()

    def _append_live_event(self, event):
        conversation_id = str(event.get("conversation_id", "unknown"))
        view = self.conversation_views.get(conversation_id)
        if view is None:
            view = ConversationView()
            number = len(self.conversation_views) + 1
            self.conversation_views[conversation_id] = view
            self.conversation_numbers[conversation_id] = number
            self.live_tabs.addTab(view, f"会话 {number}")
        view.add_event(event)
        self.live_tabs.setCurrentWidget(view)

    def _stop_eval(self):
        if self.worker:
            self.worker.stop()

    def _finished(self, code):
        self.output.insertPlainText(f"\n进程结束，退出码 {code}。\n")
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
