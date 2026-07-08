import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_EXE = os.environ.get("CRAGMM_PYTHON", r"C:\anaconda\python.exe")


class EvalWorker(QThread):
    output = Signal(str)
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
            self.output.emit(line)
        self.finished_with_code.emit(self.process.wait())

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setWindowTitle("CRAG-MM 智能实训工具")
        self.resize(980, 760)
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

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        layout.addWidget(QLabel("输出"))
        layout.addWidget(self.output, 1)

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
        return env

    def _update_command_preview(self):
        self.command_preview.setPlainText(" ".join(self._build_command()))

    def _run_eval(self):
        if self.mode_combo.currentData() == "custom":
            if not self.image_edit.text().strip() or not self.question_edit.text().strip():
                self.output.setPlainText("自定义 Task1 模式需要同时选择图片并填写问题。\n")
                return
        self.output.clear()
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = EvalWorker(self._build_command(), self._build_env())
        self.worker.output.connect(self.output.insertPlainText)
        self.worker.finished_with_code.connect(self._finished)
        self.worker.start()

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