"""
Главное окно приложения — PyQt5 GUI
main.py
"""

import sys
import os
import threading
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QSplitter, QFrame, QStatusBar, QMessageBox,
    QFileDialog, QProgressBar, QTabWidget,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont

from scapy.all import get_if_list

from capture import FlowCapture, PcapCapture
from classifier import Classifier

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "preprocessed", "model_random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "preprocessed", "scaler_app.pkl")

COLOR_ATTACK      = "#FFEBEE"
COLOR_BENIGN      = "#E8F5E9"
COLOR_ATTACK_TEXT = "#C62828"
COLOR_BENIGN_TEXT = "#2E7D32"


# ══════════════════════════════════════════════════════════════════════════════
#  FlowSignal — мост между фоновыми потоками захвата и GUI
# ══════════════════════════════════════════════════════════════════════════════

class FlowSignal(QObject):
    """
    Получает результаты из фоновых потоков захвата и передаёт их в GUI
    через потокобезопасную очередь + QTimer в главном потоке.

    Наследуем QObject (не QThread): этот класс не выполняет собственный
    event loop — он только хранит очередь и испускает Qt-сигналы.
    """
    progress_changed = pyqtSignal(int)   # 0–100, только для PCAP
    analysis_done    = pyqtSignal(int)   # total_flows, только для PCAP

    def __init__(self, classifier: Classifier, parent=None):
        super().__init__(parent)
        self.classifier = classifier
        self._queue: list = []
        self._lock        = threading.Lock()

    # ── вызываются из фоновых потоков ─────────────────────────────────────────

    def on_flow_ready(self, flow_id: tuple, features):
        """Классифицирует поток и кладёт результат в очередь."""
        result = self.classifier.predict(features)
        result.update({
            "flow_id":  flow_id,
            "time":     datetime.now().strftime("%H:%M:%S"),
            "src_ip":   flow_id[0],
            "src_port": flow_id[1],
            "dst_ip":   flow_id[2],
            "dst_port": flow_id[3],
        })
        with self._lock:
            self._queue.append(result)

    def on_progress(self, pct: int):
        self.progress_changed.emit(pct)

    def on_done(self, total_flows: int):
        self.analysis_done.emit(total_flows)

    # ── вызывается из главного потока (QTimer) ────────────────────────────────

    def pop_results(self) -> list:
        with self._lock:
            results       = self._queue.copy()
            self._queue.clear()
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  MainWindow
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обнаружение сетевых аномалий")
        self.resize(1150, 750)

        self.capture:      "FlowCapture | None" = None
        self.pcap_capture: "PcapCapture | None" = None
        self.classifier:   "Classifier  | None" = None
        self.signal_obj:   "FlowSignal  | None" = None

        self.is_running      = False
        self.is_pcap_running = False
        self.total_flows     = 0
        self.attack_count    = 0
        self._pcap_path      = ""

        self._init_classifier()
        self._build_ui()

        # Опрашиваем очередь каждые 300 мс в главном потоке
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_results)
        self._poll_timer.start(300)

    # ── инициализация ─────────────────────────────────────────────────────────

    def _init_classifier(self):
        try:
            self.classifier = Classifier(MODEL_PATH, SCALER_PATH)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    # ── построение UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Вкладки ───────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setFixedHeight(80)

        # Вкладка «Онлайн»
        online_w  = QWidget()
        online_lo = QHBoxLayout(online_w)
        online_lo.setContentsMargins(6, 6, 6, 6)
        online_lo.addWidget(QLabel("Интерфейс:"))
        self.iface_combo = QComboBox()
        self.iface_combo.setMinimumWidth(260)
        self._populate_interfaces()
        online_lo.addWidget(self.iface_combo)
        online_lo.addSpacing(12)

        self.btn_start = QPushButton("▶  Начать захват")
        self.btn_start.setFixedHeight(34)
        self.btn_start.setStyleSheet(self._btn_style("#2E7D32", "#1B5E20"))
        self.btn_start.clicked.connect(self.start_capture)
        online_lo.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  Остановить")
        self.btn_stop.setFixedHeight(34)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self._btn_style("#C62828", "#B71C1C"))
        self.btn_stop.clicked.connect(self.stop_capture)
        online_lo.addWidget(self.btn_stop)
        online_lo.addStretch()
        self.tabs.addTab(online_w, "🌐  Онлайн (интерфейс)")

        # Вкладка «PCAP»
        pcap_w  = QWidget()
        pcap_lo = QHBoxLayout(pcap_w)
        pcap_lo.setContentsMargins(6, 6, 6, 6)

        self.lbl_pcap_file = QLabel("Файл не выбран")
        self.lbl_pcap_file.setStyleSheet("color:#555; font-style:italic;")
        self.lbl_pcap_file.setMinimumWidth(260)
        pcap_lo.addWidget(self.lbl_pcap_file)
        pcap_lo.addSpacing(8)

        self.btn_open_pcap = QPushButton("📂  Выбрать файл")
        self.btn_open_pcap.setFixedHeight(34)
        self.btn_open_pcap.setStyleSheet(self._btn_style("#1565C0", "#0D47A1"))
        self.btn_open_pcap.clicked.connect(self.open_pcap_dialog)
        pcap_lo.addWidget(self.btn_open_pcap)

        self.btn_analyze = QPushButton("▶  Анализировать")
        self.btn_analyze.setFixedHeight(34)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet(self._btn_style("#2E7D32", "#1B5E20"))
        self.btn_analyze.clicked.connect(self.start_pcap_analysis)
        pcap_lo.addWidget(self.btn_analyze)

        self.btn_stop_pcap = QPushButton("■  Стоп")
        self.btn_stop_pcap.setFixedHeight(34)
        self.btn_stop_pcap.setEnabled(False)
        self.btn_stop_pcap.setStyleSheet(self._btn_style("#C62828", "#B71C1C"))
        self.btn_stop_pcap.clicked.connect(self.stop_pcap_analysis)
        pcap_lo.addWidget(self.btn_stop_pcap)

        pcap_lo.addSpacing(16)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        pcap_lo.addWidget(self.progress_bar)
        pcap_lo.addStretch()
        self.tabs.addTab(pcap_w, "📁  Офлайн (PCAP)")

        root.addWidget(self.tabs)

        # ── Статистика ────────────────────────────────────────────────────────
        stat_w  = QFrame()
        stat_lo = QHBoxLayout(stat_w)
        stat_lo.setContentsMargins(4, 0, 4, 0)

        self.btn_clear = QPushButton("Очистить таблицу")
        self.btn_clear.setFixedHeight(28)
        self.btn_clear.clicked.connect(self.clear_table)
        stat_lo.addWidget(self.btn_clear)
        stat_lo.addStretch()

        self.lbl_total   = QLabel("Потоков: 0")
        self.lbl_attacks = QLabel("Атак: 0")
        self.lbl_attacks.setStyleSheet(
            f"color:{COLOR_ATTACK_TEXT}; font-weight:bold;")
        stat_lo.addWidget(self.lbl_total)
        stat_lo.addSpacing(16)
        stat_lo.addWidget(self.lbl_attacks)
        root.addWidget(stat_w)

        # ── Таблица + лог ─────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Время", "Источник", "Порт", "Назначение", "Порт",
            "Статус", "Вероятность атаки, %", "Уверенность, %",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setFont(QFont("Consolas", 9))
        splitter.addWidget(self.table)

        log_w  = QFrame()
        log_w.setFrameShape(QFrame.StyledPanel)
        log_lo = QVBoxLayout(log_w)
        log_lo.setContentsMargins(4, 4, 4, 4)
        log_lo.addWidget(QLabel("Лог событий:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumHeight(150)
        log_lo.addWidget(self.log)
        splitter.addWidget(log_w)

        splitter.setSizes([500, 150])
        root.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Готов к работе. Выберите вкладку и режим анализа.")

    @staticmethod
    def _btn_style(color: str, hover: str) -> str:
        return (
            f"QPushButton {{ background:{color}; color:white; border-radius:6px;"
            f" font-weight:bold; padding:0 14px; }}"
            f"QPushButton:hover {{ background:{hover}; }}"
            f"QPushButton:disabled {{ background:#aaa; }}"
        )

    def _populate_interfaces(self):
        try:
            for iface in get_if_list():
                self.iface_combo.addItem(iface)
        except Exception as e:
            self._log(f"Ошибка получения интерфейсов: {e}")

    # ── Онлайн-режим ──────────────────────────────────────────────────────────

    def start_capture(self):
        if not self.classifier:
            QMessageBox.warning(self, "Ошибка", "Модель не загружена.")
            return
        iface = self.iface_combo.currentText()
        if not iface:
            QMessageBox.warning(self, "Ошибка", "Выберите сетевой интерфейс.")
            return

        self.signal_obj = FlowSignal(self.classifier, parent=self)
        self.capture    = FlowCapture(
            iface=iface,
            on_flow_ready=self.signal_obj.on_flow_ready,
        )
        self.capture.start()
        self.is_running = True

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.iface_combo.setEnabled(False)
        self.tabs.setTabEnabled(1, False)
        self.status_bar.showMessage(f"Захват запущен — интерфейс: {iface}")
        self._log(f"Захват запущен на интерфейсе: {iface}")

    def stop_capture(self):
        if self.capture:
            self.capture.flush()
            self.capture.stop()
            self.capture = None
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.iface_combo.setEnabled(True)
        self.tabs.setTabEnabled(1, True)
        self.status_bar.showMessage("Захват остановлен.")
        self._log("Захват остановлен.")

    # ── PCAP-режим ────────────────────────────────────────────────────────────

    def open_pcap_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть PCAP-файл", "",
            "PCAP files (*.pcap *.pcapng *.cap);;All files (*)",
        )
        if path:
            self._pcap_path = path
            self.lbl_pcap_file.setText(os.path.basename(path))
            self.lbl_pcap_file.setStyleSheet("color:#000; font-style:normal;")
            self.btn_analyze.setEnabled(True)
            self._log(f"Выбран файл: {path}")

    def start_pcap_analysis(self):
        if not self.classifier:
            QMessageBox.warning(self, "Ошибка", "Модель не загружена.")
            return
        if not self._pcap_path:
            QMessageBox.warning(self, "Ошибка", "Файл не выбран.")
            return

        self.signal_obj = FlowSignal(self.classifier, parent=self)
        self.signal_obj.progress_changed.connect(self._on_pcap_progress)
        self.signal_obj.analysis_done.connect(self._on_pcap_done)

        self.pcap_capture = PcapCapture(
            filepath=self._pcap_path,
            on_flow_ready=self.signal_obj.on_flow_ready,
            on_progress=self.signal_obj.on_progress,
            on_done=self.signal_obj.on_done,
        )
        self.pcap_capture.start()
        self.is_pcap_running = True

        self.btn_analyze.setEnabled(False)
        self.btn_open_pcap.setEnabled(False)
        self.btn_stop_pcap.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.tabs.setTabEnabled(0, False)

        self.status_bar.showMessage(
            f"Анализ PCAP: {os.path.basename(self._pcap_path)} ...")
        self._log(f"Начат анализ файла: {self._pcap_path}")

    def stop_pcap_analysis(self):
        if self.pcap_capture:
            self.pcap_capture.stop()
            self.pcap_capture = None
        self._pcap_finished()
        self._log("Анализ PCAP прерван пользователем.")

    def _on_pcap_progress(self, pct: int):
        self.progress_bar.setValue(pct)
        self.status_bar.showMessage(f"Анализ PCAP... {pct}%")

    def _on_pcap_done(self, total_flows: int):
        self._pcap_finished()
        self._log(
            f"Анализ PCAP завершён. Потоков: {total_flows}, "
            f"атак: {self.attack_count}."
        )
        self.status_bar.showMessage(
            f"Анализ завершён — потоков: {total_flows}, атак: {self.attack_count}.")

    def _pcap_finished(self):
        self.is_pcap_running = False
        self.btn_analyze.setEnabled(True)
        self.btn_open_pcap.setEnabled(True)
        self.btn_stop_pcap.setEnabled(False)
        self.progress_bar.setValue(100)
        self.tabs.setTabEnabled(0, True)

    # ── Общие методы ──────────────────────────────────────────────────────────

    def _poll_results(self):
        if self.signal_obj is None:
            return
        for r in self.signal_obj.pop_results():
            self._add_row(r)

    def _add_row(self, result: dict):
        row       = self.table.rowCount()
        is_attack = result["label"] == 1
        bg        = QColor(COLOR_ATTACK if is_attack else COLOR_BENIGN)
        fg        = QColor(COLOR_ATTACK_TEXT if is_attack else COLOR_BENIGN_TEXT)

        self.table.insertRow(row)
        for col, val in enumerate([
            result["time"],
            result["src_ip"],
            str(result["src_port"]),
            result["dst_ip"],
            str(result["dst_port"]),
            result["label_str"],
            f"{result['prob_attack']:.1f}",
            f"{result['confidence']:.1f}",
        ]):
            item = QTableWidgetItem(val)
            item.setBackground(bg)
            if col == 5:
                item.setForeground(fg)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.table.setItem(row, col, item)

        self.table.scrollToBottom()
        self.total_flows += 1
        if is_attack:
            self.attack_count += 1
            self._log(
                f"⚠ АТАКА: {result['src_ip']}:{result['src_port']} → "
                f"{result['dst_ip']}:{result['dst_port']} "
                f"(вероятность {result['prob_attack']:.1f}%)"
            )
        self.lbl_total.setText(f"Потоков: {self.total_flows}")
        self.lbl_attacks.setText(f"Атак: {self.attack_count}")

    def clear_table(self):
        self.table.setRowCount(0)
        self.total_flows  = 0
        self.attack_count = 0
        self.lbl_total.setText("Потоков: 0")
        self.lbl_attacks.setText("Атак: 0")
        self._log("Таблица очищена.")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        if self.is_running:
            self.stop_capture()
        if self.is_pcap_running:
            self.stop_pcap_analysis()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
