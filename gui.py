import sys
import os
import json
import logging
import shutil
import tempfile
import threading
import time
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QMainWindow,
                             QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QPushButton, QFileDialog, QMessageBox, QGroupBox, 
                             QLineEdit, QFormLayout, QCheckBox, QSizePolicy)
from PyQt6.QtGui import (QIcon, QAction, QColor, QPixmap, QPainter, QBrush,
                         QKeySequence, QPen, QLinearGradient, QActionGroup)
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QUrl, QMimeData, QDir, QTimer, QRectF
import keyboard
import numpy as np
from audio_recorder import (
    AudioRecorder,
    AudioMonitorThread,
    get_default_device_id,
    get_default_loopback_device_id,
    get_devices,
    play_test_sound,
)
from clipboard_utils import copy_file_to_clipboard
from logging_setup import configure_logging, default_log_folder

CONFIG_FILE = "settings.json"
logger = logging.getLogger(__name__)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class SignalManager(QObject):
    recording_finished = pyqtSignal(str, str)
    recording_start_requested = pyqtSignal(str)
    recording_stop_requested = pyqtSignal()

class HotkeyEdit(QLineEdit):
    """
    Custom widget to capture hotkeys by pressing them.
    Maps Qt events to 'keyboard' library compatible strings.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Click to set hotkey...")
        self.setReadOnly(True) 
        self.current_sequence = None

    def mousePressEvent(self, event):
        self.setFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        
        if key == Qt.Key.Key_Backspace or key == Qt.Key.Key_Delete:
            self.clear()
            self.current_sequence = None
            return
            
        if key == Qt.Key.Key_Escape:
            self.clearFocus()
            return

        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return

        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier: parts.append("ctrl")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:   parts.append("shift")
        if modifiers & Qt.KeyboardModifier.AltModifier:     parts.append("alt")
        if modifiers & Qt.KeyboardModifier.MetaModifier:    parts.append("windows")

        key_text = ""
        if key >= 0x20 and key <= 0x7E:
            key_text = chr(key).lower()
        else:
            key_map = {
                Qt.Key.Key_F1: "f1", Qt.Key.Key_F2: "f2", Qt.Key.Key_F3: "f3", Qt.Key.Key_F4: "f4",
                Qt.Key.Key_F5: "f5", Qt.Key.Key_F6: "f6", Qt.Key.Key_F7: "f7", Qt.Key.Key_F8: "f8",
                Qt.Key.Key_F9: "f9", Qt.Key.Key_F10: "f10", Qt.Key.Key_F11: "f11", Qt.Key.Key_F12: "f12",
                Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right", Qt.Key.Key_Up: "up", Qt.Key.Key_Down: "down",
                Qt.Key.Key_Space: "space", Qt.Key.Key_Tab: "tab", Qt.Key.Key_Return: "enter", Qt.Key.Key_Enter: "enter",
                Qt.Key.Key_Backspace: "backspace", Qt.Key.Key_Delete: "delete", Qt.Key.Key_Insert: "insert",
                Qt.Key.Key_Home: "home", Qt.Key.Key_End: "end", Qt.Key.Key_PageUp: "pageup", Qt.Key.Key_PageDown: "pagedown",
                Qt.Key.Key_CapsLock: "capslock", Qt.Key.Key_NumLock: "numlock", Qt.Key.Key_ScrollLock: "scrolllock",
                Qt.Key.Key_Print: "print_screen", Qt.Key.Key_Pause: "pause"
            }
            key_text = key_map.get(key)
            if not key_text:
                try: key_text = QKeySequence(key).toString().lower()
                except: pass

        if key_text:
            parts.append(key_text)
            
        final_hotkey = "+".join(parts)
        self.setText(final_hotkey)
        self.current_sequence = final_hotkey
        self.clearFocus()


class TestSignals(QObject):
    levels_updated = pyqtSignal(list, float)
    mic_error = pyqtSignal(str)
    audio_test_finished = pyqtSignal()
    audio_test_error = pyqtSignal(str)


class AudioSpectrumWidget(QWidget):
    """
    Visualizer widget rendering a real-time frequency spectrum with smooth peak decay
    and gradient bars.
    """

    def __init__(self, num_bands=28, parent=None):
        super().__init__(parent)
        self.num_bands = num_bands
        self._current_levels = [0.0] * num_bands
        self._target_levels = [0.0] * num_bands
        self._peak_levels = [0.0] * num_bands
        self._is_active = False

        self._decay_timer = QTimer(self)
        self._decay_timer.setInterval(28)  # ~35 fps
        self._decay_timer.timeout.connect(self._animate_step)

        self.setMinimumHeight(44)
        self.setMaximumHeight(54)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

    def set_active(self, active: bool):
        self._is_active = active
        if active:
            if not self._decay_timer.isActive():
                self._decay_timer.start()
        else:
            self._target_levels = [0.0] * self.num_bands
            if not self._decay_timer.isActive():
                self._decay_timer.start()

    def update_levels(self, band_values: list, rms: float = 0.0):
        if not self._is_active:
            return
        normalized = []
        for v in band_values:
            # Map decibels [-60 dB, 0 dB] into [0.0, 1.0]
            db = 20.0 * np.log10(max(float(v), 1e-4))
            norm = max(0.0, min(1.0, (db + 60.0) / 60.0))
            normalized.append(norm)

        if len(normalized) < self.num_bands:
            normalized += [0.0] * (self.num_bands - len(normalized))
        self._target_levels = normalized[: self.num_bands]

    def _animate_step(self):
        changed = False
        for i in range(self.num_bands):
            target = self._target_levels[i]
            cur = self._current_levels[i]
            peak = self._peak_levels[i]

            if target > cur:
                new_val = cur * 0.35 + target * 0.65
            else:
                new_val = max(0.0, cur * 0.84 - 0.008)

            if new_val > peak:
                new_peak = new_val
            else:
                new_peak = max(0.0, peak - 0.012)

            if abs(new_val - cur) > 0.003 or abs(new_peak - peak) > 0.003:
                changed = True

            self._current_levels[i] = new_val
            self._peak_levels[i] = new_peak

        if not self._is_active and all(
            lvl <= 0.005 for lvl in self._current_levels
        ):
            self._current_levels = [0.0] * self.num_bands
            self._peak_levels = [0.0] * self.num_bands
            self._decay_timer.stop()
            changed = True

        if changed or self._is_active:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        bg_rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(QBrush(QColor(17, 24, 39)))
        painter.setPen(QPen(QColor(55, 65, 81), 1))
        painter.drawRoundedRect(bg_rect, 6.0, 6.0)

        if not self._is_active and all(
            lvl <= 0.01 for lvl in self._current_levels
        ):
            painter.setPen(QColor(107, 114, 128))
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(
                bg_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Microphone idle • Click 'Test Mic' to monitor spectrum",
            )
            return

        pad_x = 8.0
        pad_y = 6.0
        avail_w = max(10.0, float(w) - (pad_x * 2.0))
        avail_h = max(10.0, float(h) - (pad_y * 2.0))
        gap = 2.5
        bar_w = max(
            2.0, (avail_w - (gap * (self.num_bands - 1))) / self.num_bands
        )

        for i in range(self.num_bands):
            level = self._current_levels[i]
            peak = self._peak_levels[i]
            bar_h = max(2.0, level * avail_h)
            x = pad_x + i * (bar_w + gap)
            y = float(h) - pad_y - bar_h

            grad = QLinearGradient(
                x, float(h) - pad_y, x, float(h) - pad_y - avail_h
            )
            grad.setColorAt(0.0, QColor(6, 182, 212))
            grad.setColorAt(0.55, QColor(16, 185, 129))
            grad.setColorAt(0.85, QColor(245, 158, 11))
            grad.setColorAt(1.0, QColor(239, 68, 68))

            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(x, y, bar_w, bar_h), 1.5, 1.5)

            if peak > 0.05:
                peak_y = max(pad_y, float(h) - pad_y - (peak * avail_h))
                painter.setBrush(QBrush(QColor(243, 244, 246)))
                painter.drawRect(QRectF(x, peak_y, bar_w, 1.5))


class SettingsWindow(QMainWindow):
    settings_saved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings - Simple Audio Recorder")
        self.setWindowIcon(QIcon(resource_path("app_icon.ico")))
        self.setGeometry(100, 100, 520, 670)

        self.tray_app = None
        self.monitor_thread = None
        self.is_playing_audio_test = False

        self.test_signals = TestSignals()
        self.test_signals.levels_updated.connect(self._on_spectrum_levels)
        self.test_signals.mic_error.connect(self._on_mic_test_error)
        self.test_signals.audio_test_finished.connect(self._on_audio_test_finished)
        self.test_signals.audio_test_error.connect(self._on_audio_test_error)

        self.mic_test_timer = QTimer(self)
        self.mic_test_timer.setSingleShot(True)
        self.mic_test_timer.setInterval(15000)
        self.mic_test_timer.timeout.connect(self.stop_microphone_test)

        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Input devices
        group_mic = QGroupBox("Input Devices")
        layout_mic = QVBoxLayout()

        # Microphone row
        row_mic = QHBoxLayout()
        lbl_mic = QLabel("Microphone:")
        lbl_mic.setFixedWidth(85)
        self.combo_mic = QComboBox()
        self.combo_mic.currentIndexChanged.connect(self._on_mic_selection_changed)
        self.btn_test_mic = QPushButton("Test Mic")
        self.btn_test_mic.setFixedWidth(90)
        self.btn_test_mic.setToolTip(
            "Test selected microphone and display real-time frequency spectrum"
        )
        self.btn_test_mic.clicked.connect(self.toggle_microphone_test)
        row_mic.addWidget(lbl_mic)
        row_mic.addWidget(self.combo_mic, 1)
        row_mic.addWidget(self.btn_test_mic)
        layout_mic.addLayout(row_mic)

        # Real-time spectrum visualizer
        self.spectrum_widget = AudioSpectrumWidget()
        layout_mic.addWidget(self.spectrum_widget)

        # System audio row
        row_loop = QHBoxLayout()
        lbl_loop = QLabel("System audio:")
        lbl_loop.setFixedWidth(85)
        self.combo_loopback = QComboBox()
        self.btn_test_loopback = QPushButton("Test Audio")
        self.btn_test_loopback.setFixedWidth(90)
        self.btn_test_loopback.setToolTip(
            "Play a test chime on the output device corresponding to the selected system audio"
        )
        self.btn_test_loopback.clicked.connect(self.play_system_audio_test)
        row_loop.addWidget(lbl_loop)
        row_loop.addWidget(self.combo_loopback, 1)
        row_loop.addWidget(self.btn_test_loopback)
        layout_mic.addLayout(row_loop)

        btn_refresh = QPushButton("Refresh Devices")
        btn_refresh.clicked.connect(self.refresh_devices)
        layout_mic.addWidget(btn_refresh)
        group_mic.setLayout(layout_mic)
        layout.addWidget(group_mic)

        # Output
        group_out = QGroupBox("Output Configuration")
        layout_out = QFormLayout()
        
        layout_folder_inner = QHBoxLayout()
        self.lbl_folder = QLabel(os.getcwd())
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_folder)
        layout_folder_inner.addWidget(self.lbl_folder)
        layout_folder_inner.addWidget(btn_browse)
        
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["MP3", "WAV"])
        self.combo_fmt.currentTextChanged.connect(self._update_source_wav_option)
        
        layout_out.addRow("Folder:", layout_folder_inner)
        layout_out.addRow("Format:", self.combo_fmt)

        layout_log_folder = QHBoxLayout()
        self.lbl_log_folder = QLabel(default_log_folder())
        btn_browse_log = QPushButton("Browse...")
        btn_browse_log.clicked.connect(self.browse_log_folder)
        layout_log_folder.addWidget(self.lbl_log_folder)
        layout_log_folder.addWidget(btn_browse_log)
        layout_out.addRow("Log folder:", layout_log_folder)

        group_out.setLayout(layout_out)
        layout.addWidget(group_out)

        # Tray Interaction
        group_tray = QGroupBox("Tray Icon Behavior")
        layout_tray = QFormLayout()
        self.combo_left_click = QComboBox()
        self.combo_left_click.addItems(["Last Used", "Microphone", "Loopback", "Both"])
        layout_tray.addRow("Left Click Action:", self.combo_left_click)
        group_tray.setLayout(layout_tray)
        layout.addWidget(group_tray)

        # Post-Processing
        group_post = QGroupBox("Post-Processing & Clipboard")
        layout_post = QVBoxLayout()
        self.chk_normalize = QCheckBox("Normalize Audio (Apply first)")
        self.chk_clipboard = QCheckBox("Copy File to Clipboard")
        self.chk_delete = QCheckBox("Delete after Copy (Move to Temp)")
        self.chk_keep_source_wav = QCheckBox("Keep original WAV source files")
        self.chk_keep_source_wav.setToolTip(
            "Keep the uncompressed microphone and system audio WAV files when the output format is MP3."
        )
        self.chk_delete.setToolTip("Moves the file to the system temp folder before copying, keeping your output folder clean.")
        self.chk_delete.setEnabled(False)
        self.chk_clipboard.toggled.connect(lambda c: self.chk_delete.setEnabled(c))
        
        layout_post.addWidget(self.chk_normalize)
        layout_post.addWidget(self.chk_clipboard)
        layout_post.addWidget(self.chk_delete)
        layout_post.addWidget(self.chk_keep_source_wav)
        group_post.setLayout(layout_post)
        layout.addWidget(group_post)

        # Hotkeys
        group_hotkeys = QGroupBox("Global Hotkeys")
        layout_hotkeys = QFormLayout()
        self.hk_mic = HotkeyEdit()
        self.hk_loop = HotkeyEdit()
        self.hk_both = HotkeyEdit()
        self.hk_stop = HotkeyEdit()
        layout_hotkeys.addRow("Record Mic:", self.hk_mic)
        layout_hotkeys.addRow("Record Loopback:", self.hk_loop)
        layout_hotkeys.addRow("Record Both:", self.hk_both)
        layout_hotkeys.addRow("Stop Recording:", self.hk_stop)
        group_hotkeys.setLayout(layout_hotkeys)
        layout.addWidget(group_hotkeys)

        btn_save = QPushButton("Save Settings")
        btn_save.clicked.connect(self.save_settings)
        layout.addWidget(btn_save)

        self.refresh_devices()

    def refresh_devices(self, preserve_selection=True):
        self.stop_microphone_test()

        prev_mic_id = self.combo_mic.currentData() if preserve_selection else None
        prev_mic_name = self.combo_mic.currentText() if preserve_selection else None

        prev_loop_id = self.combo_loopback.currentData() if preserve_selection else None
        prev_loop_name = self.combo_loopback.currentText() if preserve_selection else None

        self.combo_mic.blockSignals(True)
        self.combo_loopback.blockSignals(True)
        self.combo_mic.clear()
        self.combo_loopback.clear()
        try:
            mics = get_devices(include_loopback=False)
            loopbacks = get_devices(include_loopback=True)
            default_mic_id = get_default_device_id()
            default_loopback_id = get_default_loopback_device_id()

            mic_index_to_select = -1
            default_mic_index = 0
            for i, m in enumerate(mics):
                self.combo_mic.addItem(f"{m['name']}", m['id'])
                if prev_mic_name and m['name'] == prev_mic_name:
                    mic_index_to_select = i
                elif mic_index_to_select == -1 and prev_mic_id is not None and m['id'] == prev_mic_id:
                    mic_index_to_select = i
                if m['id'] == default_mic_id:
                    default_mic_index = i

            if mic_index_to_select >= 0:
                self.combo_mic.setCurrentIndex(mic_index_to_select)
            elif len(mics) > 0:
                self.combo_mic.setCurrentIndex(default_mic_index)

            loop_index_to_select = -1
            default_loopback_index = 0
            for i, device in enumerate(loopbacks):
                self.combo_loopback.addItem(device["name"], device["id"])
                if prev_loop_name and device["name"] == prev_loop_name:
                    loop_index_to_select = i
                elif loop_index_to_select == -1 and prev_loop_id is not None and device["id"] == prev_loop_id:
                    loop_index_to_select = i
                if device["id"] == default_loopback_id:
                    default_loopback_index = i

            if loop_index_to_select >= 0:
                self.combo_loopback.setCurrentIndex(loop_index_to_select)
            elif len(loopbacks) > 0:
                self.combo_loopback.setCurrentIndex(default_loopback_index)

        except Exception as e:
            logger.exception("Error refreshing devices")
        finally:
            self.combo_mic.blockSignals(False)
            self.combo_loopback.blockSignals(False)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", options=QFileDialog.Option.DontUseNativeDialog)
        if folder:
            self.lbl_folder.setText(folder)

    def _update_source_wav_option(self, output_format=None):
        is_mp3 = (output_format or self.combo_fmt.currentText()) == "MP3"
        self.chk_keep_source_wav.setEnabled(is_mp3)

    def browse_log_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Log Folder", options=QFileDialog.Option.DontUseNativeDialog)
        if folder:
            self.lbl_log_folder.setText(folder)

    def load_settings(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    
                self.lbl_folder.setText(data.get("output_folder", os.getcwd()))
                self.lbl_log_folder.setText(data.get("log_folder", default_log_folder()))
                fmt_idx = self.combo_fmt.findText(data.get("format", "MP3"))
                if fmt_idx >= 0: self.combo_fmt.setCurrentIndex(fmt_idx)
                
                saved_id = data.get("device_id")
                saved_name = data.get("device_name")
                if saved_name:
                    idx = self.combo_mic.findText(saved_name)
                    if idx >= 0:
                        self.combo_mic.setCurrentIndex(idx)
                    elif saved_id is not None:
                        idx = self.combo_mic.findData(saved_id)
                        if idx >= 0:
                            self.combo_mic.setCurrentIndex(idx)
                elif saved_id is not None:
                    idx = self.combo_mic.findData(saved_id)
                    if idx >= 0:
                        self.combo_mic.setCurrentIndex(idx)

                saved_loopback_id = data.get("loopback_id")
                saved_loopback_name = data.get("loopback_name")
                if saved_loopback_name:
                    idx = self.combo_loopback.findText(saved_loopback_name)
                    if idx >= 0:
                        self.combo_loopback.setCurrentIndex(idx)
                    elif saved_loopback_id is not None:
                        idx = self.combo_loopback.findData(saved_loopback_id)
                        if idx >= 0:
                            self.combo_loopback.setCurrentIndex(idx)
                elif saved_loopback_id is not None:
                    idx = self.combo_loopback.findData(saved_loopback_id)
                    if idx >= 0:
                        self.combo_loopback.setCurrentIndex(idx)

                mode = data.get("tray_click_mode", "Last Used")
                mode_idx = self.combo_left_click.findText(mode)
                if mode_idx >= 0: self.combo_left_click.setCurrentIndex(mode_idx)

                self.chk_normalize.setChecked(data.get("normalize", False))
                self.chk_clipboard.setChecked(data.get("clipboard", False))
                self.chk_delete.setChecked(data.get("delete_after", False))
                self.chk_keep_source_wav.setChecked(
                    data.get("keep_source_wav", False)
                )
                self.chk_delete.setEnabled(self.chk_clipboard.isChecked())
                self._update_source_wav_option()

                self.hk_mic.setText(data.get("hk_mic", ""))
                self.hk_loop.setText(data.get("hk_loop", ""))
                self.hk_both.setText(data.get("hk_both", ""))
                self.hk_stop.setText(data.get("hk_stop", ""))
            except Exception as e:
                logger.exception("Error loading settings")

    def save_settings(self, show_prompt=True):
        data = self.get_settings()
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f)
            logger.info("Settings saved")
            if show_prompt:
                QMessageBox.information(self, "Settings", "Settings saved successfully.")
            self.settings_saved.emit()
        except Exception as e:
            logger.exception("Failed to save settings")
            if show_prompt:
                QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")

    def get_settings(self):
        return {
            "device_id": self.combo_mic.currentData(),
            "device_name": self.combo_mic.currentText(),
            "loopback_id": self.combo_loopback.currentData(),
            "loopback_name": self.combo_loopback.currentText(),
            "output_folder": self.lbl_folder.text(),
            "log_folder": self.lbl_log_folder.text(),
            "format": self.combo_fmt.currentText(),
            "tray_click_mode": self.combo_left_click.currentText(),
            "normalize": self.chk_normalize.isChecked(),
            "clipboard": self.chk_clipboard.isChecked(),
            "delete_after": self.chk_delete.isChecked(),
            "keep_source_wav": self.chk_keep_source_wav.isChecked(),
            "hk_mic": self.hk_mic.text(),
            "hk_loop": self.hk_loop.text(),
            "hk_both": self.hk_both.text(),
            "hk_stop": self.hk_stop.text()
        }

    def set_selected_mic_by_id(self, device_id, device_name=None):
        idx = self.combo_mic.findData(device_id)
        if idx >= 0:
            self.combo_mic.setCurrentIndex(idx)
        elif device_name:
            idx = self.combo_mic.findText(device_name)
            if idx >= 0:
                self.combo_mic.setCurrentIndex(idx)
        self.save_settings(show_prompt=False)

    def set_selected_loopback_by_id(self, loopback_id, loopback_name=None):
        idx = self.combo_loopback.findData(loopback_id)
        if idx >= 0:
            self.combo_loopback.setCurrentIndex(idx)
        elif loopback_name:
            idx = self.combo_loopback.findText(loopback_name)
            if idx >= 0:
                self.combo_loopback.setCurrentIndex(idx)
        self.save_settings(show_prompt=False)

    def _on_mic_selection_changed(self):
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_microphone_test()

    def toggle_microphone_test(self):
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_microphone_test()
        else:
            self.start_microphone_test()

    def start_microphone_test(self):
        mic_id = self.combo_mic.currentData()
        if mic_id is None:
            QMessageBox.warning(self, "Microphone Test", "No microphone selected.")
            return

        tray_app = getattr(self, "tray_app", None)
        if tray_app and tray_app.is_recording():
            QMessageBox.warning(
                self,
                "Microphone Test",
                "Cannot test microphone while a recording is in progress.",
            )
            return

        self.stop_microphone_test()
        self.btn_test_mic.setText("Stop Test")
        self.btn_test_mic.setToolTip("Stop microphone testing")
        self.btn_test_mic.setStyleSheet("font-weight: bold; color: #DC2626;")
        self.spectrum_widget.set_active(True)

        def _on_levels(bands, rms):
            self.test_signals.levels_updated.emit(bands, rms)

        def _on_err(err_msg):
            self.test_signals.mic_error.emit(err_msg)

        self.monitor_thread = AudioMonitorThread(
            device_id=mic_id,
            num_bands=self.spectrum_widget.num_bands,
            callback=_on_levels,
            error_callback=_on_err,
        )
        self.monitor_thread.start()
        self.mic_test_timer.start()

    def stop_microphone_test(self):
        if self.mic_test_timer.isActive():
            self.mic_test_timer.stop()

        if self.monitor_thread:
            thread = self.monitor_thread
            self.monitor_thread = None
            thread.stop()

        self.btn_test_mic.setText("Test Mic")
        self.btn_test_mic.setToolTip(
            "Test selected microphone and display real-time frequency spectrum"
        )
        self.btn_test_mic.setStyleSheet("")
        self.spectrum_widget.set_active(False)

    def _on_spectrum_levels(self, bands, rms):
        self.spectrum_widget.update_levels(bands, rms)

    def _on_mic_test_error(self, err_msg):
        self.stop_microphone_test()
        QMessageBox.critical(
            self,
            "Microphone Test Error",
            f"Failed to test microphone:\n{err_msg}",
        )

    def play_system_audio_test(self):
        if self.is_playing_audio_test:
            return

        tray_app = getattr(self, "tray_app", None)
        if tray_app and tray_app.is_recording():
            QMessageBox.warning(
                self,
                "Audio Test",
                "Cannot test system audio while a recording is in progress.",
            )
            return

        loopback_id = self.combo_loopback.currentData()
        self.is_playing_audio_test = True
        self.btn_test_loopback.setEnabled(False)
        self.btn_test_loopback.setText("Playing...")

        def _play_worker():
            try:
                play_test_sound(loopback_id=loopback_id)
                self.test_signals.audio_test_finished.emit()
            except Exception as e:
                self.test_signals.audio_test_error.emit(str(e))

        threading.Thread(target=_play_worker, daemon=True).start()

    def _on_audio_test_finished(self):
        self.is_playing_audio_test = False
        self.btn_test_loopback.setEnabled(True)
        self.btn_test_loopback.setText("Test Audio")

    def _on_audio_test_error(self, err_msg):
        self.is_playing_audio_test = False
        self.btn_test_loopback.setEnabled(True)
        self.btn_test_loopback.setText("Test Audio")
        QMessageBox.critical(
            self,
            "Audio Test Error",
            f"Failed to play test sound:\n{err_msg}",
        )

    def closeEvent(self, event):
        self.stop_microphone_test()
        super().closeEvent(event)


class TrayApplication(QObject):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.recorder = None
        self.hotkey_handles = []
        self.last_mode = "mic" 
        
        self.signals = SignalManager()
        self.signals.recording_finished.connect(self.on_recording_finished)
        self.signals.recording_start_requested.connect(self.start_recording)
        self.signals.recording_stop_requested.connect(self.stop_recording)

        self.icon_idle_path = resource_path("icon_idle.png")
        self.icon_rec_path = resource_path("icon_rec.png")
        self.generate_icons()
            
        self.tray_icon = QSystemTrayIcon(QIcon(self.icon_idle_path), self.app)
        self.tray_icon.setToolTip("Simple Audio Recorder (Idle)")
        self.tray_icon.activated.connect(self.on_tray_activated)
        
        self.settings_window = SettingsWindow()
        self.settings_window.tray_app = self
        configure_logging(self.settings_window.get_settings()["log_folder"])
        self.settings_window.settings_saved.connect(self.on_settings_saved)

        self.build_menu()
        self.tray_icon.show()
        
        self.tray_icon.showMessage("Ready", "Left-click to toggle recording.", QSystemTrayIcon.MessageIcon.Information, 2000)
        self.register_hotkeys()

    def generate_icons(self):
        if not os.path.exists(self.icon_idle_path):
            pix = QPixmap(64, 64)
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(80, 80, 80)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(4, 4, 56, 56)
            painter.end()
            pix.save(self.icon_idle_path)

        if not os.path.exists(self.icon_rec_path):
            pix = QPixmap(64, 64)
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(220, 0, 0)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(4, 4, 56, 56)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(22, 22, 20, 20)
            painter.end()
            pix.save(self.icon_rec_path)

    def build_menu(self):
        self.menu = QMenu()
        self.action_record_mic = QAction("Start Recording (Mic)", self)
        self.action_record_mic.triggered.connect(lambda: self.start_recording("mic"))
        self.action_record_loop = QAction("Start Recording (Loopback)", self)
        self.action_record_loop.triggered.connect(lambda: self.start_recording("loopback"))
        self.action_record_both = QAction("Start Recording (Both)", self)
        self.action_record_both.triggered.connect(lambda: self.start_recording("both"))
        self.action_stop = QAction("Stop Recording", self)
        self.action_stop.triggered.connect(self.stop_recording)
        self.action_stop.setEnabled(False)

        # Audio Devices submenu
        self.menu_devices = QMenu("Audio Devices", self.menu)
        self.menu_mic = QMenu("Microphone", self.menu_devices)
        self.menu_loop = QMenu("System Audio", self.menu_devices)
        self.action_refresh_devices = QAction("Refresh Devices", self.menu_devices)
        self.action_refresh_devices.triggered.connect(self.refresh_devices_from_menu)

        self.menu_devices.addMenu(self.menu_mic)
        self.menu_devices.addMenu(self.menu_loop)
        self.menu_devices.addSeparator()
        self.menu_devices.addAction(self.action_refresh_devices)

        # Populate devices when menu is about to show
        self.menu_devices.aboutToShow.connect(self.populate_device_menus)
        self.menu.aboutToShow.connect(self.populate_device_menus)

        # Direct Refresh Devices action in main menu for 1-click access
        self.action_refresh_main = QAction("Refresh Devices", self.menu)
        self.action_refresh_main.triggered.connect(self.refresh_devices_from_menu)

        self.action_settings = QAction("Settings", self)
        self.action_settings.triggered.connect(self.open_settings)
        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.exit_app)
        
        self.menu.addAction(self.action_record_mic)
        self.menu.addAction(self.action_record_loop)
        self.menu.addAction(self.action_record_both)
        self.menu.addAction(self.action_stop)
        self.menu.addSeparator()
        self.menu.addMenu(self.menu_devices)
        self.menu.addAction(self.action_refresh_main)
        self.menu.addSeparator()
        self.menu.addAction(self.action_settings)
        self.menu.addAction(self.action_exit)
        self.tray_icon.setContextMenu(self.menu)

        self.populate_device_menus()

    def populate_device_menus(self):
        if not hasattr(self, "menu_mic") or not hasattr(self, "menu_loop"):
            return

        self.menu_mic.clear()
        self.menu_loop.clear()

        mic_group = QActionGroup(self.menu_mic)
        mic_group.setExclusive(True)

        cur_mic_id = self.settings_window.combo_mic.currentData()
        cur_mic_name = self.settings_window.combo_mic.currentText()
        mic_count = self.settings_window.combo_mic.count()

        if mic_count == 0:
            act_none = QAction("No microphones detected", self.menu_mic)
            act_none.setEnabled(False)
            self.menu_mic.addAction(act_none)
        else:
            for i in range(mic_count):
                name = self.settings_window.combo_mic.itemText(i)
                dev_id = self.settings_window.combo_mic.itemData(i)
                act = QAction(name, self.menu_mic)
                act.setCheckable(True)
                if (cur_mic_id is not None and dev_id == cur_mic_id) or (name == cur_mic_name):
                    act.setChecked(True)
                act.triggered.connect(
                    lambda checked, did=dev_id, dname=name: self.on_mic_device_selected(did, dname)
                )
                mic_group.addAction(act)
                self.menu_mic.addAction(act)

        loop_group = QActionGroup(self.menu_loop)
        loop_group.setExclusive(True)

        cur_loop_id = self.settings_window.combo_loopback.currentData()
        cur_loop_name = self.settings_window.combo_loopback.currentText()
        loop_count = self.settings_window.combo_loopback.count()

        if loop_count == 0:
            act_none = QAction("No system audio devices detected", self.menu_loop)
            act_none.setEnabled(False)
            self.menu_loop.addAction(act_none)
        else:
            for i in range(loop_count):
                name = self.settings_window.combo_loopback.itemText(i)
                dev_id = self.settings_window.combo_loopback.itemData(i)
                act = QAction(name, self.menu_loop)
                act.setCheckable(True)
                if (cur_loop_id is not None and dev_id == cur_loop_id) or (name == cur_loop_name):
                    act.setChecked(True)
                act.triggered.connect(
                    lambda checked, did=dev_id, dname=name: self.on_loopback_device_selected(did, dname)
                )
                loop_group.addAction(act)
                self.menu_loop.addAction(act)

    def on_mic_device_selected(self, dev_id, dev_name):
        logger.info("Microphone selected from tray: %s (id=%s)", dev_name, dev_id)
        self.settings_window.set_selected_mic_by_id(dev_id, dev_name)
        self.populate_device_menus()
        self.tray_icon.showMessage(
            "Microphone Selected",
            dev_name,
            QSystemTrayIcon.MessageIcon.Information,
            1200,
        )

    def on_loopback_device_selected(self, dev_id, dev_name):
        logger.info("System audio selected from tray: %s (id=%s)", dev_name, dev_id)
        self.settings_window.set_selected_loopback_by_id(dev_id, dev_name)
        self.populate_device_menus()
        self.tray_icon.showMessage(
            "System Audio Selected",
            dev_name,
            QSystemTrayIcon.MessageIcon.Information,
            1200,
        )

    def refresh_devices_from_menu(self):
        logger.info("Refreshing devices requested from tray menu")
        if self.settings_window:
            self.settings_window.refresh_devices(preserve_selection=True)
            self.populate_device_menus()
            self.tray_icon.showMessage(
                "Devices Refreshed",
                "Audio devices updated successfully.",
                QSystemTrayIcon.MessageIcon.Information,
                1200,
            )

    def register_hotkeys(self):
        self._clear_hotkeys()
        settings = self.settings_window.get_settings()
        hk_mic = settings.get("hk_mic")
        hk_loop = settings.get("hk_loop")
        hk_both = settings.get("hk_both")
        hk_stop = settings.get("hk_stop")
        try:
            if hk_mic:
                self.hotkey_handles.append(
                    keyboard.add_hotkey(
                        hk_mic,
                        lambda: self.signals.recording_start_requested.emit("mic"),
                    )
                )
            if hk_loop:
                self.hotkey_handles.append(
                    keyboard.add_hotkey(
                        hk_loop,
                        lambda: self.signals.recording_start_requested.emit("loopback"),
                    )
                )
            if hk_both:
                self.hotkey_handles.append(
                    keyboard.add_hotkey(
                        hk_both,
                        lambda: self.signals.recording_start_requested.emit("both"),
                    )
                )
            if hk_stop:
                self.hotkey_handles.append(
                    keyboard.add_hotkey(
                        hk_stop,
                        lambda: self.signals.recording_stop_requested.emit(),
                    )
                )
        except Exception:
            logger.exception("Failed to register hotkeys")

    def _clear_hotkeys(self):
        for handle in self.hotkey_handles:
            try:
                keyboard.remove_hotkey(handle)
            except Exception:
                logger.exception("Failed to remove hotkey")
        self.hotkey_handles = []

    def on_settings_saved(self):
        log_path = configure_logging(self.settings_window.get_settings()["log_folder"])
        logger.info("Logging reconfigured: %s", log_path)
        self.register_hotkeys()
        self.populate_device_menus()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.recorder and self.recorder.is_alive():
                self.stop_recording()
            else:
                settings = self.settings_window.get_settings()
                click_mode = settings.get("tray_click_mode", "Last Used")
                target_mode = self.last_mode
                if click_mode == "Microphone": target_mode = "mic"
                elif click_mode == "Loopback": target_mode = "loopback"
                elif click_mode == "Both": target_mode = "both"
                self.start_recording(target_mode)

    def is_recording(self):
        return bool(self.recorder and self.recorder.is_alive())

    def open_settings(self):
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def start_recording(self, mode="mic"):
        if self.recorder and self.recorder.is_alive(): return
        if self.settings_window:
            self.settings_window.stop_microphone_test()
            self.settings_window.refresh_devices(preserve_selection=True)
            self.populate_device_menus()
        self.last_mode = mode    
        settings = self.settings_window.get_settings()
        target_id = settings['device_id']
        target_loopback_id = settings['loopback_id']
        logger.info(
            "Starting recording request: mode=%s (mic_id=%s, loopback_id=%s)",
            mode,
            target_id,
            target_loopback_id,
        )
        
        def finish_callback(path, error):
            self.signals.recording_finished.emit(path if path else "", error if error else "")

        self.recorder = AudioRecorder(
            mic_id=target_id,
            source_mode=mode,
            output_folder=settings['output_folder'],
            output_format=settings['format'],
            normalize=settings['normalize'],
            on_finish_callback=finish_callback,
            loopback_id=target_loopback_id,
            keep_source_wav=settings['keep_source_wav'],
        )
        self.recorder.start()
        self.action_record_mic.setEnabled(False)
        self.action_record_loop.setEnabled(False)
        self.action_record_both.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.tray_icon.setIcon(QIcon(self.icon_rec_path)) 
        self.tray_icon.setToolTip(f"Recording ({mode})...")
        self.tray_icon.showMessage("Started", f"Recording {mode}", QSystemTrayIcon.MessageIcon.NoIcon, 1000)

    def stop_recording(self):
        if self.recorder and self.recorder.is_alive():
            logger.info("Stopping recording")
            self.recorder.stop()
            self.action_stop.setEnabled(False)
            self.tray_icon.setToolTip("Saving recording...")
            self.tray_icon.showMessage(
                "Saving",
                "Finalizing recording...",
                QSystemTrayIcon.MessageIcon.NoIcon,
                1000,
            )

    def on_recording_finished(self, path, error):
        self.action_record_mic.setEnabled(True)
        self.action_record_loop.setEnabled(True)
        self.action_record_both.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.tray_icon.setIcon(QIcon(self.icon_idle_path))
        self.tray_icon.setToolTip("Simple Audio Recorder (Idle)")
        self.recorder = None
        
        if error:
            logger.error("Recording failed: %s", error)
            self.tray_icon.showMessage("Error", f"Recording failed: {error}", QSystemTrayIcon.MessageIcon.Critical, 4000)
            return
            
        settings = self.settings_window.get_settings()
        final_path = path
        folder_name = os.path.basename(os.path.dirname(path))
        msg = f"Saved in {folder_name}: {os.path.basename(path)}"
        
        if settings['clipboard'] and os.path.exists(path):
            clipboard_started = time.perf_counter()
            try:
                if settings['delete_after']:
                    temp_dir = tempfile.gettempdir()
                    new_path = os.path.join(temp_dir, os.path.basename(path))
                    if os.path.exists(new_path):
                        base, ext = os.path.splitext(new_path)
                        import time
                        new_path = f"{base}_{int(time.time())}{ext}"
                    shutil.move(path, new_path)
                    final_path = new_path
                    msg = "Moved to Temp & Copied to Clipboard."
                else:
                    msg += "\nCopied to clipboard."

                # Use Robust Clipboard Utility
                success, status = copy_file_to_clipboard(final_path)
                if not success:
                    logger.error("Clipboard copy failed: %s", status)
                    msg += f"\nClipboard Error: {status}"
                
            except Exception as e:
                logger.exception("Clipboard or file move failed")
                msg += f"\nClipboard/Move error: {e}"
            finally:
                logger.info(
                    "Clipboard processing completed in %.3fs",
                    time.perf_counter() - clipboard_started,
                )

        logger.info("Recording finished: %s", path)
        self.tray_icon.showMessage("Finished", msg, QSystemTrayIcon.MessageIcon.Information, 2000)

    def exit_app(self):
        logger.info("Application exiting")
        self._clear_hotkeys()
        if self.settings_window:
            self.settings_window.stop_microphone_test()
        if self.recorder: self.recorder.stop()
        self.app.quit()
