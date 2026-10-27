import sys
import os
import json
import logging
import threading
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QMainWindow,
                             QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QPushButton, QFileDialog, QMessageBox, QGroupBox, 
                             QLineEdit, QFormLayout, QSizePolicy, QDialog,
                             QDialogButtonBox, QGridLayout)
from PyQt6.QtGui import (QIcon, QAction, QColor, QPixmap, QPainter, QBrush,
                         QKeySequence, QPen, QLinearGradient, QActionGroup)
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QUrl, QMimeData, QDir, QTimer, QRectF
import keyboard
import numpy as np
from audio_recorder import (
    AudioRecorder,
    AudioMonitorThread,
    get_devices,
    play_test_sound,
)
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
    recording_finished = pyqtSignal(str, str, object)
    capture_finished = pyqtSignal(object)
    recording_levels = pyqtSignal(str, list, float)
    recording_device_error = pyqtSignal(str, str)
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
    audio_test_finished = pyqtSignal()
    audio_test_error = pyqtSignal(str)
    mic_levels_updated = pyqtSignal(list, float)
    mic_test_error = pyqtSignal(str)


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
                "Recording levels are shown during capture",
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


class RecordingDeviceDialog(QDialog):
    def __init__(
        self,
        mode,
        mic_id,
        mic_name,
        loopback_id,
        loopback_name,
        parent=None,
        validation_error=None,
    ):
        super().__init__(parent)
        self.mode = mode
        self._mic_id = mic_id
        self._mic_name = mic_name
        self._loopback_id = loopback_id
        self._loopback_name = loopback_name
        self.setWindowTitle("Confirm recording devices")
        self.setModal(True)

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.addWidget(QLabel("Microphone input:"), 0, 0)
        self.combo_mic = QComboBox()
        grid.addWidget(self.combo_mic, 0, 1)
        grid.addWidget(QLabel("System audio output:"), 1, 0)
        self.combo_loopback = QComboBox()
        grid.addWidget(self.combo_loopback, 1, 1)
        layout.addLayout(grid)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self.refresh_devices)
        layout.addWidget(refresh)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.combo_mic.currentIndexChanged.connect(self._selection_changed)
        self.combo_loopback.currentIndexChanged.connect(self._selection_changed)
        self.refresh_devices()
        if validation_error:
            self.status_label.setText(validation_error)

    def refresh_devices(self):
        self._remember_selection()
        self.combo_mic.blockSignals(True)
        self.combo_loopback.blockSignals(True)
        self.combo_mic.clear()
        self.combo_loopback.clear()
        try:
            mics = get_devices(include_loopback=False)
            loopbacks = get_devices(include_loopback=True)
            self._populate_combo(
                self.combo_mic,
                mics,
                self._mic_id,
                self._mic_name,
                "No microphone available",
            )
            self._populate_combo(
                self.combo_loopback,
                loopbacks,
                self._loopback_id,
                self._loopback_name,
                "No system-audio output available",
            )
        except Exception as error:
            logger.exception("Error refreshing recording devices")
            self.combo_mic.addItem("Unable to list microphones")
            self.combo_loopback.addItem("Unable to list system-audio outputs")
            self.status_label.setText(f"Unable to refresh devices: {error}")
        finally:
            self.combo_mic.blockSignals(False)
            self.combo_loopback.blockSignals(False)

        mic_required = self.mode in ("mic", "both")
        loopback_required = self.mode in ("loopback", "both")
        self.combo_mic.setEnabled(mic_required)
        self.combo_loopback.setEnabled(loopback_required)
        self._update_validation()

    @staticmethod
    def _populate_combo(combo, devices, selected_id, selected_name, empty_text):
        for device in devices:
            combo.addItem(device["name"], device["id"])
        selected_index = -1
        if selected_id is not None:
            selected_index = combo.findData(selected_id)
        if selected_index >= 0:
            combo.setCurrentIndex(selected_index)
        elif not devices:
            combo.addItem(empty_text, None)
        else:
            combo.setCurrentIndex(-1)

    def _remember_selection(self):
        if self.combo_mic.currentData() is not None:
            self._mic_id = self.combo_mic.currentData()
            self._mic_name = self.combo_mic.currentText()
        if self.combo_loopback.currentData() is not None:
            self._loopback_id = self.combo_loopback.currentData()
            self._loopback_name = self.combo_loopback.currentText()

    def _selection_changed(self, _index):
        self._remember_selection()
        self._update_validation()

    def _update_validation(self):
        missing = []
        if self.mode in ("mic", "both") and self.combo_mic.currentData() is None:
            missing.append("microphone input")
        if self.mode in ("loopback", "both") and self.combo_loopback.currentData() is None:
            missing.append("system-audio output")
        if missing:
            self.status_label.setText(
                "Select an available " + " and ".join(missing) + " before confirming."
            )
        elif not self.status_label.text().startswith("Unable to"):
            self.status_label.setText("Confirm the devices for this recording.")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not missing)

    def _accept_if_valid(self):
        self._remember_selection()
        if self.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled():
            self.accept()

    def selected_devices(self):
        self._remember_selection()
        return self._mic_id, self._loopback_id


class SettingsWindow(QMainWindow):
    settings_saved = pyqtSignal()
    device_selection_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings - Simple Audio Recorder")
        self.setWindowIcon(QIcon(resource_path("app_icon.ico")))
        self.setGeometry(100, 100, 520, 670)

        self.tray_app = None
        self.is_playing_audio_test = False
        self.monitor_thread = None
        self.mic_test_timer = QTimer(self)
        self.mic_test_timer.setSingleShot(True)
        self.mic_test_timer.setInterval(15000)
        self.mic_test_timer.timeout.connect(self.stop_microphone_test)

        self.test_signals = TestSignals()
        self.test_signals.audio_test_finished.connect(self._on_audio_test_finished)
        self.test_signals.audio_test_error.connect(self._on_audio_test_error)
        self.test_signals.mic_levels_updated.connect(self._on_mic_spectrum_levels)
        self.test_signals.mic_test_error.connect(self._on_mic_test_error)

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
        self.combo_mic.currentIndexChanged.connect(
            self._on_mic_selection_changed
        )
        row_mic.addWidget(lbl_mic)
        row_mic.addWidget(self.combo_mic, 1)
        self.btn_test_mic = QPushButton("Test Mic")
        self.btn_test_mic.setFixedWidth(90)
        self.btn_test_mic.setToolTip(
            "Listen to the microphone input level for 15 seconds"
        )
        self.btn_test_mic.clicked.connect(self.toggle_microphone_test)
        row_mic.addWidget(self.btn_test_mic)
        layout_mic.addLayout(row_mic)

        # System audio row
        row_loop = QHBoxLayout()
        lbl_loop = QLabel("System audio:")
        lbl_loop.setFixedWidth(85)
        self.combo_loopback = QComboBox()
        self.combo_loopback.currentIndexChanged.connect(
            lambda _index: self._on_device_selection_changed("SystemAudio")
        )
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

        # Recording level visualizers. They are shown only while recording.
        self.spectrum_mic = AudioSpectrumWidget()
        self.spectrum_loopback = AudioSpectrumWidget()
        self.recording_levels_panel = QWidget()
        layout_levels = QVBoxLayout(self.recording_levels_panel)
        layout_levels.setContentsMargins(0, 4, 0, 0)
        self.lbl_spectrum_mic = QLabel("Microphone capture")
        self.lbl_spectrum_loopback = QLabel("System audio capture")
        layout_levels.addWidget(self.lbl_spectrum_mic)
        layout_levels.addWidget(self.spectrum_mic)
        layout_levels.addWidget(self.lbl_spectrum_loopback)
        layout_levels.addWidget(self.spectrum_loopback)
        self.recording_levels_panel.setVisible(False)
        layout_mic.addWidget(self.recording_levels_panel)

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
        group_tray.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        layout.addWidget(group_tray)

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
        layout.addStretch(1)

        self.refresh_devices()

    def refresh_devices(self, preserve_selection=True):
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_microphone_test()

        prev_mic_id = self.combo_mic.currentData() if preserve_selection else None
        prev_loop_id = self.combo_loopback.currentData() if preserve_selection else None

        self.combo_mic.blockSignals(True)
        self.combo_loopback.blockSignals(True)
        self.combo_mic.clear()
        self.combo_loopback.clear()
        try:
            mics = get_devices(include_loopback=False)
            loopbacks = get_devices(include_loopback=True)

            mic_index_to_select = -1
            for i, m in enumerate(mics):
                self.combo_mic.addItem(f"{m['name']}", m['id'])
                if prev_mic_id is not None and m['id'] == prev_mic_id:
                    mic_index_to_select = i

            if mic_index_to_select >= 0:
                self.combo_mic.setCurrentIndex(mic_index_to_select)

            loop_index_to_select = -1
            for i, device in enumerate(loopbacks):
                self.combo_loopback.addItem(device["name"], device["id"])
                if prev_loop_id is not None and device["id"] == prev_loop_id:
                    loop_index_to_select = i

            if loop_index_to_select >= 0:
                self.combo_loopback.setCurrentIndex(loop_index_to_select)

        except Exception as e:
            logger.exception("Error refreshing devices")
        finally:
            self.combo_mic.blockSignals(False)
            self.combo_loopback.blockSignals(False)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", options=QFileDialog.Option.DontUseNativeDialog)
        if folder:
            self.lbl_folder.setText(folder)

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
                if saved_id is not None:
                    idx = self.combo_mic.findData(saved_id)
                    if idx >= 0:
                        self.combo_mic.setCurrentIndex(idx)

                saved_loopback_id = data.get("loopback_id")
                if saved_loopback_id is not None:
                    idx = self.combo_loopback.findData(saved_loopback_id)
                    if idx >= 0:
                        self.combo_loopback.setCurrentIndex(idx)

                mode = data.get("tray_click_mode", "Last Used")
                mode_idx = self.combo_left_click.findText(mode)
                if mode_idx >= 0: self.combo_left_click.setCurrentIndex(mode_idx)

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

    def apply_recording_devices(self, mic_id, loopback_id):
        self.combo_mic.blockSignals(True)
        self.combo_loopback.blockSignals(True)
        try:
            mic_index = self.combo_mic.findData(mic_id)
            loopback_index = self.combo_loopback.findData(loopback_id)
            if mic_index >= 0:
                self.combo_mic.setCurrentIndex(mic_index)
            if loopback_index >= 0:
                self.combo_loopback.setCurrentIndex(loopback_index)
        finally:
            self.combo_mic.blockSignals(False)
            self.combo_loopback.blockSignals(False)
        self.save_settings(show_prompt=False)

    def _on_device_selection_changed(self, source_name):
        self.device_selection_changed.emit(source_name)

    def _on_mic_selection_changed(self, index):
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_microphone_test()
        self._on_device_selection_changed("Microphone")

    def start_recording_levels(self, mode):
        mic_active = mode in ("mic", "both")
        loopback_active = mode in ("loopback", "both")
        self.stop_microphone_test()
        self.lbl_spectrum_mic.setVisible(mic_active)
        self.spectrum_mic.setVisible(mic_active)
        self.lbl_spectrum_loopback.setVisible(loopback_active)
        self.spectrum_loopback.setVisible(loopback_active)
        self.spectrum_mic.set_active(mic_active)
        self.spectrum_loopback.set_active(loopback_active)
        self.recording_levels_panel.setVisible(True)

    def stop_recording_levels(self):
        self.spectrum_mic.set_active(False)
        self.spectrum_loopback.set_active(False)
        self.recording_levels_panel.setVisible(False)

    def update_recording_levels(self, source_name, bands, rms):
        if source_name == "Microphone":
            self.spectrum_mic.update_levels(bands, rms)
        elif source_name == "SystemAudio":
            self.spectrum_loopback.update_levels(bands, rms)

    def toggle_microphone_test(self):
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_microphone_test()
        else:
            self.start_microphone_test()

    def start_microphone_test(self):
        tray_app = getattr(self, "tray_app", None)
        if tray_app and tray_app.is_recording():
            QMessageBox.warning(
                self,
                "Microphone Test",
                "Cannot test the microphone while a recording is in progress.",
            )
            return

        mic_id = self.combo_mic.currentData()
        if mic_id is None:
            QMessageBox.warning(
                self,
                "Microphone Test",
                "Select a microphone before starting the test.",
            )
            return

        self.stop_microphone_test()
        self.btn_test_mic.setText("Stop Test")
        self.btn_test_mic.setStyleSheet("background-color: #7f1d1d;")
        self.lbl_spectrum_mic.setVisible(True)
        self.spectrum_mic.setVisible(True)
        self.lbl_spectrum_loopback.setVisible(False)
        self.spectrum_loopback.setVisible(False)
        self.recording_levels_panel.setVisible(True)
        self.spectrum_mic.set_active(True)

        self.monitor_thread = AudioMonitorThread(
            device_id=mic_id,
            num_bands=self.spectrum_mic.num_bands,
            callback=lambda bands, rms: self.test_signals.mic_levels_updated.emit(
                bands, rms
            ),
            error_callback=lambda err: self.test_signals.mic_test_error.emit(err),
        )
        self.monitor_thread.start()
        self.mic_test_timer.start()

    def stop_microphone_test(self):
        self.mic_test_timer.stop()
        monitor = self.monitor_thread
        self.monitor_thread = None
        if monitor and monitor.is_alive():
            monitor.stop()
        self.btn_test_mic.setText("Test Mic")
        self.btn_test_mic.setStyleSheet("")
        is_recording = bool(self.tray_app and self.tray_app.is_recording())
        if not is_recording:
            self.spectrum_mic.set_active(False)
            self.lbl_spectrum_mic.setVisible(True)
            self.spectrum_mic.setVisible(True)
            self.lbl_spectrum_loopback.setVisible(True)
            self.spectrum_loopback.setVisible(True)
            self.recording_levels_panel.setVisible(False)

    def _on_mic_spectrum_levels(self, bands, rms):
        self.spectrum_mic.update_levels(bands, rms)

    def _on_mic_test_error(self, err_msg):
        self.stop_microphone_test()
        QMessageBox.critical(
            self,
            "Microphone Test Error",
            f"Failed to test the microphone:\n{err_msg}",
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
        if not self.tray_app or not self.tray_app.is_recording():
            self.stop_recording_levels()
        super().closeEvent(event)


class TrayApplication(QObject):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.recorder = None
        self.saving_recorders = set()
        self.hotkey_handles = []
        self.last_mode = "mic"
        
        self.signals = SignalManager()
        self.signals.recording_finished.connect(self.on_recording_finished)
        self.signals.capture_finished.connect(self.on_capture_finished)
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
        self.settings_window.device_selection_changed.connect(
            self.on_device_selection_changed
        )
        self.signals.recording_levels.connect(
            self.settings_window.update_recording_levels
        )
        self.signals.recording_device_error.connect(self.on_recording_device_error)
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

        # Audio device selection remains available directly from the tray.
        self.menu_devices = QMenu("Audio Devices", self.menu)
        self.menu_mic = QMenu("Microphone", self.menu_devices)
        self.menu_loop = QMenu("System Audio", self.menu_devices)
        self.action_refresh_devices = QAction("Refresh Devices", self.menu_devices)
        self.action_refresh_devices.triggered.connect(self.refresh_devices_from_menu)
        self.menu_devices.addMenu(self.menu_mic)
        self.menu_devices.addMenu(self.menu_loop)
        self.menu_devices.addSeparator()
        self.menu_devices.addAction(self.action_refresh_devices)
        self.menu_devices.aboutToShow.connect(self.populate_device_menus)
        self.menu.aboutToShow.connect(self.populate_device_menus)

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
        self.menu.addAction(self.action_settings)
        self.menu.addAction(self.action_exit)
        self.tray_icon.setContextMenu(self.menu)

        self.populate_device_menus()

    def populate_device_menus(self):
        mic_group = QActionGroup(self.menu_mic)
        mic_group.setExclusive(True)
        self.menu_mic.clear()

        current_mic_id = self.settings_window.combo_mic.currentData()
        current_mic_name = self.settings_window.combo_mic.currentText()
        if self.settings_window.combo_mic.count() == 0:
            action = QAction("No microphones detected", self.menu_mic)
            action.setEnabled(False)
            self.menu_mic.addAction(action)
        else:
            for index in range(self.settings_window.combo_mic.count()):
                name = self.settings_window.combo_mic.itemText(index)
                device_id = self.settings_window.combo_mic.itemData(index)
                action = QAction(name, self.menu_mic)
                action.setData(device_id)
                action.setCheckable(True)
                action.setChecked(
                    (current_mic_id is not None and device_id == current_mic_id)
                    or name == current_mic_name
                )
                action.triggered.connect(
                    lambda checked, did=device_id, dname=name:
                    self.on_mic_device_selected(did, dname)
                )
                mic_group.addAction(action)
                self.menu_mic.addAction(action)

        loop_group = QActionGroup(self.menu_loop)
        loop_group.setExclusive(True)
        self.menu_loop.clear()

        current_loop_id = self.settings_window.combo_loopback.currentData()
        current_loop_name = self.settings_window.combo_loopback.currentText()
        if self.settings_window.combo_loopback.count() == 0:
            action = QAction("No system audio devices detected", self.menu_loop)
            action.setEnabled(False)
            self.menu_loop.addAction(action)
        else:
            for index in range(self.settings_window.combo_loopback.count()):
                name = self.settings_window.combo_loopback.itemText(index)
                device_id = self.settings_window.combo_loopback.itemData(index)
                action = QAction(name, self.menu_loop)
                action.setData(device_id)
                action.setCheckable(True)
                action.setChecked(
                    (current_loop_id is not None and device_id == current_loop_id)
                    or name == current_loop_name
                )
                action.triggered.connect(
                    lambda checked, did=device_id, dname=name:
                    self.on_loopback_device_selected(did, dname)
                )
                loop_group.addAction(action)
                self.menu_loop.addAction(action)

    def on_mic_device_selected(self, device_id, device_name):
        logger.info("Microphone selected from tray: %s (id=%s)", device_name, device_id)
        self.settings_window.set_selected_mic_by_id(device_id, device_name)
        self._mark_selected_device(self.menu_mic, device_id, device_name)

    def on_loopback_device_selected(self, device_id, device_name):
        logger.info("System audio selected from tray: %s (id=%s)", device_name, device_id)
        self.settings_window.set_selected_loopback_by_id(device_id, device_name)
        self._mark_selected_device(self.menu_loop, device_id, device_name)

    def on_device_selection_changed(self, source_name):
        if not self.is_recording():
            return
        self.settings_window.save_settings(show_prompt=False)
        if (
            source_name == "Microphone"
            and self.last_mode not in ("mic", "both")
        ) or (
            source_name == "SystemAudio"
            and self.last_mode not in ("loopback", "both")
        ):
            return
        self.request_device_switch(source_name)

    @staticmethod
    def _mark_selected_device(menu, device_id, device_name):
        for action in menu.actions():
            if action.data() == device_id or action.text() == device_name:
                action.setChecked(True)
            elif action.isCheckable():
                action.setChecked(False)

    def refresh_devices_from_menu(self):
        logger.info("Refreshing devices requested from tray menu")
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

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.is_recording():
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
        return bool(
            self.recorder
            and getattr(self.recorder, "is_capturing", self.recorder.is_alive())
        )

    def open_settings(self):
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def start_recording(self, mode="mic"):
        if self.is_recording():
            return
        self.settings_window.stop_microphone_test()
        settings = self.settings_window.get_settings()
        validation_error = None
        
        recorder_instance = None

        def capture_callback():
            self.signals.capture_finished.emit(recorder_instance)

        def finish_callback(path, error):
            self.signals.recording_finished.emit(
                path if path else "",
                error if error else "",
                recorder_instance,
            )

        def levels_callback(source_name, bands, rms):
            self.signals.recording_levels.emit(source_name, bands, rms)

        def source_error_callback(source_name, message):
            self.signals.recording_device_error.emit(source_name, message)

        while True:
            dialog = RecordingDeviceDialog(
                mode,
                settings.get("device_id"),
                settings.get("device_name"),
                settings.get("loopback_id"),
                settings.get("loopback_name"),
                self.settings_window,
                validation_error,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            target_id, target_loopback_id = dialog.selected_devices()
            self.settings_window.apply_recording_devices(
                target_id, target_loopback_id
            )
            settings = self.settings_window.get_settings()
            recorder_instance = AudioRecorder(
                mic_id=settings["device_id"],
                source_mode=mode,
                output_folder=settings["output_folder"],
                output_format=settings["format"],
                on_finish_callback=finish_callback,
                on_capture_stopped_callback=capture_callback,
                on_levels_callback=levels_callback,
                on_source_error_callback=source_error_callback,
                loopback_id=settings["loopback_id"],
            )
            try:
                recorder_instance.validate_devices()
            except Exception as error:
                validation_error = (
                    "The selected device is no longer available. "
                    f"Choose another device: {error}"
                )
                continue
            break

        self.last_mode = mode
        logger.info(
            "Starting recording request: mode=%s (mic_id=%s, loopback_id=%s)",
            mode,
            settings["device_id"],
            settings["loopback_id"],
        )
        self.recorder = recorder_instance
        self.settings_window.start_recording_levels(mode)
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
            rec = self.recorder
            self.action_stop.setEnabled(False)
            self.tray_icon.setToolTip("Stopping capture...")
            rec.stop()

    def request_device_switch(self, source_name):
        if not self.is_recording() or not self.recorder:
            return
        settings = self.settings_window.get_settings()
        device_id = (
            settings["device_id"]
            if source_name == "Microphone"
            else settings["loopback_id"]
        )
        if self.recorder.switch_device(source_name, device_id):
            logger.info(
                "Hot-switched %s to device %s without stopping recording",
                source_name,
                device_id,
            )
            return
        self.tray_icon.showMessage(
            "Device Switch Failed",
            "The selected audio device could not be activated.",
            QSystemTrayIcon.MessageIcon.Warning,
            2500,
        )

    def on_capture_finished(self, rec):
        self.settings_window.stop_recording_levels()
        logger.info(
            "Capture stopped for recording: %s",
            getattr(rec, "recording_folder", "unknown"),
        )
        if self.recorder is rec:
            self.recorder = None
        if rec is not None:
            self.saving_recorders.add(rec)

        if not self.is_recording():
            self.action_record_mic.setEnabled(True)
            self.action_record_loop.setEnabled(True)
            self.action_record_both.setEnabled(True)
            self.action_stop.setEnabled(False)
            self.tray_icon.setIcon(QIcon(self.icon_idle_path))
            if self.saving_recorders:
                count = len(self.saving_recorders)
                self.tray_icon.setToolTip(
                    f"Saving {count} recording{'s' if count > 1 else ''} in background..."
                )
            else:
                self.tray_icon.setToolTip("Simple Audio Recorder (Idle)")
            self.tray_icon.showMessage(
                "Saving",
                "Recording stopped. Saving in background...",
                QSystemTrayIcon.MessageIcon.NoIcon,
                1000,
            )

    def on_recording_finished(self, path, error, rec=None):
        if rec in self.saving_recorders:
            self.saving_recorders.discard(rec)
        if self.recorder is rec:
            self.recorder = None

        if not self.is_recording():
            self.action_record_mic.setEnabled(True)
            self.action_record_loop.setEnabled(True)
            self.action_record_both.setEnabled(True)
            self.action_stop.setEnabled(False)
            self.tray_icon.setIcon(QIcon(self.icon_idle_path))
            if self.saving_recorders:
                count = len(self.saving_recorders)
                self.tray_icon.setToolTip(
                    f"Saving {count} recording{'s' if count > 1 else ''} in background..."
                )
            else:
                self.tray_icon.setToolTip("Simple Audio Recorder (Idle)")
        else:
            self.tray_icon.setIcon(QIcon(self.icon_rec_path))
            self.tray_icon.setToolTip(f"Recording ({self.last_mode})...")

        if error:
            logger.error("Recording failed: %s", error)
            self.tray_icon.showMessage("Error", f"Recording failed: {error}", QSystemTrayIcon.MessageIcon.Critical, 4000)
            return
            
        msg = ""
        if path and os.path.exists(path):
            folder_name = os.path.basename(os.path.dirname(path))
            msg = f"Saved in {folder_name}: {os.path.basename(path)}"

            logger.info("Recording segment saved: %s", path)

        self.tray_icon.showMessage("Finished", msg, QSystemTrayIcon.MessageIcon.Information, 2000)

    def on_recording_device_error(self, source_name, message):
        logger.warning("%s capture is unavailable: %s", source_name, message)
        self.tray_icon.showMessage(
            f"{source_name} unavailable",
            f"{message}\nSelect another device manually.",
            QSystemTrayIcon.MessageIcon.Warning,
            4000,
        )

    def exit_app(self):
        logger.info("Application exiting")
        self._clear_hotkeys()
        if self.settings_window:
            self.settings_window.stop_recording_levels()
        if self.recorder and self.recorder.is_alive():
            self.recorder.stop()
            self.recorder.join(timeout=3.0)
        for rec in list(self.saving_recorders):
            if rec.is_alive():
                logger.info("Waiting for background saving to complete...")
                rec.join(timeout=10.0)
        self.app.quit()
