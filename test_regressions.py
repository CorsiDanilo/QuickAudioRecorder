import ast
import importlib
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from pathlib import Path


def _load_modules():
    for name in ("soundcard", "soundfile", "lameenc", "numpy"):
        sys.modules.setdefault(name, types.ModuleType(name))

    try:
        import PyQt6.QtCore  # noqa: F401
        import PyQt6.QtWidgets  # noqa: F401
    except ImportError:
        qt_widgets = types.ModuleType("PyQt6.QtWidgets")
        qt_widgets.QApplication = type("QApplication", (), {})
        qt_core = types.ModuleType("PyQt6.QtCore")
        qt_core.QUrl = type("QUrl", (), {})
        qt_core.QMimeData = type("QMimeData", (), {})
        pyqt6 = types.ModuleType("PyQt6")
        pyqt6.QtWidgets = qt_widgets
        pyqt6.QtCore = qt_core
        sys.modules.setdefault("PyQt6", pyqt6)
        sys.modules.setdefault("PyQt6.QtWidgets", qt_widgets)
        sys.modules.setdefault("PyQt6.QtCore", qt_core)

    clipboard_utils = importlib.import_module("clipboard_utils")
    audio_recorder = importlib.import_module("audio_recorder")
    return clipboard_utils, audio_recorder


clipboard_utils, audio_recorder = _load_modules()


class RegressionTests(unittest.TestCase):
    def test_clipboard_path_is_passed_as_data_not_powershell_code(self):
        with tempfile.TemporaryDirectory() as folder:
            filepath = os.path.join(folder, "recording's.wav")
            open(filepath, "wb").close()
            captured = {}

            def run(command, **kwargs):
                captured["command"] = command
                captured["kwargs"] = kwargs
                return types.SimpleNamespace(returncode=0, stderr="")

            with patch.object(clipboard_utils.subprocess, "run", side_effect=run):
                success, _ = clipboard_utils.copy_file_to_clipboard(filepath)

        self.assertTrue(success)
        self.assertNotIn(filepath, captured["command"][-1])
        self.assertEqual(
            captured["kwargs"]["env"]["QUICK_AUDIO_RECORDER_PATH"], filepath
        )

    def test_output_path_does_not_reuse_existing_recording_name(self):
        with tempfile.TemporaryDirectory() as folder:
            first = audio_recorder._unique_output_path(folder, "wav")
            open(first, "wb").close()
            second = audio_recorder._unique_output_path(folder, "wav")

        self.assertNotEqual(first, second)
        self.assertTrue(second.endswith("_1.wav"))

    def test_recorder_error_finishes_without_waiting_for_manual_stop(self):
        class FailedRecorder:
            def __init__(self, *_):
                self.error = "device unavailable"

            def start(self):
                pass

            def is_alive(self):
                return False

            def stop(self):
                pass

        callback_result = []
        recorder = audio_recorder.AudioRecorder(
            mic_id="mic",
            source_mode="mic",
            output_folder=tempfile.gettempdir(),
            on_finish_callback=lambda path, error: callback_result.append(error),
        )
        recorder._get_device = lambda is_loopback: object()

        with (
            patch.object(audio_recorder, "RawRecorder", FailedRecorder),
            patch.object(audio_recorder.logger, "exception"),
        ):
            thread = threading.Thread(target=recorder.run)
            thread.start()
            thread.join(1)
            finished_before_manual_stop = not thread.is_alive()
            if thread.is_alive():
                recorder.stop()
                thread.join(1)

        self.assertTrue(finished_before_manual_stop)
        self.assertEqual(callback_result, ["Recorder error: device unavailable"])

    def test_hotkey_callbacks_dispatch_through_qt_signals(self):
        tree = ast.parse(Path("gui.py").read_text(encoding="utf-8"))
        callbacks = [
            call.args[1]
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_hotkey"
        ]

        self.assertEqual(len(callbacks), 4)
        for callback in callbacks:
            self.assertIsInstance(callback, ast.Lambda)
            self.assertIsInstance(callback.body, ast.Call)
            self.assertEqual(callback.body.func.attr, "emit")

    def test_runtime_requirements_include_pyqt6(self):
        requirements = {
            line.strip().lower()
            for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("pyqt6", requirements)
        self.assertIn("pyaudiowpatch", requirements)
        self.assertNotIn("soundcard", requirements)

    def test_workflow_installs_runtime_requirements_before_building(self):
        workflow = Path(".github/workflows/build-windows.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("pip install -r requirements.txt", workflow)
        self.assertIn("pyinstaller", workflow.lower())

    def test_logging_uses_daily_rotation_and_seven_backups(self):
        import logging
        from logging.handlers import TimedRotatingFileHandler

        import logging_setup

        with tempfile.TemporaryDirectory() as folder:
            log_path = logging_setup.configure_logging(folder)
            handlers = [
                handler
                for handler in logging.getLogger().handlers
                if getattr(handler, "_quick_audio_recorder", False)
            ]
            try:
                self.assertEqual(log_path, os.path.join(folder, "QuickAudioRecorder.log"))
                self.assertTrue(os.path.exists(log_path))
                self.assertEqual(len(handlers), 1)
                self.assertIsInstance(handlers[0], TimedRotatingFileHandler)
                self.assertEqual(handlers[0].when, "MIDNIGHT")
                self.assertEqual(handlers[0].backupCount, 7)
            finally:
                for handler in handlers:
                    logging.getLogger().removeHandler(handler)
                    handler.close()

    def test_raw_recorder_probes_supported_audio_parameters(self):
        class FakeAudio:
            def __init__(self):
                self.attempts = []

            def is_format_supported(
                self, samplerate, input_device, input_channels, input_format
            ):
                self.attempts.append((samplerate, input_channels))
                if samplerate != 48000:
                    raise RuntimeError("unsupported sample rate")

        audio = FakeAudio()
        recorder = audio_recorder.RawRecorder(
            {
                "name": "Test microphone",
                "index": 7,
                "defaultSampleRate": 44100,
                "maxInputChannels": 1,
            },
            "unused.wav",
            samplerate=44100,
            channels=2,
        )

        self.assertEqual(recorder._probe_params(audio), (48000, 1))
        self.assertEqual(audio.attempts, [(44100, 1), (48000, 1)])

    def test_hotkey_cleanup_removes_only_registered_hotkeys(self):
        import gui

        class FakeKeyboard:
            def __init__(self):
                self.removed = []

            def remove_hotkey(self, handle):
                self.removed.append(handle)

        keyboard = FakeKeyboard()
        application = gui.TrayApplication.__new__(gui.TrayApplication)
        application.hotkey_handles = ["mic-handle", "stop-handle"]

        with patch.object(gui, "keyboard", keyboard):
            application._clear_hotkeys()

        self.assertEqual(keyboard.removed, ["mic-handle", "stop-handle"])
        self.assertEqual(application.hotkey_handles, [])

    def test_audio_recorder_uses_selected_loopback_device(self):
        class FakeAudio:
            def __init__(self):
                self.requested_index = None

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def get_device_info_by_index(self, index):
                self.requested_index = index
                return {
                    "index": index,
                    "name": "Selected speakers [Loopback]",
                    "isLoopbackDevice": True,
                    "maxInputChannels": 2,
                }

        audio = FakeAudio()
        recorder = audio_recorder.AudioRecorder(
            mic_id=3,
            loopback_id=12,
            source_mode="loopback",
            output_folder=tempfile.gettempdir(),
        )

        with patch.object(audio_recorder.pyaudio, "PyAudio", return_value=audio):
            selected = recorder._get_device(is_loopback=True)

        self.assertEqual(audio.requested_index, 12)
        self.assertEqual(selected["index"], 12)

    def test_settings_expose_and_persist_loopback_selection(self):
        source = Path("gui.py").read_text(encoding="utf-8")

        self.assertIn("self.combo_loopback", source)
        self.assertIn('"loopback_id"', source)
        self.assertIn("loopback_id=target_loopback_id", source)

    def test_application_icon_is_separate_from_system_tray_icon(self):
        main_source = Path("main.py").read_text(encoding="utf-8")
        gui_source = Path("gui.py").read_text(encoding="utf-8")

        self.assertTrue(Path("app_icon.ico").is_file())
        self.assertIn("app.setWindowIcon", main_source)
        self.assertIn("self.setWindowIcon", gui_source)
        self.assertIn("QSystemTrayIcon(QIcon(self.icon_idle_path)", gui_source)

    def test_builds_include_runtime_and_executable_icon(self):
        workflow = Path(".github/workflows/build-windows.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("--icon=app_icon.ico", workflow)
        self.assertIn("--add-data", workflow)


if __name__ == "__main__":
    unittest.main()
