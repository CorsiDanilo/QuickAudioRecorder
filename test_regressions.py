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
        try:
            __import__(name)
        except ImportError:
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
        self.assertIn('      - "*.*.*"', workflow)

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

    def test_generate_test_chime_produces_pcm_bytes(self):
        sample_rate = 44100
        duration = 0.5
        pcm_bytes = audio_recorder.generate_test_chime(
            sample_rate=sample_rate, channels=2, duration=duration
        )
        self.assertIsInstance(pcm_bytes, bytes)
        expected_len = int(sample_rate * duration) * 2 * 2  # 2 channels * 2 bytes per int16
        self.assertEqual(len(pcm_bytes), expected_len)

    def test_find_output_device_for_loopback_maps_matching_name(self):
        class FakeAudio:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def get_device_info_by_index(self, index):
                if index == 19:
                    return {
                        "index": 19,
                        "name": "Headphones (Realtek) [Loopback]",
                        "isLoopbackDevice": True,
                    }
                raise ValueError("Unknown index")

            def get_device_info_generator_by_host_api(self, host_api_type):
                return [
                    {
                        "index": 10,
                        "name": "Headphones (Realtek)",
                        "maxOutputChannels": 2,
                        "isLoopbackDevice": False,
                    },
                    {
                        "index": 11,
                        "name": "Speakers (Other)",
                        "maxOutputChannels": 2,
                        "isLoopbackDevice": False,
                    },
                ]

            def get_default_wasapi_device(self, d_out=True):
                return {"index": 11}

        with patch.object(audio_recorder.pyaudio, "PyAudio", return_value=FakeAudio()):
            output_id = audio_recorder.find_output_device_for_loopback(19)

        self.assertEqual(output_id, 10)

    def test_find_output_device_for_loopback_falls_back_to_default_output(self):
        class FakeAudio:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def get_default_wasapi_device(self, d_out=True):
                return {"index": 99}

        with patch.object(audio_recorder.pyaudio, "PyAudio", return_value=FakeAudio()):
            output_id = audio_recorder.find_output_device_for_loopback(None)

        self.assertEqual(output_id, 99)

    def test_gui_exposes_test_buttons_and_spectrum_widget(self):
        source = Path("gui.py").read_text(encoding="utf-8")
        self.assertIn("self.btn_test_mic", source)
        self.assertIn("self.btn_test_loopback", source)
        self.assertIn("self.spectrum_widget", source)
        self.assertIn("AudioSpectrumWidget", source)
        self.assertIn("AudioMonitorThread", source)

    def test_tray_menu_includes_audio_devices_submenu_and_refresh_action(self):
        source = Path("gui.py").read_text(encoding="utf-8")
        self.assertIn("self.menu_devices", source)
        self.assertIn("self.menu_mic", source)
        self.assertIn("self.menu_loop", source)
        self.assertIn("self.action_refresh_devices", source)
        self.assertIn("self.populate_device_menus", source)

    def test_start_recording_refreshes_devices_before_starting(self):
        source = Path("gui.py").read_text(encoding="utf-8")
        self.assertIn("self.settings_window.refresh_devices(preserve_selection=True)", source)

    def test_mid_recording_device_disconnection_finalizes_cleanly(self):
        class DisconnectingRecorder:
            def __init__(self, device, filepath, **_):
                self.device = device
                self.filepath = filepath
                self.error = "Unanticipated host error"
                self.device_disconnected = True
                self.frames_written = 44100
                self.finished_event = threading.Event()
                self.stop_event = threading.Event()

            def start(self):
                import soundfile as sf
                import numpy as np
                sf.write(self.filepath, np.zeros((100, 2), dtype=np.float32), 44100)
                self.finished_event.set()

            def is_alive(self):
                return False

            def stop(self):
                self.stop_event.set()

        callback_results = []
        with tempfile.TemporaryDirectory() as folder:
            recorder = audio_recorder.AudioRecorder(
                mic_id=1,
                source_mode="mic",
                output_folder=folder,
                output_format="wav",
                on_finish_callback=lambda path, error: callback_results.append((path, error)),
            )
            recorder._get_device = lambda is_loopback: {"index": 1, "name": "Fake Mic"}

            with patch.object(audio_recorder, "RawRecorder", DisconnectingRecorder):
                recorder.run()

            self.assertTrue(recorder.device_disconnected)
            self.assertIsNone(recorder.error_message)
            self.assertEqual(len(callback_results), 1)
            final_path, err = callback_results[0]
            self.assertIsNone(err)
            self.assertTrue(os.path.exists(final_path))

    def test_gui_implements_device_disconnection_resume(self):
        source = Path("gui.py").read_text(encoding="utf-8")
        self.assertIn("_resume_recording_after_device_change", source)
        self.assertIn("was_device_change", source)
        self.assertIn("device_disconnected", source)

    def test_audio_recorder_notifies_capture_stopped_before_postprocessing(self):
        events = []

        class DummyRawRecorder:
            def __init__(self, device, filepath, **_):
                self.error = None
                self.device_disconnected = False

            def start(self):
                events.append("raw_start")

            def stop(self):
                events.append("raw_stop")

            def is_alive(self):
                return False

        with tempfile.TemporaryDirectory() as folder:
            rec = audio_recorder.AudioRecorder(
                mic_id=1,
                source_mode="mic",
                output_folder=folder,
                output_format="wav",
                on_capture_stopped_callback=lambda: events.append("capture_stopped"),
                on_finish_callback=lambda path, err: events.append("finish_callback"),
            )
            rec._get_device = lambda is_loopback: {"index": 1, "name": "Fake"}
            # create the expected partial file so finalize succeeds
            def fake_create_raw(device, filepath):
                import soundfile as sf
                import numpy as np
                sf.write(filepath, np.zeros((100, 1), dtype=np.float32), 44100)
                return DummyRawRecorder(device, filepath)

            rec._create_raw_recorder = fake_create_raw
            rec.stop_event.set()

            with patch.object(audio_recorder.pyaudio, "PyAudio"):
                rec.run()

            self.assertIn("raw_stop", events)
            self.assertIn("capture_stopped", events)
            self.assertIn("finish_callback", events)
            # capture_stopped must occur after raw_stop and before finish_callback
            raw_stop_idx = events.index("raw_stop")
            cap_stop_idx = events.index("capture_stopped")
            finish_idx = events.index("finish_callback")
            self.assertGreater(cap_stop_idx, raw_stop_idx)
            self.assertGreater(finish_idx, cap_stop_idx)

    def test_gui_supports_starting_new_recording_while_previous_is_saving(self):
        import gui

        app = gui.TrayApplication.__new__(gui.TrayApplication)
        app.recorder = None
        app.saving_recorders = set()
        app.last_mode = "mic"
        app.icon_idle_path = "icon_idle.png"
        app.icon_rec_path = "icon_rec.png"

        class DummyAction:
            def __init__(self):
                self.enabled = False

            def setEnabled(self, val):
                self.enabled = val

        class DummyTrayIcon:
            def __init__(self):
                self.icon = None
                self.tooltip = ""
                self.messages = []

            def setIcon(self, icon):
                self.icon = icon

            def setToolTip(self, text):
                self.tooltip = text

            def showMessage(self, *args):
                self.messages.append(args)

        app.action_record_mic = DummyAction()
        app.action_record_loop = DummyAction()
        app.action_record_both = DummyAction()
        app.action_stop = DummyAction()
        app.tray_icon = DummyTrayIcon()
        app.settings_window = types.SimpleNamespace(
            get_settings=lambda: {"clipboard": False, "delete_after": False}
        )

        with patch.object(gui, "QIcon", lambda path: path):
            # 1. Start recording 1
            class FakeRecorder:
                def __init__(self, name):
                    self.name = name
                    self.is_capturing = True
                    self.device_disconnected = False

                def is_alive(self):
                    return True

            rec1 = FakeRecorder("rec1")
            app.recorder = rec1
            self.assertTrue(app.is_recording())

            # 2. Stop capture on recording 1 (postprocessing begins)
            app.on_capture_finished(rec1)
            self.assertFalse(app.is_recording())
            self.assertIn(rec1, app.saving_recorders)
            self.assertTrue(app.action_record_mic.enabled)
            self.assertTrue(app.action_record_loop.enabled)
            self.assertTrue(app.action_record_both.enabled)
            self.assertFalse(app.action_stop.enabled)

            # 3. User immediately starts recording 2 while rec1 is saving
            rec2 = FakeRecorder("rec2")
            app.recorder = rec2
            self.assertTrue(app.is_recording())

            # 4. rec1 finishes saving in the background
            app.on_recording_finished("/path/to/rec1.wav", None, rec1)
            self.assertNotIn(rec1, app.saving_recorders)
            # rec2 is still recording! State must remain recording
            self.assertTrue(app.is_recording())
            self.assertIn("Recording", app.tray_icon.tooltip)

            # 5. Stop capture on recording 2
            app.on_capture_finished(rec2)
            self.assertFalse(app.is_recording())
            self.assertIn(rec2, app.saving_recorders)

            # 6. rec2 finishes saving
            app.on_recording_finished("/path/to/rec2.wav", None, rec2)
            self.assertNotIn(rec2, app.saving_recorders)
            self.assertFalse(app.is_recording())
            self.assertIn("Idle", app.tray_icon.tooltip)


if __name__ == "__main__":
    unittest.main()

