import pyaudiowpatch as pyaudio
import soundfile as sf
import logging
import threading
import time
import os
import lameenc
import numpy as np
import tempfile
import shutil

logger = logging.getLogger(__name__)

def _unique_output_path(output_folder, output_format):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    extension = output_format.lower()
    path = os.path.join(output_folder, f"Recording_{timestamp}.{extension}")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(
            output_folder, f"Recording_{timestamp}_{counter}.{extension}"
        )
        counter += 1
    return path

class RawRecorder(threading.Thread):
    """
    Helper thread to record a single device to a WAV file.
    """
    def __init__(self, device, filepath, samplerate=44100, channels=2, audio=None):
        super().__init__()
        self.device = device
        self.filepath = filepath
        self.samplerate = samplerate
        self.channels = channels
        self.audio = audio
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.error = None

    def run(self):
        audio = self.audio
        owns_audio = audio is None
        try:
            if owns_audio:
                audio = pyaudio.PyAudio()
            samplerate, channels = self._probe_params(audio)
            with sf.SoundFile(
                self.filepath,
                mode="w",
                samplerate=samplerate,
                channels=channels,
                subtype="PCM_16",
            ) as f_wav:
                def callback(in_data, frame_count, time_info, status):
                    if self.stop_event.is_set():
                        return (None, pyaudio.paComplete)
                    try:
                        data = np.frombuffer(in_data, dtype=np.int16)
                        if channels > 1:
                            data = data.reshape(-1, channels)
                        data = data.astype(np.float32) / 32768.0
                        f_wav.write(data)
                    except Exception as error:
                        self.error = f"{type(error).__name__}: {error}".rstrip()
                        logger.exception(
                            "Audio callback failed for %s", self.filepath
                        )
                        return (None, pyaudio.paAbort)
                    return (None, pyaudio.paContinue)

                stream_args = {
                    "format": pyaudio.paInt16,
                    "channels": channels,
                    "rate": samplerate,
                    "frames_per_buffer": 2048,
                    "input": True,
                    "input_device_index": int(self.device["index"]),
                }
                use_callback = bool(self.device.get("isLoopbackDevice"))
                if use_callback:
                    stream_args["stream_callback"] = callback
                stream = audio.open(**stream_args)
                try:
                    if use_callback:
                        while not self.stop_event.wait(0.1):
                            if self.error:
                                raise RuntimeError(self.error)
                    else:
                        while not self.stop_event.is_set():
                            raw_data = stream.read(
                                2048, exception_on_overflow=False
                            )
                            data = np.frombuffer(raw_data, dtype=np.int16)
                            if channels > 1:
                                data = data.reshape(-1, channels)
                            data = data.astype(np.float32) / 32768.0
                            f_wav.write(data)
                finally:
                    stream.stop_stream()
                    stream.close()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}".rstrip()
            logger.exception("Raw recorder failed for %s", self.filepath)
        finally:
            if owns_audio and audio is not None:
                audio.terminate()
            self.finished_event.set()

    def stop(self):
        self.stop_event.set()
        self.join()

    def _probe_params(self, audio):
        device_channels = int(self.device.get("maxInputChannels", 0))
        channels = min(self.channels, device_channels) if device_channels else 1
        default_rate = int(float(self.device.get("defaultSampleRate", self.samplerate)))
        sample_rates = []
        for rate in (self.samplerate, default_rate, 48000, 44100, 16000, 8000):
            if rate not in sample_rates:
                sample_rates.append(rate)

        failures = []
        for samplerate in sample_rates:
            for channel_count in ([channels, 1] if channels != 1 else [1]):
                try:
                    audio.is_format_supported(
                        samplerate,
                        input_device=int(self.device["index"]),
                        input_channels=channel_count,
                        input_format=pyaudio.paInt16,
                    )
                    return samplerate, channel_count
                except Exception as error:
                    detail = f"{type(error).__name__}: {error}".rstrip()
                    failures.append(
                        f"{samplerate}Hz/{channel_count}ch: {detail}"
                    )

        raise RuntimeError(
            f"Device '{self.device.get('name', 'unknown')}' rejected all "
            "audio parameter combinations:\n" + "\n".join(failures)
        )

class AudioRecorder(threading.Thread):
    """
    Orchestrates recording from Microphone, Loopback, or Both.
    """
    def __init__(self, mic_id, source_mode, output_folder, output_format="mp3",
                 normalize=False, on_finish_callback=None, loopback_id=None):
        super().__init__()
        self.mic_id = mic_id
        self.source_mode = source_mode # "mic", "loopback", "both"
        self.output_folder = output_folder
        self.output_format = output_format.lower()
        self.normalize = normalize
        self.callback = on_finish_callback
        self.loopback_id = loopback_id
        
        self.recording = False
        self.stop_event = threading.Event()
        self.error_message = None
        self.final_filepath = None
        
        # Temp files
        self.temp_files = []
        self.recorders = []

    def _get_device(self, is_loopback):
        with pyaudio.PyAudio() as audio:
            if not is_loopback:
                try:
                    return audio.get_device_info_by_index(int(self.mic_id))
                except (TypeError, ValueError, IOError) as error:
                    raise RuntimeError(
                        f"Could not open microphone device '{self.mic_id}': {error}"
                    ) from error

            if self.loopback_id is not None:
                selected = audio.get_device_info_by_index(int(self.loopback_id))
                if not selected.get("isLoopbackDevice", False):
                    raise RuntimeError(
                        f"Selected audio device '{selected.get('name', self.loopback_id)}' "
                        "is not a loopback device."
                    )
                return selected

            default_speaker = audio.get_default_wasapi_device(d_out=True)
            speaker_name = default_speaker.get("name", "")
            try:
                return audio.get_default_wasapi_loopback()
            except LookupError as error:
                raise RuntimeError(
                    f"Could not detect system audio loopback device for '{speaker_name}'."
                ) from error

    def run(self):
        self.recording = True
        self.error_message = None
        self.temp_files = []
        self.recorders = []
        self.audio = None
        
        try:
            # 1. Setup Recorders
            if self.source_mode == "both":
                # Need two recorders
                dev_mic = self._get_device(is_loopback=False)
                dev_loop = self._get_device(is_loopback=True)
                
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                t2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1, t2]

                self.audio = pyaudio.PyAudio()
                self.recorders.append(self._create_raw_recorder(dev_mic, t1))
                self.recorders.append(self._create_raw_recorder(dev_loop, t2))
                
            elif self.source_mode == "loopback":
                dev = self._get_device(is_loopback=True)
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1]
                self.audio = pyaudio.PyAudio()
                self.recorders.append(self._create_raw_recorder(dev, t1))
                
            else: # mic
                dev = self._get_device(is_loopback=False)
                t1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files = [t1]
                self.audio = pyaudio.PyAudio()
                self.recorders.append(self._create_raw_recorder(dev, t1))

            logger.info("Starting recording mode: %s", self.source_mode)

            # 2. Start Recording
            for r in self.recorders:
                r.start()
            
            # Wait for stop signal or an unexpected recorder failure.
            while not self.stop_event.wait(0.1):
                failed = next(
                    (
                        r
                        for r in self.recorders
                        if r.error
                        or (
                            getattr(r, "finished_event", None)
                            and r.finished_event.is_set()
                        )
                        or not r.is_alive()
                    ),
                    None,
                )
                if failed:
                    detail = failed.error or "stopped unexpectedly"
                    raise Exception(f"Recorder error: {detail}")
            
            # 3. Stop Recording
            for r in self.recorders:
                r.stop()
                if r.error:
                    raise Exception(f"Recorder error: {r.error}")

            # 4. Mix/Process
            if len(self.temp_files) == 2:
                mixed_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self._mix_audio(self.temp_files[0], self.temp_files[1], mixed_wav)
                # Use mixed file as source for next steps
                source_wav = mixed_wav
                self.temp_files.append(mixed_wav) # Mark for cleanup
            else:
                source_wav = self.temp_files[0]

            # 5. Normalization
            if self.normalize:
                self._normalize_audio(source_wav)
            
            # 6. Finalize
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            self.final_filepath = _unique_output_path(
                self.output_folder, self.output_format
            )
            
            if self.output_format == "mp3":
                self._convert_to_mp3(source_wav, self.final_filepath)
            else:
                shutil.copy2(source_wav, self.final_filepath)

        except Exception as e:
            self.error_message = str(e)
            logger.exception("Error during recording process")
        finally:
            self.recording = False
            for recorder in self.recorders:
                if recorder.is_alive():
                    try:
                        recorder.stop()
                    except Exception:
                        pass
            if self.audio is not None:
                try:
                    self.audio.terminate()
                except Exception:
                    logger.exception("Failed to terminate audio backend")
                self.audio = None
            # Clean up all temp files
            for t in self.temp_files:
                if os.path.exists(t):
                    try:
                        os.remove(t)
                    except: pass
                
            if self.callback:
                self.callback(self.final_filepath, self.error_message)

    def _create_raw_recorder(self, device, filepath):
        recorder = RawRecorder(device, filepath)
        recorder.audio = self.audio
        return recorder

    def stop(self):
        self.stop_event.set()

    def _mix_audio(self, file1, file2, out_file):
        d1, sr1 = sf.read(file1)
        d2, sr2 = sf.read(file2)

        if sr2 != sr1:
            d2 = _resample_audio(d2, sr2, sr1)

        target_channels = max(
            d1.shape[1] if d1.ndim > 1 else 1,
            d2.shape[1] if d2.ndim > 1 else 1,
        )
        d1 = _to_channels(d1, target_channels)
        d2 = _to_channels(d2, target_channels)
        
        # Ensure same length
        max_len = max(len(d1), len(d2))
        
        if len(d1) < max_len:
            pad_width = max_len - len(d1)
            d1 = np.concatenate(
                (d1, np.zeros((pad_width, target_channels), dtype=d1.dtype))
            )
            
        if len(d2) < max_len:
            pad_width = max_len - len(d2)
            d2 = np.concatenate(
                (d2, np.zeros((pad_width, target_channels), dtype=d2.dtype))
            )
            
        # Mix (Sum)
        mixed = d1 + d2
        # Clip
        mixed = np.clip(mixed, -1.0, 1.0)
        
        sf.write(out_file, mixed, sr1)

    def _normalize_audio(self, filepath):
        try:
            data, sr = sf.read(filepath)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                target_peak = 0.99 
                factor = target_peak / max_val
                data = data * factor
                sf.write(filepath, data, sr)
        except Exception as e:
            logger.exception("Normalization failed for %s", filepath)

    def _convert_to_mp3(self, src_wav, dst_mp3):
        data, sr = sf.read(src_wav)
        channels = data.shape[1] if data.ndim > 1 else 1
        
        pcm_data = (data * 32767).clip(-32768, 32767).astype(np.int16)
        
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(192)
        encoder.set_in_sample_rate(sr)
        encoder.set_channels(channels)
        encoder.set_quality(2)
        
        mp3_data = encoder.encode(pcm_data.tobytes())
        mp3_data += encoder.flush()
        
        with open(dst_mp3, "wb") as f_mp3:
            f_mp3.write(mp3_data)

def get_devices(include_loopback=False):
    try:
        with pyaudio.PyAudio() as audio:
            if include_loopback:
                devices = list(audio.get_loopback_device_info_generator())
            else:
                devices = [
                    device
                    for device in audio.get_device_info_generator_by_host_api(
                        host_api_type=pyaudio.paWASAPI
                    )
                    if device.get("maxInputChannels", 0) > 0
                    and not device.get("isLoopbackDevice", False)
                ]
            return [
                {"id": int(device["index"]), "name": device["name"]}
                for device in devices
            ]
    except Exception:
        logger.exception("Error fetching audio devices")
        return []


def get_default_device_id():
    try:
        with pyaudio.PyAudio() as audio:
            return int(audio.get_default_wasapi_device(d_in=True)["index"])
    except Exception:
        logger.exception("Error fetching default audio input device")
        return None


def get_default_loopback_device_id():
    try:
        with pyaudio.PyAudio() as audio:
            return int(audio.get_default_wasapi_loopback()["index"])
    except Exception:
        logger.exception("Error fetching default loopback device")
        return None


def _to_channels(data, channels):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[1] == channels:
        return data
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1)
    if channels == 1:
        return np.mean(data, axis=1, keepdims=True)
    return data[:, :channels]


def _resample_audio(data, source_rate, target_rate):
    if source_rate == target_rate or len(data) == 0:
        return data

    target_length = max(1, round(len(data) * target_rate / source_rate))
    source_positions = np.linspace(0, len(data) - 1, target_length)
    if data.ndim == 1:
        return np.interp(source_positions, np.arange(len(data)), data)
    return np.column_stack(
        [
            np.interp(source_positions, np.arange(len(data)), data[:, channel])
            for channel in range(data.shape[1])
        ]
    )
