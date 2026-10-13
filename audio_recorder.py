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
PROCESSING_BLOCK_FRAMES = 16_384

def _unique_output_path(output_folder, output_format, prefix="Recording"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    extension = output_format.lower()
    path = os.path.join(output_folder, f"{prefix}_{timestamp}.{extension}")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(
            output_folder, f"{prefix}_{timestamp}_{counter}.{extension}"
        )
        counter += 1
    return path


def _create_recording_folder(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(output_folder, f"Recording_{timestamp}")
    counter = 1
    while os.path.exists(folder):
        folder = os.path.join(output_folder, f"Recording_{timestamp}_{counter}")
        counter += 1
    os.makedirs(folder)
    return folder


class _StreamingAudioReader:
    """Read a WAV as blocks at a shared sample rate and channel count."""

    def __init__(
        self,
        filepath,
        target_samplerate,
        target_channels,
        block_frames=PROCESSING_BLOCK_FRAMES,
    ):
        self._file = sf.SoundFile(filepath, mode="r")
        self._source_samplerate = self._file.samplerate
        self._source_frames = len(self._file)
        self._target_samplerate = target_samplerate
        self._target_channels = target_channels
        self._block_frames = block_frames
        self._ratio = self._source_samplerate / target_samplerate
        self._buffer = np.empty((0, self._file.channels), dtype=np.float32)
        self._buffer_start = 0
        self._next_position = 0.0
        self._finished = False

    def read(self, frames):
        if self._finished:
            return self._silence(frames), False

        if self._source_samplerate == self._target_samplerate:
            data = self._file.read(frames, dtype="float32", always_2d=True)
            has_audio = len(data) > 0
            if len(data) < frames:
                self._finished = True
            return self._pad_and_convert(data, frames), has_audio

        positions = self._next_position + np.arange(frames) * self._ratio
        last_source_frame = self._source_frames - 1
        valid_frames = int(
            np.searchsorted(positions, last_source_frame, side="right")
        )
        if valid_frames == 0:
            self._finished = True
            return self._silence(frames), False

        self._ensure_buffer_through(int(np.ceil(positions[valid_frames - 1])))
        offsets = positions[:valid_frames] - self._buffer_start
        lower = np.floor(offsets).astype(np.int64)
        upper = np.minimum(lower + 1, len(self._buffer) - 1)
        fractions = (offsets - lower)[:, np.newaxis]
        data = self._buffer[lower] * (1.0 - fractions) + self._buffer[upper] * fractions

        self._next_position += frames * self._ratio
        self._discard_consumed_buffer()
        if valid_frames < frames:
            self._finished = True
        return self._pad_and_convert(data, frames), True

    def close(self):
        self._file.close()

    def _ensure_buffer_through(self, source_frame):
        while self._buffer_start + len(self._buffer) - 1 < source_frame:
            missing_frames = source_frame - (
                self._buffer_start + len(self._buffer) - 1
            )
            data = self._file.read(
                max(self._block_frames, missing_frames),
                dtype="float32",
                always_2d=True,
            )
            if not len(data):
                break
            self._buffer = np.concatenate((self._buffer, data))

    def _discard_consumed_buffer(self):
        keep_from = int(np.floor(self._next_position))
        discard_count = max(0, keep_from - self._buffer_start)
        if discard_count:
            self._buffer = self._buffer[discard_count:]
            self._buffer_start += discard_count

    def _pad_and_convert(self, data, frames):
        data = _to_channels(data, self._target_channels).astype(np.float32, copy=False)
        if len(data) == frames:
            return data
        padded = self._silence(frames)
        padded[: len(data)] = data
        return padded

    def _silence(self, frames):
        return np.zeros((frames, self._target_channels), dtype=np.float32)

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
                 normalize=False, on_finish_callback=None, loopback_id=None,
                 keep_source_wav=False):
        super().__init__()
        self.mic_id = mic_id
        self.source_mode = source_mode # "mic", "loopback", "both"
        self.output_folder = output_folder
        self.output_format = output_format.lower()
        self.normalize = normalize
        self.callback = on_finish_callback
        self.loopback_id = loopback_id
        self.keep_source_wav = keep_source_wav
        
        self.recording = False
        self.stop_event = threading.Event()
        self.error_message = None
        self.final_filepath = None
        
        # Temp files
        self.temp_files = []
        self.capture_files = []
        self.recording_folder = None
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
        self.capture_files = []
        self.recording_folder = None
        self.recorders = []
        self.audio = None
        
        try:
            # 1. Setup Recorders
            if self.source_mode == "both":
                # Need two recorders
                dev_mic = self._get_device(is_loopback=False)
                dev_loop = self._get_device(is_loopback=True)

                self.recording_folder = _create_recording_folder(self.output_folder)
                t1 = self._capture_path("Microphone")
                t2 = self._capture_path("SystemAudio")
                self.temp_files = [t1, t2]

                self.audio = pyaudio.PyAudio()
                self.recorders.append(self._create_raw_recorder(dev_mic, t1))
                self.recorders.append(self._create_raw_recorder(dev_loop, t2))
                
            elif self.source_mode == "loopback":
                dev = self._get_device(is_loopback=True)
                self.recording_folder = _create_recording_folder(self.output_folder)
                t1 = self._capture_path("SystemAudio")
                self.temp_files = [t1]
                self.audio = pyaudio.PyAudio()
                self.recorders.append(self._create_raw_recorder(dev, t1))
                
            else: # mic
                dev = self._get_device(is_loopback=False)
                self.recording_folder = _create_recording_folder(self.output_folder)
                t1 = self._capture_path("Microphone")
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
            stop_started = time.perf_counter()
            for r in self.recorders:
                r.stop()
                if r.error:
                    raise Exception(f"Recorder error: {r.error}")
            logger.info(
                "Input streams stopped in %.3fs", time.perf_counter() - stop_started
            )

            # 4. Finalize
            if not self.recording_folder:
                raise RuntimeError("Recording output folder was not created")
            self.final_filepath = _unique_output_path(
                self.recording_folder, self.output_format, prefix="Merged"
            )

            if len(self.temp_files) == 2:
                if not self.normalize:
                    mix_started = time.perf_counter()
                    if self.output_format == "mp3":
                        self._mix_audio_to_mp3(
                            self.temp_files[0], self.temp_files[1], self.final_filepath
                        )
                    else:
                        self._mix_audio(
                            self.temp_files[0], self.temp_files[1], self.final_filepath
                        )
                    logger.info(
                        "Mixed and finalized both tracks in %.3fs",
                        time.perf_counter() - mix_started,
                    )
                    return

                descriptor, mixed_wav = tempfile.mkstemp(suffix=".wav")
                os.close(descriptor)
                mix_started = time.perf_counter()
                self._mix_audio(self.temp_files[0], self.temp_files[1], mixed_wav)
                source_wav = mixed_wav
                self.temp_files.append(mixed_wav)
                logger.info("Mixed both tracks in %.3fs", time.perf_counter() - mix_started)
            else:
                source_wav = self.temp_files[0]

            # 5. Normalization
            if self.normalize:
                self._normalize_audio(source_wav)

            if self.output_format == "mp3":
                self._convert_to_mp3(source_wav, self.final_filepath)
            else:
                copy_started = time.perf_counter()
                shutil.copy2(source_wav, self.final_filepath)
                logger.info("WAV copy completed in %.3fs", time.perf_counter() - copy_started)

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
            if self.error_message is None:
                try:
                    source_files = self._finalize_capture_files()
                    self._finalize_source_files(source_files)
                except Exception as error:
                    self.error_message = str(error)
                    logger.exception("Failed to finalize source recordings")
            # Clean up generated intermediate files, keeping the source captures.
            for t in self.temp_files:
                if t in self.capture_files:
                    continue
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

    def _capture_path(self, name):
        path = os.path.join(self.recording_folder, f"{name}.partial.wav")
        self.capture_files.append(path)
        return path

    def _finalize_capture_files(self):
        source_files = []
        for partial_path in self.capture_files:
            if not os.path.exists(partial_path):
                continue
            final_path = partial_path.replace(".partial.wav", ".wav")
            os.replace(partial_path, final_path)
            source_files.append(final_path)
        logger.info("Source WAV recordings saved in %s", self.recording_folder)
        return source_files

    def _finalize_source_files(self, source_files):
        if self.output_format != "mp3" or self.keep_source_wav:
            if self.output_format == "mp3" and self.keep_source_wav:
                logger.info("Keeping original WAV source recordings")
            return

        started = time.perf_counter()
        for source_wav in source_files:
            source_mp3 = os.path.splitext(source_wav)[0] + ".mp3"
            self._convert_to_mp3(source_wav, source_mp3)
            os.remove(source_wav)
        logger.info(
            "Source recordings converted to MP3 in %.3fs",
            time.perf_counter() - started,
        )

    def stop(self):
        self.stop_event.set()

    def _mix_audio(self, file1, file2, out_file):
        samplerate, channels = self._mixed_audio_format(file1, file2)
        with sf.SoundFile(
            out_file,
            mode="w",
            samplerate=samplerate,
            channels=channels,
            subtype="PCM_16",
        ) as output:
            for mixed_block in self._iter_mixed_audio_blocks(file1, file2):
                output.write(mixed_block)

    def _normalize_audio(self, filepath):
        started = time.perf_counter()
        normalized_path = None
        try:
            with sf.SoundFile(filepath, mode="r") as source:
                max_val = 0.0
                for block in source.blocks(
                    blocksize=PROCESSING_BLOCK_FRAMES,
                    dtype="float32",
                    always_2d=True,
                ):
                    if len(block):
                        max_val = max(max_val, float(np.max(np.abs(block))))
                if max_val == 0:
                    logger.info("Normalization skipped for silent recording")
                    return
                samplerate = source.samplerate
                channels = source.channels
                subtype = source.subtype

            descriptor, normalized_path = tempfile.mkstemp(suffix=".wav")
            os.close(descriptor)
            factor = 0.99 / max_val
            with sf.SoundFile(filepath, mode="r") as source, sf.SoundFile(
                normalized_path,
                mode="w",
                samplerate=samplerate,
                channels=channels,
                subtype=subtype,
            ) as output:
                for block in source.blocks(
                    blocksize=PROCESSING_BLOCK_FRAMES,
                    dtype="float32",
                    always_2d=True,
                ):
                    output.write(block * factor)
            os.replace(normalized_path, filepath)
            normalized_path = None
            logger.info("Normalization completed in %.3fs", time.perf_counter() - started)
        except Exception:
            logger.exception("Normalization failed for %s", filepath)
        finally:
            if normalized_path and os.path.exists(normalized_path):
                os.remove(normalized_path)

    def _convert_to_mp3(self, src_wav, dst_mp3):
        started = time.perf_counter()
        temporary_mp3 = f"{dst_mp3}.partial"
        try:
            with sf.SoundFile(src_wav, mode="r") as source:
                encoder = self._create_mp3_encoder(source.samplerate, source.channels)
                with open(temporary_mp3, "wb") as output:
                    for block in source.blocks(
                        blocksize=PROCESSING_BLOCK_FRAMES,
                        dtype="float32",
                        always_2d=True,
                    ):
                        output.write(encoder.encode(self._pcm_bytes(block)))
                    output.write(encoder.flush())
            os.replace(temporary_mp3, dst_mp3)
            logger.info("MP3 conversion completed in %.3fs", time.perf_counter() - started)
        finally:
            if os.path.exists(temporary_mp3):
                os.remove(temporary_mp3)

    def _mix_audio_to_mp3(self, file1, file2, dst_mp3):
        samplerate, channels = self._mixed_audio_format(file1, file2)
        encoder = self._create_mp3_encoder(samplerate, channels)
        with open(dst_mp3, "wb") as output:
            for mixed_block in self._iter_mixed_audio_blocks(file1, file2):
                output.write(encoder.encode(self._pcm_bytes(mixed_block)))
            output.write(encoder.flush())

    def _iter_mixed_audio_blocks(self, file1, file2):
        samplerate, channels = self._mixed_audio_format(file1, file2)
        first = _StreamingAudioReader(file1, samplerate, channels)
        second = _StreamingAudioReader(file2, samplerate, channels)
        try:
            while True:
                first_block, first_has_audio = first.read(PROCESSING_BLOCK_FRAMES)
                second_block, second_has_audio = second.read(PROCESSING_BLOCK_FRAMES)
                if not first_has_audio and not second_has_audio:
                    return
                yield np.clip(first_block + second_block, -1.0, 1.0)
        finally:
            first.close()
            second.close()

    @staticmethod
    def _mixed_audio_format(file1, file2):
        first = sf.info(file1)
        second = sf.info(file2)
        return first.samplerate, max(first.channels, second.channels)

    @staticmethod
    def _create_mp3_encoder(samplerate, channels):
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(192)
        encoder.set_in_sample_rate(samplerate)
        encoder.set_channels(channels)
        encoder.set_quality(2)
        return encoder

    @staticmethod
    def _pcm_bytes(data):
        pcm_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
        return np.ascontiguousarray(pcm_data).tobytes()

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


def find_output_device_for_loopback(loopback_id):
    """
    Finds the WASAPI playback (output) device corresponding to the given loopback device ID.
    If loopback_id is None or no matching name is found, falls back to the default WASAPI output device.
    """
    try:
        with pyaudio.PyAudio() as audio:
            if loopback_id is not None:
                try:
                    lb_info = audio.get_device_info_by_index(int(loopback_id))
                    clean_name = (
                        lb_info.get("name", "")
                        .replace(" [Loopback]", "")
                        .strip()
                    )
                    for device in audio.get_device_info_generator_by_host_api(
                        host_api_type=pyaudio.paWASAPI
                    ):
                        if (
                            device.get("maxOutputChannels", 0) > 0
                            and not device.get("isLoopbackDevice", False)
                        ):
                            d_name = device.get("name", "").strip()
                            if d_name == clean_name or clean_name in d_name:
                                return int(device["index"])
                except Exception:
                    logger.warning(
                        "Could not resolve loopback ID %s to output device",
                        loopback_id,
                    )
            default_out = audio.get_default_wasapi_device(d_out=True)
            return int(default_out["index"])
    except Exception:
        logger.exception("Error finding output device for loopback")
        return None


def generate_test_chime(sample_rate=44100, channels=2, duration=0.9):
    """
    Synthesize a pleasant harmonic chime (two successive notes with soft decay)
    using numpy, without requiring any external audio files.
    """
    t = np.linspace(
        0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32
    )
    # Note 1: 587.33 Hz (D5) from 0.0s to 0.45s
    t1 = t[t < 0.45]
    n1 = 0.28 * np.sin(2 * np.pi * 587.33 * t1) * np.exp(-t1 * 7.0)
    # Note 2: 880.0 Hz (A5) from 0.2s to end
    offset2 = 0.2
    t2 = t[t >= offset2] - offset2
    n2 = 0.28 * np.sin(2 * np.pi * 880.0 * t2) * np.exp(-t2 * 5.5)

    audio_signal = np.zeros_like(t)
    audio_signal[: len(n1)] += n1
    audio_signal[int(sample_rate * offset2) : int(sample_rate * offset2) + len(n2)] += n2

    if channels == 1:
        pcm_channels = audio_signal[:, np.newaxis]
    else:
        # Subtle stereo widening
        left = audio_signal.copy()
        right = audio_signal.copy()
        left[: len(n1)] *= 1.15
        right[int(sample_rate * offset2) :] *= 1.15
        pcm_channels = np.column_stack([left, right])

    pcm_data = (np.clip(pcm_channels, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm_data.tobytes()


def play_test_sound(loopback_id=None, output_device_id=None):
    """
    Plays the synthetic test chime on the target WASAPI output device.
    """
    target_idx = None
    try:
        with pyaudio.PyAudio() as audio:
            target_idx = output_device_id
            if target_idx is None:
                target_idx = find_output_device_for_loopback(loopback_id)
            if target_idx is None:
                default_out = audio.get_default_wasapi_device(d_out=True)
                target_idx = int(default_out["index"])

            info = audio.get_device_info_by_index(target_idx)
            sample_rate = int(float(info.get("defaultSampleRate", 44100)))
            max_channels = int(info.get("maxOutputChannels", 2))
            channels = min(2, max(1, max_channels))

            for probe_rate in (sample_rate, 48000, 44100):
                try:
                    audio.is_format_supported(
                        probe_rate,
                        output_device=target_idx,
                        output_channels=channels,
                        output_format=pyaudio.paInt16,
                    )
                    sample_rate = probe_rate
                    break
                except Exception:
                    continue

            pcm_bytes = generate_test_chime(
                sample_rate=sample_rate, channels=channels
            )
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                output=True,
                output_device_index=target_idx,
                frames_per_buffer=1024,
            )
            try:
                chunk_size = 1024 * channels * 2
                for offset in range(0, len(pcm_bytes), chunk_size):
                    chunk = pcm_bytes[offset : offset + chunk_size]
                    stream.write(chunk)
            finally:
                stream.stop_stream()
                stream.close()
    except Exception:
        logger.exception("Failed to play test sound on device %s", target_idx)
        raise


class AudioMonitorThread(threading.Thread):
    """
    Captures live audio from an input device in real time, computes FFT spectrum bands
    and RMS level, and notifies callbacks. Does not write to disk.
    """

    def __init__(
        self,
        device_id,
        num_bands=28,
        callback=None,
        error_callback=None,
    ):
        super().__init__(daemon=True)
        self.device_id = device_id
        self.num_bands = num_bands
        self.callback = callback
        self.error_callback = error_callback
        self.stop_event = threading.Event()

    def run(self):
        audio = None
        stream = None
        try:
            audio = pyaudio.PyAudio()
            device = audio.get_device_info_by_index(int(self.device_id))
            device_channels = int(device.get("maxInputChannels", 0))
            if device_channels < 1:
                raise RuntimeError("Selected device has no input channels.")

            default_rate = int(float(device.get("defaultSampleRate", 44100)))
            sample_rate = default_rate
            for rate in (default_rate, 48000, 44100, 16000, 8000):
                try:
                    audio.is_format_supported(
                        rate,
                        input_device=int(self.device_id),
                        input_channels=1,
                        input_format=pyaudio.paInt16,
                    )
                    sample_rate = rate
                    break
                except Exception:
                    continue

            frames_per_buffer = 1024
            freqs = np.fft.rfftfreq(frames_per_buffer, 1.0 / sample_rate)
            edges = np.geomspace(
                60, min(16000, sample_rate // 2), self.num_bands + 1
            )
            band_masks = [
                (freqs >= edges[i]) & (freqs < edges[i + 1])
                for i in range(self.num_bands)
            ]
            window = np.hanning(frames_per_buffer)

            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=int(self.device_id),
                frames_per_buffer=frames_per_buffer,
            )

            while not self.stop_event.is_set():
                try:
                    raw_data = stream.read(
                        frames_per_buffer, exception_on_overflow=False
                    )
                except Exception:
                    if self.stop_event.is_set():
                        break
                    continue

                if self.stop_event.is_set():
                    break

                samples = (
                    np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                rms = float(np.sqrt(np.mean(samples**2)))

                fft_vals = np.abs(np.fft.rfft(samples * window)) / (
                    frames_per_buffer / 2
                )
                band_values = [
                    float(np.mean(fft_vals[mask])) if np.any(mask) else 0.0
                    for mask in band_masks
                ]

                if self.callback and not self.stop_event.is_set():
                    self.callback(band_values, rms)

        except Exception as error:
            logger.exception("Audio monitor thread failed")
            if self.error_callback and not self.stop_event.is_set():
                self.error_callback(str(error))
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if audio is not None:
                try:
                    audio.terminate()
                except Exception:
                    pass

    def stop(self):
        self.stop_event.set()
        self.join(timeout=1.0)

