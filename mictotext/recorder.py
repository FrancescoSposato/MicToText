"""Microphone capture, both blocking (CLI) and start/stop controlled (web UI)."""

from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

_METER_WIDTH = 30
_METER_INTERVAL = 0.1  # seconds between meter refreshes


def list_input_devices() -> str:
    """Return a printable list of input devices (* marks the default)."""
    default_input = sd.default.device[0]
    lines = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        host_api = sd.query_hostapis(device["hostapi"])["name"]
        marker = "*" if index == default_input else " "
        lines.append(
            f"{marker} [{index:>2}] {device['name']} "
            f"({host_api}, {int(device['default_samplerate'])} Hz)"
        )
    return "\n".join(lines) or "No input devices found."


def list_input_devices_structured() -> list[dict]:
    """Same listing as list_input_devices(), as JSON-friendly dicts (for the web UI)."""
    default_input = sd.default.device[0]
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        host_api = sd.query_hostapis(device["hostapi"])["name"]
        devices.append({
            "index": index,
            "name": device["name"],
            "host_api": host_api,
            "samplerate": int(device["default_samplerate"]),
            "is_default": index == default_input,
        })
    return devices


def _print_meter(elapsed: float, db: float) -> None:
    filled = int(round((min(max(db, -60.0), 0.0) + 60.0) / 60.0 * _METER_WIDTH))
    minutes, seconds = divmod(int(elapsed), 60)
    bar = "#" * filled + " " * (_METER_WIDTH - filled)
    print(f"\r  {minutes:02d}:{seconds:02d} [{bar}] {db:6.1f} dB ", end="", flush=True)


class MicRecorder:
    """A microphone recording that is started and stopped by explicit calls.

    Used directly by the web UI (start on one HTTP request, stop on another) and
    internally by record_until_enter() for the CLI flow.
    """

    def __init__(self, output_path: Path, device: int | str | None = None, channels: int = 1) -> None:
        self.output_path = output_path
        self.device = device
        self.channels = channels

        self._blocks: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()
        self._stream: sd.InputStream | None = None
        self._file: sf.SoundFile | None = None
        self._writer_thread: threading.Thread | None = None

        self.samplerate = 0
        self._frames_written = 0
        self._peak = 0.0
        self._current_rms = 0.0
        self._overflow_count = 0
        self._start_time = 0.0

    def start(self) -> None:
        info = sd.query_devices(self.device, "input")
        self.samplerate = int(info["default_samplerate"])
        self.channels = max(1, min(self.channels, int(info["max_input_channels"])))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(str(self.output_path), mode="w", samplerate=self.samplerate,
                                  channels=self.channels, subtype="PCM_16")

        def callback(indata, frames, time_info, status) -> None:
            if status.input_overflow:
                self._overflow_count += 1
            self._blocks.put(indata.copy())

        self._stream = sd.InputStream(device=self.device, samplerate=self.samplerate,
                                      channels=self.channels, dtype="float32", callback=callback)
        self._stream.start()
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _write_block(self, block: np.ndarray) -> None:
        self._file.write(block)
        self._frames_written += len(block)
        if block.size:
            self._peak = max(self._peak, float(np.max(np.abs(block))))
            self._current_rms = float(np.sqrt(np.mean(block ** 2)))

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                block = self._blocks.get(timeout=0.1)
            except queue.Empty:
                continue
            self._write_block(block)
        # Stream is stopped: flush whatever the callback queued in the meantime.
        while True:
            try:
                block = self._blocks.get_nowait()
            except queue.Empty:
                break
            self._write_block(block)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def level_db(self) -> float:
        return 20 * math.log10(max(self._current_rms, 1e-6))

    def stop(self) -> tuple[float, float, int]:
        """Stop recording and close the file. Returns (duration_s, peak_amplitude, overflow_count)."""
        self._stream.stop()
        self._stream.close()
        self._stop_event.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=5)
        self._file.close()
        duration = self._frames_written / self.samplerate if self.samplerate else 0.0
        return duration, self._peak, self._overflow_count


def record_until_enter(output_path: Path, device: int | str | None = None, channels: int = 1) -> float:
    """CLI flow: record from the microphone to a 16-bit WAV file, stopped by pressing Enter.

    The device's native sample rate is used to avoid "Invalid sample rate" errors;
    faster-whisper resamples to 16 kHz when decoding the file.
    """
    info = sd.query_devices(device, "input")
    print(f"  Microphone: {info['name']} @ {int(info['default_samplerate'])} Hz")
    input("  Press ENTER to start recording...")

    recorder = MicRecorder(output_path, device=device, channels=channels)
    recorder.start()
    print("  Recording... press ENTER to stop.")

    stop_event = threading.Event()

    def wait_for_enter() -> None:
        try:
            input()
        except EOFError:
            pass
        stop_event.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    while not stop_event.is_set():
        _print_meter(recorder.elapsed, recorder.level_db)
        time.sleep(_METER_INTERVAL)
    print()

    duration, peak, overflow_count = recorder.stop()
    if overflow_count:
        print(f"  Warning: {overflow_count} input overflow(s); some audio may be missing.")
    if peak < 0.01:
        print("  Warning: the signal is almost silent. Check the selected microphone and "
              "Windows Settings > Privacy > Microphone (allow desktop apps).")
    if duration < 1.0:
        print("  Warning: the recording is shorter than one second.")
    return duration
