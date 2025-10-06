"""Minicap device for high-quality screenshots and video recording."""

import os
import re
import io
import time
import cv2
import av
import numpy as np
import threading
import traceback
import subprocess
import tempfile
import requests

from fractions import Fraction
from datetime import datetime
from PIL import Image
from adbutils import adb

# Constants
GITHUB_PROXYURL = "https://goppx.com/"
GITHUB_BASEURL = "https://github.com/openatx"
MINICAP_BASE_URL = (
    GITHUB_BASEURL + "/stf-binaries/raw/master/node_modules/minicap-prebuilt/prebuilt"
)

MNC_BIN_REMOTE = "/data/local/tmp/minicap"
MNC_SO_REMOTE = "/data/local/tmp/minicap.so"


def download_file(url: str) -> str:
    """Download URL to local temporary file, return file path"""
    print(f"Downloading: {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile("wb+", delete=False) as f:
        for chunk in r.iter_content(1024 * 8):
            f.write(chunk)
        return f.name


class MinicapInstaller:
    def __init__(self, device_id: str = None):
        self.device = adb.device(serial=device_id)
        self.abi = self.get_prop("ro.product.cpu.abi")
        self.sdk = self.get_prop("ro.build.version.sdk")
        if not self.is_installed():
            self.install_minicap()

    def get_prop(self, prop: str) -> str:
        return self.device.shell(f"getprop {prop}").strip()

    @property
    def minicap_urls(self):
        yield f"{MINICAP_BASE_URL}/{self.abi}/lib/android-{self.sdk}/minicap.so"
        yield f"{MINICAP_BASE_URL}/{self.abi}/bin/minicap"

    def install_minicap(self):
        for url in self.minicap_urls:
            try:
                local = download_file(url)
            except Exception:
                local = download_file(GITHUB_PROXYURL + url)
            fname = os.path.basename(url)
            remote = MNC_SO_REMOTE if fname.endswith(".so") else MNC_BIN_REMOTE
            self.device.sync.push(local, remote)
            self.device.shell(f"chmod 777 {remote}")
            os.remove(local)

    def is_installed(self) -> bool:
        for fname, remote in [
            ("minicap", MNC_BIN_REMOTE),
            ("minicap.so", MNC_SO_REMOTE),
        ]:
            out = self.device.shell(f"ls {remote}")
            if fname not in out or "No such file or directory" in out:
                return False
        return True


class MinicapDriver:
    def __init__(
        self,
        device_id: str = None,
        video_save_path: str = ".",
        video_save_name: str = "record",
        fps: int = 2,
        chunk_duration: int = 300,
        video_with_reasoning: bool = True,
    ):
        self.device = adb.device(serial=device_id)
        MinicapInstaller(device_id)
        self.width, self.height = self._get_screen_size()
        self.rotation = self._get_rotation_degree()

        # Control recording state
        self.paused = True
        self._stop_event = threading.Event()
        self._record_thread = None
        self.alive = False  # Recording "alive" flag

        # Video saving parameters
        self.video_save_path = video_save_path
        self.video_save_name = video_save_name
        self.fps = fps
        self.chunk_duration = chunk_duration
        self.video_with_reasoning = video_with_reasoning

        # Statistics
        self.frame_count = 0
        self.chunk_count = 0

    # Support with syntax
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_prop(self, prop: str) -> str:
        return self.device.shell(f"getprop {prop}").strip()

    def _get_screen_size(self) -> tuple:
        out = self.device.shell("wm size")
        m = re.search(r"(\d+)x(\d+)", out)
        if not m:
            raise RuntimeError(f"Cannot parse screen size: {out!r}")
        return int(m.group(1)), int(m.group(2))

    def _get_rotation_degree(self) -> int:
        out = self.device.shell("dumpsys input | grep 'SurfaceOrientation'")
        m = re.search(r"SurfaceOrientation: (\d+)", out)
        if m:
            rot = int(m.group(1))
        else:
            out2 = self.device.shell("dumpsys display")
            m2 = re.search(r"rotation=(\d+)", out2)
            rot = int(m2.group(1)) if m2 else 0
        return (rot % 4) * 90

    def _get_raw_stream(self) -> bytes:
        spec = f"{self.width}x{self.height}@{self.width}x{self.height}/{self.rotation}"
        cmd = [
            "adb",
            "-s",
            self.device.serial,
            "exec-out",
            f"LD_LIBRARY_PATH=/data/local/tmp {MNC_BIN_REMOTE} -s -P {spec}",
        ]
        return subprocess.check_output(cmd)

    def get_screenshot_bytes(self) -> bytes:
        """
        Get raw JPEG bytes without any decoding
        """
        raw = self._get_raw_stream()
        start = raw.find(b"\xff\xd8")
        if start < 0:
            raise RuntimeError("JPEG SOI not found")
        end = raw.rfind(b"\xff\xd9")
        return raw[start : (end + 2) if end > 0 else None]

    def get_screenshot(self) -> Image.Image:
        """
        For compatibility with screenshot mode, return PIL Image
        """
        b = self.get_screenshot_bytes()
        return Image.open(io.BytesIO(b))

    # ------------ Recording Control ------------

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False

    def start_record(self):
        """
        Asynchronously start recording
        """
        if self._record_thread and self._record_thread.is_alive():
            raise RuntimeError("Recording already in progress")
        os.makedirs(self.video_save_path, exist_ok=True)
        self.alive = True
        self._stop_event.clear()
        self._record_thread = threading.Thread(target=self._record_loop, daemon=False)
        self._record_thread.start()
        self.unpause()

    def stop_record(self):
        """
        Stop recording and wait for thread to exit
        """
        if not self._record_thread:
            return
        self.alive = False
        self._stop_event.set()
        self._record_thread.join()
        self._record_thread = None

    def _get_output_path(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.video_save_name}_{timestamp}_{self.chunk_count + 1}.mkv"
        return os.path.join(self.video_save_path, filename)

    def _record_loop(self):
        """
        Recording loop: write frame by frame to MP4, reset PTS for each chunk,
        and use faststart + fragmented moov to ensure playback after interruption.
        Pause periods are automatically clipped out, leaving no blanks in the video.
        """
        main_thread = threading.main_thread()

        while self.alive and main_thread.is_alive():
            output_path = self._get_output_path()
            container = av.open(output_path, mode="w")
            stream = container.add_stream("libx264", rate=self.fps)

            # Set resolution based on rotation
            if self.rotation in (90, 270):
                stream.width, stream.height = self.height, self.width
            else:
                stream.width, stream.height = self.width, self.height
            if self.rotation:
                stream.metadata["rotate"] = str(self.rotation)

            stream.pix_fmt = "yuv444p"
            stream.options = {"crf": "0"}
            stream.codec_context.time_base = Fraction(1, 1000)

            # Record start time of this chunk
            chunk_start = time.time()
            # Start time of pause, for accumulating pause duration
            pause_start = None
            chunk_frame_count = 0
            print(f"Starting async recording -> {output_path}")

            def _close():
                try:
                    for pkt in stream.encode():
                        container.mux(pkt)
                except Exception as e:
                    print(f"Error flushing video stream: {e}")
                    traceback.print_exc()
                finally:
                    container.close()

            try:
                while (
                    self.alive
                    and main_thread.is_alive()
                    and not self._stop_event.is_set()
                ):
                    # If in pause state, start timing and skip capture
                    if self.paused:
                        if pause_start is None:
                            pause_start = time.time()
                        time.sleep(0.1)
                        continue
                    # After resuming from pause, adjust chunk_start to exclude pause time
                    if pause_start is not None:
                        chunk_start += time.time() - pause_start
                        pause_start = None

                    t0 = time.time()
                    try:
                        b = self.get_screenshot_bytes()
                    except (OSError, subprocess.CalledProcessError) as e:
                        print(f"Device error, stopping recording: {e}")
                        traceback.print_exc()
                        self.alive = False
                        self._stop_event.set()
                        break

                    # Decode to ndarray with OpenCV
                    arr = np.frombuffer(b, dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    # Draw subtitles on frame
                    if self.video_with_reasoning:
                        # text = ACTION_INFO.action_args  # or other source
                        # img_bgr = render_subtitle_on_frame(img_bgr, text)
                        # Update resolution in real time
                        h, w = img_bgr.shape[:2]
                        stream.width = w
                        stream.height = h

                    frame = av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

                    # PTS calculation excludes all pause duration
                    pts = int((t0 - chunk_start) * 1000)
                    frame.pts = pts
                    frame.time_base = Fraction(1, 1000)

                    for packet in stream.encode(frame):
                        container.mux(packet)

                    self.frame_count += 1
                    chunk_frame_count += 1

                    elapsed = time.time() - t0
                    time.sleep(max(0, 1 / self.fps - elapsed))

                    if t0 - chunk_start >= self.chunk_duration:
                        break

            except Exception as e:
                print(f"Unexpected error in recording loop: {e}")
                traceback.print_exc()
            finally:
                _close()
                if chunk_frame_count:
                    print(
                        f"Video chunk {self.chunk_count + 1} recording ended, saved to {output_path}, wrote {chunk_frame_count} frames."
                    )
                else:
                    try:
                        os.remove(output_path)
                        print(f"Deleted empty chunk file: {output_path}")
                    except OSError as e:
                        print(f"Failed to delete empty file: {e}")

                self.chunk_count += 1

            if self._stop_event.is_set() or not self.alive:
                break

    def close(self):
        """Equivalent to stop_record"""
        self.stop_record()