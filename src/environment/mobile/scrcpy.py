"""Scrcpy driver for advanced device control."""

import asyncio
import functools
import os
import socket
import struct
import threading
import time
from fractions import Fraction
from typing import Any, Callable, Optional, Tuple
from datetime import datetime

import av
import numpy as np
from PIL import Image
from adbutils import AdbConnection, AdbError, Network, adb
from av import VideoFrame
from av.codec import CodecContext
from av.error import InvalidDataError


# Actions
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2

# Type
TYPE_INJECT_KEYCODE = 0
TYPE_INJECT_TEXT = 1
TYPE_INJECT_TOUCH_EVENT = 2
TYPE_INJECT_SCROLL_EVENT = 3
TYPE_BACK_OR_SCREEN_ON = 4
TYPE_EXPAND_NOTIFICATION_PANEL = 5
TYPE_EXPAND_SETTINGS_PANEL = 6
TYPE_COLLAPSE_PANELS = 7
TYPE_GET_CLIPBOARD = 8
TYPE_SET_CLIPBOARD = 9
TYPE_SET_SCREEN_POWER_MODE = 10
TYPE_ROTATE_DEVICE = 11

# Events
EVENT_INIT = "init"
EVENT_FRAME = "frame"
EVENT_DISCONNECT = "disconnect"

# Lock screen orientation
LOCK_SCREEN_ORIENTATION_UNLOCKED = -1
LOCK_SCREEN_ORIENTATION_INITIAL = -2
LOCK_SCREEN_ORIENTATION_0 = 0
LOCK_SCREEN_ORIENTATION_1 = 1
LOCK_SCREEN_ORIENTATION_2 = 2
LOCK_SCREEN_ORIENTATION_3 = 3


class ScrcpyDriver:
    """
    Driver class for managing the scrcpy client and video saving functionalities.
    It initializes the scrcpy client, starts the video stream, and handles saving frames to a video file.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        video_save_path: Optional[str] = None,
        video_save_name: Optional[str] = None,
        fps: int = 60,
        bitrate: int = 50_000_000,  # 50 Mbps
        chunk_duration: Optional[float] = None,
    ):
        # Initialize Scrcpy client
        self.client = ScrcpyClient(
            device=device,
            max_fps=fps,
            bitrate=bitrate,
            encoder_name="c2.android.avc.encoder",
            codec_name="h264",
        )
        self.client.start(threaded=True)

        # Video saving attributes
        self.video_save_path = video_save_path
        self.video_save_name = video_save_name
        self.chunk_duration = chunk_duration
        self.fps = fps
        self.screenshot_time = None
        self.start_time = None
        self.last_frame_time = None

        # Pause related
        self.paused = False
        self._pause_start_time = None
        self._total_paused_duration = 0

        self.output_lock = threading.Lock()
        
        # Initialize video saving if path is specified
        if video_save_path:
            self.new_output_container()
            self.client.add_listener(EVENT_FRAME, self.save_frame_to_file)

    def new_output_container(self):
        """
        Create a new output container (file) for saving the next video segment.
        Each chunk file is named with timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(self.video_save_path, exist_ok=True)
        chunk_path = os.path.join(
            self.video_save_path, f"{self.video_save_name}_{timestamp}.mp4"
        )
        self.output_container = av.open(chunk_path, "w")
        self.output_stream = self.output_container.add_stream("libx264", rate=self.fps)
        self.output_stream.pix_fmt = "yuv444p"
        self.output_stream.options = {"crf": "0"}  # Lossless quality

        # Reset time base and pause accumulation duration
        self.start_time = None
        self._total_paused_duration = 0

    async def tap(self, x: int, y: int, screen_size: Tuple[int, int]):
        """Tap at specified coordinates."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.client.touch, x, y, screen_size[0], screen_size[1], ACTION_DOWN
        )
        await asyncio.get_event_loop().run_in_executor(
            None, self.client.touch, x, y, screen_size[0], screen_size[1], ACTION_UP
        )

    async def long_press(self, x: int, y: int, screen_size: Tuple[int, int], duration: float = 1.0):
        """Long press at specified coordinates."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.client.touch, x, y, screen_size[0], screen_size[1], ACTION_DOWN
        )
        await asyncio.sleep(duration)
        await asyncio.get_event_loop().run_in_executor(
            None, self.client.touch, x, y, screen_size[0], screen_size[1], ACTION_UP
        )

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float,
        screen_size: Tuple[int, int],
    ):
        """Swipe gesture from start to end coordinates."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.client.touch, start_x, start_y, screen_size[0], screen_size[1], ACTION_DOWN
        )
        
        steps = 30
        delay = duration / steps
        delta_x = (end_x - start_x) / steps
        delta_y = (end_y - start_y) / steps
        
        for i in range(1, steps):
            x = int(start_x + delta_x * i)
            y = int(start_y + delta_y * i)
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.touch, x, y, screen_size[0], screen_size[1], ACTION_MOVE
            )
            await asyncio.sleep(delay)
        
        await asyncio.get_event_loop().run_in_executor(
            None, self.client.touch, end_x, end_y, screen_size[0], screen_size[1], ACTION_UP
        )

    async def get_screenshot(self) -> Optional[Image.Image]:
        """Get latest frame as PIL Image."""
        if self.client.last_frame is None:
            return None
        
        # Avoid reading the same frame repeatedly
        while self.screenshot_time == self.client.last_frame_time:
            await asyncio.sleep(0.01)
        
        arr = self.client.last_frame[..., ::-1]
        self.screenshot_time = self.client.last_frame_time
        return Image.fromarray(arr)

    def save_frame_to_file(self, frame):
        """Save frame to video file."""
        if frame is None or self.paused:
            return

        with self.output_lock:
            now = time.time()
            if self.start_time is None:
                self.start_time = now
                frame_time = 0.0
            else:
                frame_time = now - self.start_time - self._total_paused_duration

            # Update resolution in real time
            h, w = frame.shape[:2]
            self.output_stream.width = w
            self.output_stream.height = h

            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.time_base = Fraction(1, self.fps)
            video_frame.pts = int(frame_time * video_frame.time_base.denominator)

            # Encode and write
            for packet in self.output_stream.encode(video_frame):
                self.output_container.mux(packet)

            # Create new chunk if duration exceeded
            if self.chunk_duration and frame_time >= self.chunk_duration:
                for packet in self.output_stream.encode(None):
                    self.output_container.mux(packet)
                self.output_container.close()
                self.new_output_container()

    async def pause(self):
        """Pause recording."""
        if not self.paused:
            self.paused = True
            self._pause_start_time = time.time()

    async def unpause(self):
        """Resume recording."""
        if self.paused:
            paused_duration = time.time() - self._pause_start_time
            self._total_paused_duration += paused_duration
            self._pause_start_time = None
            self.paused = False

    async def close(self):
        """Close client and output container."""
        if hasattr(self, "client") and self.client:
            self.client.stop()
            del self.client

        if hasattr(self, "output_container") and self.output_container:
            try:
                # Flush encoder
                for packet in self.output_stream.encode(None):
                    self.output_container.mux(packet)
                self.output_container.close()
                del self.output_container
            except Exception as e:
                print(f"Error closing output container: {e}")

    def __del__(self):
        if hasattr(self, "client") and self.client:
            self.client.stop()


# from https://github.com/leng-yue/py-scrcpy-client/blob/main/scrcpy/core.py
def action_inject(control_type: int):
    def wrapper(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            package = struct.pack(">B", control_type) + f(*args, **kwargs)
            if args[0].control_socket is not None:
                try:
                    with args[0].control_socket_lock:
                        bytes_sent = args[0].control_socket.send(package)
                        if bytes_sent != len(package):
                            print(
                                f"Warning: incomplete send, only sent {bytes_sent}/{len(package)} bytes"
                            )
                except socket.error as e:
                    print(f"Send error: {e}")
            else:
                print("Warning: control_socket not initialized")
            return package

        return inner

    return wrapper


class ScrcpyClient:
    """
    Scrcpy client class for managing the connection and communication with the scrcpy server on the Android device.
    It handles deploying the server, initializing connections, and streaming video frames.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        max_width: int = 0,
        bitrate: int = 8000000,
        max_fps: int = 0,
        flip: bool = False,
        block_frame: bool = False,
        stay_awake: bool = False,
        lock_screen_orientation: int = LOCK_SCREEN_ORIENTATION_UNLOCKED,
        connection_timeout: int = 3000,
        encoder_name: str = "OMX.google.h264.encoder",
        codec_name: str = "h264",
    ):
        """
        Initializes the ScrcpyClient with the given parameters.

        Args:
            device: Android device, select first one if none, from serial if str.
            max_width: Frame width that will be broadcast from android server.
            bitrate: Bitrate.
            max_fps: Maximum fps, 0 means not limited (supported after android 10).
            flip: Flip the video.
            block_frame: Only return nonempty frames.
            stay_awake: Keep Android device awake.
            lock_screen_orientation: Lock screen orientation, LOCK_SCREEN_ORIENTATION_*.
            connection_timeout: Timeout for connection, unit is ms.
            encoder_name: Encoder name, enum: [OMX.google.h264.encoder, OMX.qcom.video.encoder.avc, c2.qti.avc.encoder, c2.android.avc.encoder], default is OMX.google.h264.encoder.
            codec_name: Codec name, enum: [h264, h265, av1], default is h264.
        """
        # Check Params
        assert max_width >= 0, "max_width must be greater than or equal to 0"
        assert bitrate >= 0, "bitrate must be greater than or equal to 0"
        assert max_fps >= 0, "max_fps must be greater than or equal to 0"
        assert -1 <= lock_screen_orientation <= 3, (
            "lock_screen_orientation must be LOCK_SCREEN_ORIENTATION_*"
        )
        assert connection_timeout >= 0, (
            "connection_timeout must be greater than or equal to 0"
        )
        assert encoder_name in [
            "OMX.google.h264.encoder",
            "OMX.qcom.video.encoder.avc",
            "c2.qti.avc.encoder",
            "c2.android.avc.encoder",
        ]
        assert codec_name in ["h264", "h265", "av1"]

        # Params
        self.flip = flip
        self.max_width = max_width
        self.bitrate = bitrate
        self.max_fps = max_fps
        self.block_frame = block_frame
        self.stay_awake = stay_awake
        self.lock_screen_orientation = lock_screen_orientation
        self.connection_timeout = connection_timeout
        self.encoder_name = encoder_name
        self.codec_name = codec_name

        # Connect to device
        if device is None:
            devices = adb.device_list()
            assert len(devices) > 0, "No adb device connected."
            device = devices[0]
        elif isinstance(device, str):
            device = adb.device(serial=device)

        self.device = device
        self.listeners = dict(frame=[], init=[], disconnect=[])

        # User accessible
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: Optional[int] = None
        self.resolution: Optional[Tuple[int, int]] = None
        self.device_name: Optional[str] = None

        # Need to destroy
        self.alive = False
        self.__server_stream: Optional[AdbConnection] = None
        self.__video_socket: Optional[socket.socket] = None
        self.control_socket: Optional[socket.socket] = None
        self.control_socket_lock = threading.Lock()

        # Available if start with threaded or daemon_threaded
        self.stream_loop_thread = None

    def __init_server_connection(self) -> None:
        """
        Connect to android server, there will be two sockets, video and control socket.
        This method will set: video_socket, control_socket, resolution variables
        """
        for _ in range(self.connection_timeout // 100):
            try:
                self.__video_socket = self.device.create_connection(
                    Network.LOCAL_ABSTRACT, "scrcpy"
                )
                break
            except AdbError:
                time.sleep(0.1)
                pass
        else:
            raise ConnectionError("Failed to connect scrcpy-server after 3 seconds")

        dummy_byte = self.__video_socket.recv(1)
        if not len(dummy_byte) or dummy_byte != b"\x00":
            raise ConnectionError("Did not receive Dummy Byte!")

        self.control_socket = self.device.create_connection(
            Network.LOCAL_ABSTRACT, "scrcpy"
        )
        self.device_name = self.__video_socket.recv(64).decode("utf-8").rstrip("\x00")
        if not len(self.device_name):
            raise ConnectionError("Did not receive Device Name!")

        res = self.__video_socket.recv(4)
        self.resolution = struct.unpack(">HH", res)
        self.__video_socket.setblocking(False)

    def __deploy_server(self) -> None:
        """
        Deploy server to android device
        """
        jar_name = "scrcpy-server.jar"
        server_file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), jar_name
        )
        self.device.sync.push(server_file_path, f"/data/local/tmp/{jar_name}")
        commands = [
            f"CLASSPATH=/data/local/tmp/{jar_name}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            "2.4",  # Scrcpy server version
            "log_level=info",
            f"max_size={self.max_width}",
            f"max_fps={self.max_fps}",
            f"video_bit_rate={self.bitrate}",
            f"video_encoder={self.encoder_name}",
            f"video_codec={self.codec_name}",
            "tunnel_forward=true",
            "send_frame_meta=false",
            "control=true",
            "audio=false",
            "show_touches=false",
            "stay_awake=false",
            "power_off_on_close=false",
            "clipboard_autosync=false",
        ]

        self.__server_stream: AdbConnection = self.device.shell(
            commands,
            stream=True,
        )

        # Wait for server to start
        self.__server_stream.read(10)

    def start(self, threaded: bool = False, daemon_threaded: bool = False) -> None:
        """
        Start listening video stream

        Args:
            threaded: Run stream loop in a different thread to avoid blocking
            daemon_threaded: Run stream loop in a daemon thread to avoid blocking
        """
        assert self.alive is False

        self.__deploy_server()
        self.__init_server_connection()
        self.alive = True
        self.__send_to_listeners(EVENT_INIT)

        if threaded or daemon_threaded:
            self.stream_loop_thread = threading.Thread(
                target=self.__stream_loop, daemon=daemon_threaded
            )
            self.stream_loop_thread.start()
        else:
            self.__stream_loop()

    def stop(self) -> None:
        """
        Stop listening (both threaded and blocked)
        """
        self.alive = False
        self.__send_to_listeners(EVENT_DISCONNECT)
        if self.__server_stream is not None:
            try:
                self.__server_stream.close()
            except Exception:
                pass

        if self.control_socket is not None:
            try:
                self.control_socket.close()
            except Exception:
                pass

        if self.__video_socket is not None:
            try:
                self.__video_socket.close()
            except Exception:
                pass

    def __stream_loop(self) -> None:
        """
        Core loop for video parsing
        """
        codec = CodecContext.create(self.codec_name, "r")
        main_thread = threading.main_thread()  # Get main thread object
        while self.alive and main_thread.is_alive():
            try:
                raw_h264 = self.__video_socket.recv(0x10000)
                if raw_h264 == b"":
                    self.alive = False
                    raise ConnectionError("Video stream is disconnected")
                packets = codec.parse(raw_h264)
                for packet in packets:
                    frames = codec.decode(packet)
                    for frame in frames:
                        frame = frame.to_ndarray(format="bgr24")
                        if self.flip:
                            frame = frame[:, ::-1, :]
                        self.last_frame = frame
                        self.last_frame_time = time.time()
                        self.resolution = (frame.shape[1], frame.shape[0])
                        self.__send_to_listeners(EVENT_FRAME, frame)
            except (BlockingIOError, InvalidDataError):
                time.sleep(0.01)
                if not self.block_frame:
                    self.__send_to_listeners(EVENT_FRAME, None)
            except (ConnectionError, OSError) as e:  # Socket Closed
                if self.alive:
                    self.stop()
                    raise e

    def add_listener(self, cls: str, listener: Callable[..., Any]) -> None:
        """
        Add a video listener

        Args:
            cls: Listener category, support: init, frame
            listener: A function to receive frame np.ndarray
        """
        self.listeners[cls].append(listener)

    def remove_listener(self, cls: str, listener: Callable[..., Any]) -> None:
        """
        Remove a video listener

        Args:
            cls: Listener category, support: init, frame
            listener: A function to receive frame np.ndarray
        """
        self.listeners[cls].remove(listener)

    def __send_to_listeners(self, cls: str, *args, **kwargs) -> None:
        """
        Send event to listeners

        Args:
            cls: Listener type
            *args: Other arguments
            *kwargs: Other arguments
        """
        for fun in self.listeners[cls]:
            fun(*args, **kwargs)

    @action_inject(TYPE_INJECT_TOUCH_EVENT)
    def touch(
        self,
        x: int,
        y: int,
        resolution_x: int,
        resolution_y: int,
        action: int = ACTION_DOWN,
        touch_id: int = 0x1234567887654321,
    ) -> bytes:
        """
        Touch screen

        Args:
            x: horizontal position
            y: vertical position
            action: ACTION_DOWN | ACTION_UP | ACTION_MOVE
            touch_id: Default using virtual id -1, you can specify it to emulate multi finger touch
        """
        x, y = max(x, 0), max(y, 0)
        return struct.pack(
            ">BqiiHHHii",
            action,
            touch_id,
            int(x),
            int(y),
            int(resolution_x),
            int(resolution_y),
            0xFFFF,
            1,
            1,
        )

    @action_inject(TYPE_INJECT_SCROLL_EVENT)
    def scroll(
        self,
        x: int,
        y: int,
        h: int,
        v: int,
        resolution_x: int,
        resolution_y: int,
    ) -> bytes:
        """
        Scroll screen

        Args:
            x: horizontal position
            y: vertical position
            h: horizontal movement
            v: vertical movement
        """

        x, y = max(x, 0), max(y, 0)
        return struct.pack(
            ">iiHHii",
            int(x),
            int(y),
            int(resolution_x),
            int(resolution_y),
            int(h),
            int(v),
        )
