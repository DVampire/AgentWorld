import logging
from enum import IntEnum
from typing import Any, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree
from rich.logging import RichHandler

from src.utils import Singleton

YELLOW_HEX = "#d4b702"

class LogLevel(IntEnum):
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG

class Logger(logging.Logger, metaclass=Singleton):
    def __init__(self, name="logger", level=logging.INFO):
        # Initialize the parent class
        super().__init__(name, level)

        # Define a formatter for log messages
        self.formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s:%(levelname)s - %(filename)s:%(lineno)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def init_logger(self, config, level: int = LogLevel.INFO):
        """
        Initialize the logger with a file path and optional main process check.

        Args:
            log_path (str): The log file path.
            level (int, optional): The logging level. Defaults to logging.INFO.
            accelerator (Accelerator, optional): Accelerator instance to determine the main process.
        """

        log_path = config.log_path

        self.handlers.clear()

        self.console = Console(
            width=None,
            markup=True,
            color_system="truecolor",
            force_terminal=True
        )
        rich_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=True,
            omit_repeated_times=False
        )
        rich_handler.setLevel(level)
        rich_handler.setFormatter(self.formatter)
        self.addHandler(rich_handler)

        self.file_console = Console(
            width=None,
            markup=True,
            color_system="truecolor",
            force_terminal=True,
            file=open(log_path, "a", encoding="utf-8")
        )
        rich_file_handler = RichHandler(
            console=self.file_console,
            rich_tracebacks=True,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=True,
            omit_repeated_times=False,
        )
        rich_file_handler.setLevel(level)
        rich_file_handler.setFormatter(self.formatter)
        self.addHandler(rich_file_handler)

        self.propagate = False

    def info(self, msg, *args, **kwargs):
        """
        Only for string messages, not for rich objects.
        """
        kwargs.setdefault("stacklevel", 2)

        if "style" in kwargs:
            kwargs.pop("style")
        if "level" in kwargs:
            kwargs.pop("level")
        super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Only for string messages, not for rich objects.
        """
        kwargs.setdefault("stacklevel", 2)
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().debug(msg, *args, **kwargs)

    def log(self,
            msg: Optional[Any] = None,
            level: LogLevel = LogLevel.INFO,
            **kwargs):
        """
        Log a rich object or a string message to both console and file.
        """
        if isinstance(msg, str):
            self.info(msg, **kwargs)
        elif isinstance(msg, (Group, Panel, Rule, Syntax, Table, Tree)):
            self.console.print(msg, **kwargs)
            self.file_console.print(msg, **kwargs)

logger = Logger()