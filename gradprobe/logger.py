"""
Logging system for GradProbe.

Provides a configurable logger to control console output verbosity.
This allows filtering of memory/VRAM monitoring, performance metrics,
and troubleshooting information separately from normal progress messages.
"""

import sys
import os
import atexit
from typing import Optional, TextIO, Union, Tuple, List
from multiprocessing import Process, Queue
from datetime import datetime


class LogLevel:
    """Log level constants."""
    ERROR = 40
    WARNING = 30
    INFO = 20
    MEMORY = 15
    DEBUG = 10

    _NAMES = {
        ERROR: "ERROR",
        WARNING: "WARNING",
        INFO: "INFO",
        MEMORY: "MEMORY",
        DEBUG: "DEBUG",
    }

    @classmethod
    def get_name(cls, level: int) -> str:
        """Get the name of a log level."""
        return cls._NAMES.get(level, f"LEVEL{level}")

    @classmethod
    def from_name(cls, name: str) -> int:
        """Get log level from name (case-insensitive)."""
        name = name.upper()
        for level, level_name in cls._NAMES.items():
            if level_name == name:
                return level
        raise ValueError(f"Unknown log level: {name}")


class Logger:
    """
    Logger for GradProbe with configurable verbosity levels.

    Log levels (from least to most verbose):
    - ERROR (40): Error messages only
    - WARNING (30): Warnings and errors
    - INFO (20): Normal progress, configuration, results (default)
    - MEMORY (15): Memory/VRAM monitoring + all above
    - DEBUG (10): Detailed troubleshooting + all above

    Level filtering supports two modes:
    - Threshold mode: Pass an int (e.g., LogLevel.INFO) to show that level and above
    - Specific mode: Pass tuple/list (e.g., (LogLevel.INFO, LogLevel.ERROR)) to show only those

    File logging:
    - Enabled by default, logs written to logs/<program_name>_<timestamp>.log
    - Uses multiprocessing with unbounded queue for non-blocking writes
    - All formatting and filtering done in background worker process
    - By default, all log levels are written to file

    Example usage:
        # Threshold mode (default)
        logger = Logger(
            level=LogLevel.INFO,        # Console: INFO and above
            program_name="prune_mlp"
        )

        # Specific levels only
        logger = Logger(
            level=(LogLevel.INFO, LogLevel.ERROR),  # Console: only INFO and ERROR
            file_log_level=(LogLevel.DEBUG, LogLevel.MEMORY),  # File: only DEBUG and MEMORY
            program_name="prune_mlp"
        )

        # Disable file logging
        logger = Logger(
            level=LogLevel.INFO,
            program_name="prune_mlp",
            enable_file_logging=False
        )
    """

    # Global instance for easy access throughout the codebase
    _instance: Optional['Logger'] = None

    def __init__(
        self,
        level: Union[int, Tuple[int, ...], List[int]] = LogLevel.INFO,
        program_name: Optional[str] = None,
        file: Optional[TextIO] = None,
        enable_file_logging: bool = True,
        file_log_level: Union[int, Tuple[int, ...], List[int]] = LogLevel.DEBUG,
        log_dir: str = "logs"
    ):
        """
        Initialize the logger.

        Args:
            level: Console log level filter (int for threshold, tuple/list for specific levels)
            program_name: Name of the program (used for log file naming)
            file: Output file handle for console (default: sys.stdout)
            enable_file_logging: Whether to enable logging to disk (default: True)
            file_log_level: File log level filter (int for threshold, tuple/list for specific)
            log_dir: Directory for log files (default: "logs")
        """
        self.level = level
        self.program_name = program_name or "gradprobe"
        self.file = file or sys.stdout
        self.file_log_level = file_log_level
        self.log_dir = log_dir

        # File logging state
        self.enable_file_logging = enable_file_logging
        self.log_queue: Optional[Queue] = None
        self.log_process: Optional[Process] = None
        self.log_file_path: Optional[str] = None

        if self.enable_file_logging:
            self._start_file_logging()

        # Store as global instance
        Logger._instance = self

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def _start_file_logging(self):
        """Start the background file logging process."""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(
            self.log_dir,
            f"{self.program_name}_{timestamp}.log"
        )

        # Create unbounded queue (maxsize=0 means unlimited)
        self.log_queue = Queue(maxsize=0)

        # Start background process with file_log_level for filtering
        self.log_process = Process(
            target=_file_logger_worker,
            args=(self.log_queue, self.log_file_path, self.file_log_level),
            daemon=True  # Will be killed if main process exits
        )
        self.log_process.start()

    def cleanup(self):
        """Clean up file logging resources."""
        if self.log_process and self.log_process.is_alive():
            # Send sentinel to stop the worker
            self.log_queue.put(None)
            # Wait for worker to finish writing
            self.log_process.join(timeout=5.0)
            if self.log_process.is_alive():
                # Force terminate if it didn't stop
                self.log_process.terminate()
                self.log_process.join(timeout=1.0)

    @classmethod
    def get_instance(cls) -> 'Logger':
        """
        Get the global logger instance.

        Returns a default logger if none has been initialized.
        """
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance

    def set_level(self, level: Union[int, Tuple[int, ...], List[int]]):
        """Set the console log level filter."""
        self.level = level

    def set_level_from_name(self, name: str):
        """Set the console log level from a name string."""
        self.level = LogLevel.from_name(name)

    def _should_log(self, level: int, filter_level: Union[int, Tuple[int, ...], List[int]]) -> bool:
        """
        Check if a message at the given level should be logged.

        Args:
            level: The level of the message
            filter_level: The filter (int for threshold, tuple/list for specific levels)

        Returns:
            True if the message should be logged
        """
        if isinstance(filter_level, (tuple, list)):
            # Specific levels mode
            return level in filter_level
        else:
            # Threshold mode
            return level >= filter_level

    def _log(self, level: int, message: str):
        """Internal logging method."""
        # Console output
        if self._should_log(level, self.level):
            print(message, file=self.file)

        # File output - send raw (level, message) to worker
        if self.enable_file_logging:
            # Non-blocking put to queue - worker does filtering and formatting
            self.log_queue.put((level, message))

    def error(self, message: str):
        """Log an error message."""
        self._log(LogLevel.ERROR, message)

    def warning(self, message: str):
        """Log a warning message."""
        self._log(LogLevel.WARNING, message)

    def info(self, message: str):
        """Log an informational message (normal progress, configuration, results)."""
        self._log(LogLevel.INFO, message)

    def memory(self, message: str):
        """Log memory/VRAM monitoring information."""
        self._log(LogLevel.MEMORY, message)

    def debug(self, message: str):
        """Log detailed debugging/troubleshooting information."""
        self._log(LogLevel.DEBUG, message)

    def is_enabled_for(self, level: int) -> bool:
        """Check if a log level is enabled for console output."""
        return self._should_log(level, self.level)


def _file_logger_worker(
    queue: Queue,
    log_file_path: str,
    file_log_level: Union[int, Tuple[int, ...], List[int]]
):
    """
    Background worker process for writing logs to disk.

    This runs in a separate process and consumes log entries from the queue,
    filtering, formatting, and writing them to disk. This ensures the main
    process never blocks on disk I/O or formatting overhead.

    Args:
        queue: Queue to consume (level, message) tuples from
        log_file_path: Path to the log file
        file_log_level: Log level filter (int or tuple/list)
    """
    def should_log(level: int) -> bool:
        """Check if level should be logged to file."""
        if isinstance(file_log_level, (tuple, list)):
            return level in file_log_level
        else:
            return level >= file_log_level

    with open(log_file_path, 'w', encoding='utf-8') as f:
        while True:
            entry = queue.get()

            # None is the sentinel value to stop
            if entry is None:
                break

            # entry is (level, message)
            level, message = entry

            # Filter by level
            if not should_log(level):
                continue

            # Format the log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            level_name = LogLevel.get_name(level)
            log_line = f"[{timestamp}] [{level_name}] {message}"

            # Write to file
            f.write(log_line + '\n')
            f.flush()  # Ensure immediate write to disk


# Convenience function for getting the global logger
def get_logger() -> Logger:
    """Get the global logger instance."""
    return Logger.get_instance()
