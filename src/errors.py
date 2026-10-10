"""Application-specific exceptions."""


class EditorError(Exception):
    """Base class for all user-facing errors."""


class ConfigError(EditorError):
    """Invalid configuration."""


class FFmpegNotFoundError(EditorError):
    """FFmpeg or FFprobe could not be located."""


class MediaError(EditorError):
    """Missing, corrupt or unsupported media."""


class SyncError(EditorError):
    """Synchronization failed."""


class EDLError(EditorError):
    """Invalid edit decision list."""


class RenderError(EditorError):
    """Rendering failed."""


class OptionalDependencyError(EditorError):
    """An optional dependency is not installed."""
