"""Speech-to-text toolkit with interchangeable recognition engines."""

from .engines import ENGINES, Segment, Transcription, get_engine

__all__ = ["ENGINES", "Segment", "Transcription", "get_engine"]
__version__ = "1.0.0"
