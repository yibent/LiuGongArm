"""Lightweight contracts; model imports are deferred until a pipeline is used."""
from .types import Detection, FrameResult, RuntimeConfig
from .memory import MemoryView, ObjectMemory, ObjectMemoryStore
__all__ = ['FindTrackPipeline', 'MultiViewFindTrackPipeline', 'Detection', 'FrameResult',
           'RuntimeConfig', 'MemoryView', 'ObjectMemory', 'ObjectMemoryStore']


def __getattr__(name):
    if name == 'FindTrackPipeline':
        from .pipeline import FindTrackPipeline
        return FindTrackPipeline
    if name == 'MultiViewFindTrackPipeline':
        from .multiview import MultiViewFindTrackPipeline
        return MultiViewFindTrackPipeline
    raise AttributeError(name)
