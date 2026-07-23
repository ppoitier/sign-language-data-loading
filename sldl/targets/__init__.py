from sldl.targets.target import TargetEncoder

__all__ = ["TargetEncoder"]

# The concrete target encoders below require optional dependencies (PyTorch,
# and for some of them `sign_language_tools`). They are exported here for
# convenience when those extras are installed, but importing `sldl.targets`
# itself never requires them.
try:
    from sldl.targets.segments import SegmentTarget

    __all__.append("SegmentTarget")
except ImportError:
    pass

try:
    from sldl.targets.frame_labels import FrameLabelsTarget

    __all__.append("FrameLabelsTarget")
except ImportError:
    pass

try:
    from sldl.targets.continuous_recognition import ContinuousRecognitionTarget

    __all__.append("ContinuousRecognitionTarget")
except ImportError:
    pass

try:
    from sldl.targets.temporal_boundary_offset import TemporalBoundaryOffsetsTarget

    __all__.append("TemporalBoundaryOffsetsTarget")
except ImportError:
    pass
