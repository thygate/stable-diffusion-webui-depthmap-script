import enum


class GenerationOptions(enum.Enum):
    """This Enum provides the options that are used in the usual generation
    (that is, consumed by the core_generation_funnel).
    Please use this to avoid typos. Also, this enum provides default values for these options."""
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, default_value=None, *args):
        """Saves default value as a member (called "df") of a member of this enum"""
        self.df = default_value

    COMPUTE_DEVICE = "GPU"
    MODEL_TYPE = "res101"  # Will become enum element
    BOOST = True
    NET_SIZE_MATCH = False
    NET_WIDTH = 448
    NET_HEIGHT = 448

    DO_OUTPUT_DEPTH = True
    OUTPUT_DEPTH_INVERT = False
    OUTPUT_DEPTH_COMBINE = False
    OUTPUT_DEPTH_COMBINE_AXIS = "Horizontal"  # Format (str) is subject to change
    DO_OUTPUT_DEPTH_PREDICTION = False  # Hidden, do not use, subject to change

    CLIPDEPTH = False
    CLIPDEPTH_MODE = "Range"
    CLIPDEPTH_FAR = 0.0
    CLIPDEPTH_NEAR = 1.0

    GEN_STEREO = False
    STEREO_MODES = ["left-right", "red-cyan-anaglyph"]
    STEREO_DIVERGENCE = 2.5
    STEREO_SEPARATION = 0.0
    STEREO_FILL_ALGO = "polylines_sharp"
    STEREO_OFFSET_EXPONENT = 2.0
    STEREO_BALANCE = 0.0

    GEN_NORMALMAP = False
    NORMALMAP_PRE_BLUR = False
    NORMALMAP_PRE_BLUR_KERNEL = 3
    NORMALMAP_SOBEL = True
    NORMALMAP_SOBEL_KERNEL = 3
    NORMALMAP_POST_BLUR = False
    NORMALMAP_POST_BLUR_KERNEL = 3
    NORMALMAP_INVERT = False

    GEN_HEATMAP = False

    GEN_SIMPLE_MESH = False
    SIMPLE_MESH_OCCLUDE = True
    SIMPLE_MESH_SPHERICAL = False

    GEN_INPAINTED_MESH = False
    GEN_INPAINTED_MESH_DEMOS = False

    GEN_REMBG = False
    SAVE_BACKGROUND_REMOVAL_MASKS = False  # Legacy, will be reworked
    PRE_DEPTH_BACKGROUND_REMOVAL = False  # Legacy, will be reworked
    REMBG_MODEL = "u2net"
