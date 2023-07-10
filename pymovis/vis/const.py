import os

""" Path constants """
PYMOVIS_PATH = os.path.dirname(__file__)

""" Shadow constants """
SHADOW_MAP_SIZE     = 4 * 1024
BACKGROUND_MAP_SIZE = 512

""" Camera constants """
CAM_TRACK_SENSITIVITY  = 0.002
CAM_TUMBLE_SENSITIVITY = 0.002
CAM_ZOOM_SENSITIVITY   = 0.05
CAM_DOLLY_SENSITIVITY  = 0.2

""" Text constants """
TEXT_RESOLUTION = 256
FONT_DIR_PATH = os.path.join(os.path.dirname(__file__), "../../data/fonts/")
CONSOLAS_FONT_PATH = os.path.join(FONT_DIR_PATH, "consola.ttf")

""" Length conversion """
INCH_TO_METER = 0.0254

""" Material constants """
MAX_MATERIAL_NUM      = 5
MAX_MATERIAL_TEXTURES = 25

""" Texture constants """
BACKGROUND_TEXTURE_FILE = "background.hdr"
TEXTURE_DIR_PATH        = os.path.join(os.path.dirname(__file__), "../../data/textures/")

""" Skeleton constants """
MAX_JOINT_NUM = 100

""" Model constants """
AXIS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../data/fbx/etc/axis.fbx")