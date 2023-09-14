import os

""" Path constants """
AGL_PATH = os.path.dirname(__file__)

""" Shadow constants """
SHADOW_MAP_SIZE     = 8 * 1024
BACKGROUND_MAP_SIZE = 1024

""" Camera constants """
CAM_TRACK_SENSITIVITY  = 0.002
CAM_TUMBLE_SENSITIVITY = 0.002
CAM_ZOOM_SENSITIVITY   = 0.05
CAM_DOLLY_SENSITIVITY  = 0.2

""" Light constants """
MAX_LIGHT_NUM = 4

""" Text constants """
TEXT_RESOLUTION = 256
FONT_DIR_PATH = os.path.join(AGL_PATH, "data/fonts/")
CONSOLAS_FONT_PATH = os.path.join(FONT_DIR_PATH, "consola.ttf")

""" Length conversion """
INCH_TO_METER = 0.0254

""" Material constants """
MAX_MATERIAL_NUM      = 5
MAX_MATERIAL_TEXTURES = 25

""" Texture constants """
BACKGROUND_TEXTURE_FILE = "background.hdr"
TEXTURE_DIR_PATH        = os.path.join(AGL_PATH, "data/textures/")

""" Skeleton constants """
MAX_JOINT_NUM = 100

""" Model constants """
AXIS_MODEL_PATH = os.path.join(AGL_PATH, "data/fbx/etc/axis.fbx")

""" Other constants """
MAX_INSTANCE_NUM = 100