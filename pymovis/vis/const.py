import os

""" Shadow constants """
SHADOW_MAP_SIZE = 8192
BACKGROUND_MAP_SIZE = 512

""" Camera constants """
CAM_TRACK_SENSITIVITY = 0.002
CAM_TUMBLE_SENSITIVITY = 0.002
CAM_ZOOM_SENSITIVITY = 0.05
CAM_DOLLY_SENSITIVITY = 0.2

""" Text constants """
TEXT_RESOLUTION = 256
FONT_DIR_PATH = os.path.join(os.path.dirname(__file__), "../../data/fonts/")
CONSOLAS_FONT_PATH = os.path.join(FONT_DIR_PATH, "consola.ttf")

""" Length conversion """
INCH_TO_METER = 0.0254

""" Material constants """
MAX_MATERIAL_NUM = 5
MAX_MATERIAL_TEXTURES = 25

""" Texture constants """
BACKGROUND_TEXTURE_FILE = "background.hdr"
TEXTURE_DIR_PATH = os.path.join(os.path.dirname(__file__), "../../data/textures/")

""" Skeleton constants """
MAX_JOINT_NUM = 100
LAFAN1_FBX_DICT = {
    "Hips":          "Model:Hips",
    "LeftUpLeg":     "Model:LeftUpLeg",
    "LeftLeg":       "Model:LeftLeg",
    "LeftFoot":      "Model:LeftFoot",
    "LeftToe":       "Model:LeftToe",
    "RightUpLeg":    "Model:RightUpLeg",
    "RightLeg":      "Model:RightLeg",
    "RightFoot":     "Model:RightFoot",
    "RightToe":      "Model:RightToe",
    "Spine":         "Model:Spine",
    "Spine1":        "Model:Spine1",
    "Spine2":        "Model:Spine2",
    "Neck":          "Model:Neck",
    "Head":          "Model:Head",
    "LeftShoulder":  "Model:LeftShoulder",
    "LeftArm":       "Model:LeftArm",
    "LeftForeArm":   "Model:LeftForeArm",
    "LeftHand":      "Model:LeftHand",
    "RightShoulder": "Model:RightShoulder",
    "RightArm":      "Model:RightArm",
    "RightForeArm":  "Model:RightForeArm",
    "RightHand":     "Model:RightHand",
}

YBOT_FBX_DICT = {
    "Hips":          "mixamorig:Hips",
    "LeftUpLeg":     "mixamorig:LeftUpLeg",
    "LeftLeg":       "mixamorig:LeftLeg",
    "LeftFoot":      "mixamorig:LeftFoot",
    "LeftToeBase":   "mixamorig:LeftToeBase",
    "RightUpLeg":    "mixamorig:RightUpLeg",
    "RightLeg":      "mixamorig:RightLeg",
    "RightFoot":     "mixamorig:RightFoot",
    "RightToeBase":  "mixamorig:RightToeBase",
    "Spine":         "mixamorig:Spine",
    "Spine1":        "mixamorig:Spine1",
    "Spine2":        "mixamorig:Spine2",
    "Neck":          "mixamorig:Neck",
    "Head":          "mixamorig:Head",
    "LeftShoulder":  "mixamorig:LeftShoulder",
    "LeftArm":       "mixamorig:LeftArm",
    "LeftForeArm":   "mixamorig:LeftForeArm",
    "LeftHand":      "mixamorig:LeftHand",
    "RightShoulder": "mixamorig:RightShoulder",
    "RightArm":      "mixamorig:RightArm",
    "RightForeArm":  "mixamorig:RightForeArm",
    "RightHand":     "mixamorig:RightHand",
}

""" Model constants """
AXIS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../data/fbx/etc/axis.fbx")