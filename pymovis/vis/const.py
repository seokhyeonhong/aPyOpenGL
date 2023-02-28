""" Shadow constants """
SHADOW_MAP_SIZE = 2048

""" Camera constants """
CAM_TRACK_SENSITIVITY = 0.002
CAM_TUMBLE_SENSITIVITY = 0.002
CAM_ZOOM_SENSITIVITY = 0.05
CAM_DOLLY_SENSITIVITY = 0.2

""" Text constants """
TEXT_RESOLUTION = 64

""" Length conversion """
INCH_TO_METER = 0.0254

""" Material constants """
MAX_MATERIAL_NUM = 5
MAX_MATERIAL_TEXTURES = 25

""" Skeleton constants """
MAX_JOINT_NUM = 100
LAFAN_BVH_TO_FBX = {
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

LAFAN_BVH_TO_YBOT = {
    "Hips":          "mixamorig:Hips",
    "LeftUpLeg":     "mixamorig:LeftUpLeg",
    "LeftLeg":       "mixamorig:LeftLeg",
    "LeftFoot":      "mixamorig:LeftFoot",
    "LeftToe":       "mixamorig:LeftToeBase",
    "RightUpLeg":    "mixamorig:RightUpLeg",
    "RightLeg":      "mixamorig:RightLeg",
    "RightFoot":     "mixamorig:RightFoot",
    "RightToe":      "mixamorig:RightToeBase",
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