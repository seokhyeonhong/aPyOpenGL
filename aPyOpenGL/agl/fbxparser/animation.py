from __future__ import annotations

import fbx
import glm

from .keyframe import KeyInterpType, Keyframe, NodeKeyframes, SceneKeyframes

FbxAnimLayer = fbx.FbxAnimLayer
FbxCriteria  = fbx.FbxCriteria
FbxEuler     = fbx.FbxEuler
FbxNode      = fbx.FbxNode

def get_scene_animation(anim_stack, node, scale):
    criteria = FbxCriteria.ObjectType(FbxAnimLayer.ClassId)
    num_anim_layers = anim_stack.GetMemberCount(criteria)

    if num_anim_layers > 1:
        print(f"Warning: {num_anim_layers} animation layers found, only the first one will be used")
    
    scene_keyframes = SceneKeyframes(anim_stack.GetName())
    anim_layer = anim_stack.GetMember(criteria, 0)
    get_animations(scene_keyframes, anim_layer, node, scale)
    
    time_mode = anim_stack.GetScene().GetGlobalSettings().GetTimeMode()
    scene_keyframes.start_frame = anim_stack.LocalStart.Get().GetFrameCount(time_mode)
    scene_keyframes.end_frame = anim_stack.LocalStop.Get().GetFrameCount(time_mode)
    scene_keyframes.fps = fbx.FbxTime.GetFrameRate(time_mode)
    
    return scene_keyframes

def get_animations(scene_kfs: SceneKeyframes, anim_layer, node, scale):
    animation = get_keyframe_animation(node, anim_layer, scale)
    scene_kfs.node_keyframes.append(animation)
    for i in range(node.GetChildCount()):
        get_animations(scene_kfs, anim_layer, node.GetChild(i), scale)

def get_keyframe_animation(node, anim_layer, scale) -> NodeKeyframes:
    node_kfs = NodeKeyframes(node.GetName())

    # rotation order
    order = node.GetRotationOrder(FbxNode.eSourcePivot)
    if order == FbxEuler.eOrderXYZ:
        node_kfs.euler_order = glm.ivec3(0, 1, 2)
    elif order == FbxEuler.eOrderXZY:
        node_kfs.euler_order = glm.ivec3(0, 2, 1)
    elif order == FbxEuler.eOrderYXZ:
        node_kfs.euler_order = glm.ivec3(1, 0, 2)
    elif order == FbxEuler.eOrderYZX:
        node_kfs.euler_order = glm.ivec3(1, 2, 0)
    elif order == FbxEuler.eOrderZXY:
        node_kfs.euler_order = glm.ivec3(2, 0, 1)
    elif order == FbxEuler.eOrderZYX:
        node_kfs.euler_order = glm.ivec3(2, 1, 0)
    else:
        print(f"Warning: unsupported rotation order {order}")
    
    # get keyframes
    for i, channel in enumerate(["X", "Y", "Z"]):
        anim_curve = node.LclTranslation.GetCurve(anim_layer, channel)
        if anim_curve:
            node_kfs.pos[i] = get_keyframes(anim_curve, scale)
        
        anim_curve = node.LclRotation.GetCurve(anim_layer, channel)
        if anim_curve:
            node_kfs.euler[i] = get_keyframes(anim_curve, 1.0)
        
        anim_curve = node.LclScaling.GetCurve(anim_layer, channel)
        if anim_curve:
            node_kfs.scale[i] = get_keyframes(anim_curve, 1.0)

    return node_kfs

def get_keyframes(anim_curve, scale) -> list[Keyframe]:
    keys = []
    for i in range(anim_curve.KeyGetCount()):
        value = scale * anim_curve.KeyGetValue(i)
        frame = anim_curve.KeyGetTime(i).GetFrameCount()
        type = get_interpolation_type(anim_curve.KeyGetInterpolation(i))

        key = Keyframe(value, frame, type)
        keys.append(key)
        
    return keys

def get_interpolation_type(flag):
    FbxAnimCurveDef = fbx.FbxAnimCurveDef
    if (flag & FbxAnimCurveDef.eInterpolationConstant) == FbxAnimCurveDef.eInterpolationConstant:
        return KeyInterpType.eCONSTANT
    if (flag & FbxAnimCurveDef.eInterpolationLinear) == FbxAnimCurveDef.eInterpolationLinear:
        return KeyInterpType.eLINEAR
    if (flag & FbxAnimCurveDef.eInterpolationCubic) == FbxAnimCurveDef.eInterpolationCubic:
        return KeyInterpType.eCUBIC
    return KeyInterpType.eUNKNOWN