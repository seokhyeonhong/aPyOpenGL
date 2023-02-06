import fbx
import glm
import numpy as np

from pymovis.motion.data.fbx_parser import JointData

def to_quat(x, order):
    rx = glm.angleAxis(np.deg2rad(x[0], dtype=np.float32), glm.vec3(1, 0, 0))
    ry = glm.angleAxis(np.deg2rad(x[1], dtype=np.float32), glm.vec3(0, 1, 0))
    rz = glm.angleAxis(np.deg2rad(x[2], dtype=np.float32), glm.vec3(0, 0, 1))

    if order == fbx.FbxEuler.eOrderXYZ:
        return rz * ry * rx
    if order == fbx.FbxEuler.eOrderXZY:
        return ry * rz * rx
    if order == fbx.FbxEuler.eOrderYXZ:
        return rz * rx * ry
    if order == fbx.FbxEuler.eOrderYZX:
        return rx * rz * ry
    if order == fbx.FbxEuler.eOrderZXY:
        return ry * rx * rz
    if order == fbx.FbxEuler.eOrderZYX:
        return rx * ry * rz

    raise ValueError("Unknown Euler order: {}".format(order))

def to_vec3(x):
    return glm.vec3(x[0], x[1], x[2])

def parse_nodes_by_type(node, joints, parent_idx, type, scale):
    is_type = False
    for i in range(node.GetNodeAttributeCount()):
        attr = node.GetNodeAttributeByIndex(i)
        if attr.GetAttributeType() == type:
            is_type = True
            break
    
    if is_type:
        name = node.GetName()
        order = node.GetRotationOrder(fbx.FbxNode.eDestinationPivot)
        local_T = to_vec3(node.LclTranslation.Get())
        local_R = to_quat(node.LclRotation.Get(), order)
        local_S = to_vec3(node.LclScaling.Get())

        dest_pre_R = node.GetPreRotation(fbx.FbxNode.eDestinationPivot)
        pre_R = to_quat(dest_pre_R, order)

        joint = JointData()
        joint.name = name
        joint.local_T = scale * local_T
        joint.local_R = local_R
        joint.pre_R = pre_R
        joint.local_S = local_S
        joint.parent_idx = parent_idx
        joints.append(joint)
        parent_idx = len(joints) - 1
    else:
        return
    
    for i in range(node.GetChildCount()):
        parse_nodes_by_type(node.GetChild(i), joints, parent_idx, type, scale)