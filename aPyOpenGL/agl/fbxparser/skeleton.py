import fbx
import glm
import numpy as np

from .parser import JointData

FbxEuler = fbx.FbxEuler
FbxNode  = fbx.FbxNode

def to_quat(x, order):
    rx = glm.angleAxis(np.deg2rad(x[0], dtype=np.float32), glm.vec3(1, 0, 0))
    ry = glm.angleAxis(np.deg2rad(x[1], dtype=np.float32), glm.vec3(0, 1, 0))
    rz = glm.angleAxis(np.deg2rad(x[2], dtype=np.float32), glm.vec3(0, 0, 1))

    if order == FbxEuler.eOrderXYZ:
        return rz * ry * rx
    elif order == FbxEuler.eOrderXZY:
        return ry * rz * rx
    elif order == FbxEuler.eOrderYXZ:
        return rz * rx * ry
    elif order == FbxEuler.eOrderYZX:
        return rx * rz * ry
    elif order == FbxEuler.eOrderZXY:
        return ry * rx * rz
    elif order == FbxEuler.eOrderZYX:
        return rx * ry * rz
    else:
        raise ValueError(f"Unknown Euler order: {order}")

def to_vec3(x):
    return glm.vec3(x[0], x[1], x[2])

def parse_nodes_by_type(node, joints, parent_idx, type, scale):
    if node.GetTypeName() == "Null":
        for i in range(node.GetChildCount()):
            parse_nodes_by_type(node.GetChild(i), joints, parent_idx, type, scale)
    
    is_type = False
    for i in range(node.GetNodeAttributeCount()):
        attr = node.GetNodeAttributeByIndex(i)
        if attr.GetAttributeType() == type:
            is_type = True
            break
    
    if is_type:
        name = node.GetName()
        order = node.GetRotationOrder(FbxNode.eDestinationPivot)
        local_T = to_vec3(node.LclTranslation.Get())
        local_R = to_quat(node.LclRotation.Get(), order)
        local_S = to_vec3(node.LclScaling.Get())

        dest_pre_R = node.GetPreRotation(FbxNode.eDestinationPivot)
        pre_quat = to_quat(dest_pre_R, order)

        joint = JointData()
        joint.name       = name
        joint.local_S    = local_S
        joint.local_T    = scale * local_T
        joint.local_R    = local_R
        joint.pre_quat   = pre_quat
        joint.parent_idx = parent_idx
        joints.append(joint)
        parent_idx = len(joints) - 1
    else:
        return
    
    for i in range(node.GetChildCount()):
        parse_nodes_by_type(node.GetChild(i), joints, parent_idx, type, scale)