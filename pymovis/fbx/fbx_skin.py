import fbx
import glm
import numpy as np

from pymovis.fbx.fbx_parser import SkinningData

def get_skinning(skinning_data: SkinningData, geometry, control_idx_to_vertex_idx, vertex_num, scale):
    skin_count = geometry.GetDeformerCount(fbx.FbxDeformer.eSkin)

    if skin_count > 1:
        print("Warning: More than one skin deformer found")
        skin_count = 1

    if skin_count == 0:
        return False
    
    skinning_data.joint_indices1 = [glm.ivec4(-1) for _ in range(vertex_num)]
    skinning_data.joint_weights1 = [glm.vec4(0.0) for _ in range(vertex_num)]
    skinning_data.joint_indices2 = [glm.ivec4(-1) for _ in range(vertex_num)]
    skinning_data.joint_weights2 = [glm.vec4(0.0) for _ in range(vertex_num)]

    vertex_skinning_count        = [0 for _ in range(vertex_num)]
    vertex_skinning_weight_sum   = [0.0 for _ in range(vertex_num)]

    for i in range(skin_count):
        cluster_count = geometry.GetDeformer(i, fbx.FbxDeformer.eSkin).GetClusterCount()

        for j in range(cluster_count):
            cluster = geometry.GetDeformer(i, fbx.FbxDeformer.eSkin).GetCluster(j)
            if cluster.GetLinkMode() != fbx.FbxCluster.eNormalize:
                print("Warning: Skinning mode unknown")
                cluster.SetLinkMode(fbx.FbxCluster.eNormalize)
            
            if cluster.GetLink() is not None:
                joint_name = cluster.GetLink().GetName()
                if joint_name not in skinning_data.name_to_idx:
                    skinning_data.joint_order.append(joint_name)
                    joint_idx = len(skinning_data.joint_order) - 1
                    skinning_data.name_to_idx[joint_name] = joint_idx
                else:
                    joint_idx = skinning_data.name_to_idx[joint_name]
            else:
                print("Warning: Link error")
                continue

            idx_count = cluster.GetControlPointIndicesCount()
            indices   = cluster.GetControlPointIndices()
            weights   = cluster.GetControlPointWeights()

            for k in range(idx_count):
                control_point_idx = indices[k]
                vertex_weight     = float(weights[k])

                vertex_indices = control_idx_to_vertex_idx[control_point_idx]
                for vertex_idx in vertex_indices:
                    if vertex_skinning_count[vertex_idx] == 0:
                        skinning_data.joint_indices1[vertex_idx].x = joint_idx
                        skinning_data.joint_weights1[vertex_idx].x = vertex_weight
                    elif vertex_skinning_count[vertex_idx] == 1:
                        skinning_data.joint_indices1[vertex_idx].y = joint_idx
                        skinning_data.joint_weights1[vertex_idx].y = vertex_weight
                    elif vertex_skinning_count[vertex_idx] == 2:
                        skinning_data.joint_indices1[vertex_idx].z = joint_idx
                        skinning_data.joint_weights1[vertex_idx].z = vertex_weight
                    elif vertex_skinning_count[vertex_idx] == 3:
                        skinning_data.joint_indices1[vertex_idx].w = joint_idx
                        skinning_data.joint_weights1[vertex_idx].w = vertex_weight
                    elif vertex_skinning_count[vertex_idx] == 4:
                        skinning_data.joint_indices2[vertex_idx].x = joint_idx
                        skinning_data.joint_weights2[vertex_idx].x = vertex_weight
                    elif vertex_skinning_count[vertex_idx] == 5:
                        skinning_data.joint_indices2[vertex_idx].y = joint_idx
                        skinning_data.joint_weights2[vertex_idx].y = vertex_weight
                    elif vertex_skinning_count[vertex_idx] == 6:
                        skinning_data.joint_indices2[vertex_idx].z = joint_idx
                        skinning_data.joint_weights2[vertex_idx].z = vertex_weight
                    elif vertex_skinning_count[vertex_idx] == 7:
                        skinning_data.joint_indices2[vertex_idx].w = joint_idx
                        skinning_data.joint_weights2[vertex_idx].w = vertex_weight
                    else:
                        print("Warning: Too many skinning weights")
                        continue

                    vertex_skinning_weight_sum[vertex_idx] += vertex_weight
                    vertex_skinning_count[vertex_idx] += 1

            # global initial transform of the geometry node that contains the link node
            matrix = fbx.FbxAMatrix()

            matrix = cluster.GetTransformMatrix(matrix)
            Q = matrix.GetQ()
            T = matrix.GetT()
            S = matrix.GetS()

            q = glm.quat(Q[3], Q[0], Q[1], Q[2])
            t = glm.vec3(T[0], T[1], T[2]) * scale
            s = glm.vec3(S[0], S[1], S[2])

            global_xform = glm.translate(glm.mat4(1.0), t) * glm.mat4_cast(q) * glm.scale(glm.mat4(1.0), s)

            # joint transformations at binding pose
            matrix = cluster.GetTransformLinkMatrix(matrix)
            Q = matrix.GetQ()
            T = matrix.GetT()
            S = matrix.GetS()

            q = glm.quat(Q[3], Q[0], Q[1], Q[2])
            t = glm.vec3(T[0], T[1], T[2]) * scale
            s = glm.vec3(S[0], S[1], S[2])

            xform = glm.translate(glm.mat4(1.0), t) * glm.mat4_cast(q) * glm.scale(glm.mat4(1.0), s)

            skinning_data.offset_xform.append(glm.inverse(xform) * global_xform)

            if cluster.GetAssociateModel() is not None:
                print("Warning: Associate model is not None")
    
    for i in range(vertex_num):
        skinning_data.joint_weights1[i] /= vertex_skinning_weight_sum[i]
        skinning_data.joint_weights2[i] /= vertex_skinning_weight_sum[i]
        
    return True