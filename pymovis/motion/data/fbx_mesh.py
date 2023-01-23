from __future__ import annotations

import fbx
import numpy as np

from pymovis.motion.data.fbx_parser import MeshData

def get_mesh_data(fbx_mesh_, scale) -> MeshData:
    if fbx_mesh_.GetElementNormalCount() == 0:
        if not fbx_mesh_.GenerateNormals():
            raise Exception("FBX normal import error")
    
    if fbx_mesh_.GetElementTangentCount() == 0:
        if not fbx_mesh_.GenerateTangentsDataForAllUVSets():
            raise Exception("FBX tangent import error")
    
    polygon_count = fbx_mesh_.GetPolygonCount()
    control_points = fbx_mesh_.GetControlPoints()
    
    data = MeshData()
    vertex_id = 0
    for i in range(polygon_count):
        if fbx_mesh_.GetPolygonSize(i) != 3:
            raise Exception("Only triangles are supported")

        for j in range(fbx_mesh_.GetPolygonSize(i)):
            control_point_idx = fbx_mesh_.GetPolygonVertex(i, j)

            data.indices.append(vertex_id)
            if control_point_idx not in data.control_point_idx_to_vertex_idx:
                data.control_point_idx_to_vertex_idx[control_point_idx] = [vertex_id]
            else:
                data.control_point_idx_to_vertex_idx[control_point_idx].append(vertex_id)

            position = get_mesh_position(fbx_mesh_, control_points, i, j, scale)
            data.positions.append(position)

            normal = get_mesh_normal(fbx_mesh_, vertex_id, i, j)
            data.normals.append(normal)

            uv = get_mesh_uv(fbx_mesh_, i, j)
            data.uvs.append(uv)

            vertex_id += 1
    
    if vertex_id != 3 * polygon_count:
        raise Exception("Vertex count is not correct")
        
    return data

def get_mesh_position(fbx_mesh_, control_points, i, j, scale=0.01):
    control_point_idx = fbx_mesh_.GetPolygonVertex(i, j)
    position = control_points[control_point_idx]
    position = scale * np.array([position[0], position[1], position[2]], dtype=np.float32)
    return position

def get_mesh_normal(fbx_mesh_, vertex_id, i, j):
    for l in range(fbx_mesh_.GetElementNormalCount()):
        le_normal = fbx_mesh_.GetElementNormal(l)

        if le_normal.GetMappingMode() == fbx.FbxLayerElement.eByPolygonVertex:
            if le_normal.GetReferenceMode() == fbx.FbxLayerElement.eDirect:
                normal = le_normal.GetDirectArray().GetAt(vertex_id)
                normal = np.array([normal[0], normal[1], normal[2]], dtype=np.float32)
            elif le_normal.GetReferenceMode() == fbx.FbxLayerElement.eIndexToDirect:
                idx = le_normal.GetIndexArray().GetAt(vertex_id)
                normal = le_normal.GetDirectArray().GetAt(idx)
                normal = np.array([normal[0], normal[1], normal[2]], dtype=np.float32)
            else:
                raise Exception("Unknown reference mode")
        elif le_normal.GetMappingMode() == fbx.FbxLayerElement.eByControlPoint:
            control_point_idx = fbx_mesh_.GetPolygonVertex(i, j)
            normal_idx = 0
            if le_normal.GetReferenceMode() == fbx.FbxLayerElement.eDirect:
                normal_idx = control_point_idx
            elif le_normal.GetReferenceMode() == fbx.FbxLayerElement.eIndexToDirect:
                normal_idx = le_normal.GetIndexArray().GetAt(control_point_idx)
            else:
                raise Exception("Unknown reference mode")
            
            normal = le_normal.GetDirectArray().GetAt(normal_idx)
            normal = np.array([normal[0], normal[1], normal[2]], dtype=np.float32)
    
    return normal / (np.linalg.norm(normal) + 1e-8)

def get_mesh_uv(fbx_mesh_, i, j):
    control_point_idx = fbx_mesh_.GetPolygonVertex(i, j)

    for i in range(fbx_mesh_.GetElementUVCount()):
        le_uv = fbx_mesh_.GetElementUV(i)

        if le_uv.GetMappingMode() == fbx.FbxLayerElement.eByControlPoint:
            if le_uv.GetReferenceMode() == fbx.FbxLayerElement.eDirect:
                uv = le_uv.GetDirectArray().GetAt(control_point_idx)
                uv = np.array([uv[0], uv[1]], dtype=np.float32)
            elif le_uv.GetReferenceMode() == fbx.FbxLayerElement.eIndexToDirect:
                idx = le_uv.GetIndexArray().GetAt(control_point_idx)
                uv = le_uv.GetDirectArray().GetAt(idx)
                uv = np.array([uv[0], uv[1]], dtype=np.float32)
            else:
                raise Exception("Unknown reference mode")
        elif le_uv.GetMappingMode() == fbx.FbxLayerElement.eByPolygonVertex:
            texture_uv_idx = fbx_mesh_.GetTextureUVIndex(i, j)
            if le_uv.GetReferenceMode() == fbx.FbxLayerElement.eDirect or \
                le_uv.GetReferenceMode() == fbx.FbxLayerElement.eIndexToDirect:
                uv = le_uv.GetDirectArray().GetAt(texture_uv_idx)
                uv = np.array([uv[0], uv[1]], dtype=np.float32)
            else:
                raise Exception("Unknown reference mode")
        elif le_uv.GetMappingMode() in [fbx.FbxLayerElement.eByPolygon, fbx.FbxLayerElement.eAllSame, fbx.FbxLayerelement.eNone]:
            raise Exception("Not implemented mapping mode")
        else:
            raise Exception("Unknown mapping mode")
    
    return uv