from __future__ import annotations

import fbx
import glm

from pymovis.motion.data.fbx_parser import MaterialInfo

def get_materials(geometry) -> list[MaterialInfo]:
    materials = []

    if geometry:
        node = geometry.GetNode()
        if node:
            material_count = node.GetMaterialCount()
    
    if material_count > 0:
        color = fbx.FbxColor()
        for count in range(material_count):
            info = MaterialInfo()
            info.material_id = count

            material = node.GetMaterial(count)
            info.name = material.GetName()

            implementation = fbx.GetImplementation(material, "ImplementationHLSL")
            implementation_type = "HLSL"
            if not implementation:
                implementation = fbx.GetImplementation(material, "ImplementationCGFX")
                implementation_type = "CGFX"

            if not implementation:
                if material.GetClassId().Is(fbx.FbxSurfacePhong.ClassId):
                    info.type = "phong"

                    # ambient
                    val = material.Ambient.Get()
                    color.Set(val[0], val[1], val[2])
                    info.ambient = glm.vec3(color.mRed, color.mGreen, color.mBlue)

                    # diffuse
                    val = material.Diffuse.Get()
                    color.Set(val[0], val[1], val[2])
                    info.diffuse = glm.vec3(color.mRed, color.mGreen, color.mBlue)

                    # specular
                    val = material.Specular.Get()
                    color.Set(val[0], val[1], val[2])
                    info.specular = glm.vec3(color.mRed, color.mGreen, color.mBlue)

                    # emissive
                    val = material.Emissive.Get()
                    color.Set(val[0], val[1], val[2])
                    info.emissive = glm.vec3(color.mRed, color.mGreen, color.mBlue)

                    # opacity
                    val = material.TransparencyFactor.Get()
                    info.opacity = 1.0 - val

                    # shininess
                    val = material.Shininess.Get()
                    info.shininess = val

                    # reflectivity
                    val = material.ReflectionFactor.Get()
                    info.reflectivity = val

                    materials.append(info)
                elif material.GetClassId().Is(fbx.FbxSurfaceLambert.ClassId):
                    info.type = "lambert"

                    # ambient
                    val = material.Ambient.Get()
                    color.Set(val[0], val[1], val[2])
                    info.ambient = glm.vec3(color.mRed, color.mGreen, color.mBlue)

                    # diffuse
                    val = material.Diffuse.Get()
                    color.Set(val[0], val[1], val[2])
                    info.diffuse = glm.vec3(color.mRed, color.mGreen, color.mBlue)

                    # emissive
                    val = material.Emissive.Get()
                    color.Set(val[0], val[1], val[2])
                    info.emissive = glm.vec3(color.mRed, color.mGreen, color.mBlue)

                    # opacity
                    val = material.TransparencyFactor.Get()
                    info.opacity = 1.0 - val

                    materials.append(info)
                else:
                    print("Unknown material type: ", material.GetClassId().GetName())

    return materials

def get_polygon_material_connection(mesh):
    material_connection = []

    polygon_count = mesh.GetPolygonCount()
    is_all_same = True
    for l in range(mesh.GetElementMaterialCount()):
        material_element = mesh.GetElementMaterial(l)
        if material_element.GetMappingMode() == fbx.FbxLayerElement.eByPolygon:
            is_all_same = False
            break
    
    if is_all_same:
        if mesh.GetElementMaterialCount() == 0:
            material_connection = [-1] * polygon_count
        else:
            material_element = mesh.GetElementMaterial(0)
            if material_element.GetMappingMode() == fbx.FbxLayerElement.eAllSame:
                material_id = material_element.GetIndexArray().GetAt(0)
                material_connection = [material_id] * polygon_count
    else:
        material_connection = [0] * polygon_count
        for i in range(polygon_count):
            material_num = mesh.GetElementMaterialCount()
            if material_num >= 1:
                material_element = mesh.GetElementMaterial(0)
                material_connection[i] = material_element.GetIndexArray().GetAt(i)

    return material_connection