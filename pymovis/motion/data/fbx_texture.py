from __future__ import annotations

import fbx
from motion.data.fbx_parser import TextureInfo

def get_textures(geometry) -> list[TextureInfo]:
    if geometry.GetNode() is None:
        return []
    
    num_materials = geometry.GetNode().GetSrcObjectCount()
    for material_idx in range(num_materials):
        material = geometry.GetNode().GetSrcObject(material_idx)
        
        # go through all the possible textures
        if not isinstance(material, fbx.FbxSurfaceMaterial):
            continue
        
        for texture_idx in range(fbx.FbxLayerElement.sTypeTextureCount()):
            property = material.FindProperty(fbx.FbxLayerElement.sTextureChannelNames(texture_idx))
            textures = find_and_display_texture_info_by_property(property, material_idx)

    return textures

def find_and_display_texture_info_by_property(property, material_idx):
    textures = []

    if not property.IsValid():
        print("Invalid property")
        return textures
    
    texture_count = property.GetSrcObjectCount()
    for j in range(texture_count):
        if not isinstance(property.GetSrcObject(j), fbx.FbxTexture):
            continue

        layered_texture = property.GetSrcObject(j)
        if isinstance(layered_texture, fbx.FbxLayeredTexture):
            for k in range(layered_texture.GetSrcObjectCount()):
                texture = layered_texture.GetSrcObject(k)
                if not isinstance(texture, fbx.FbxTexture):
                    continue
                
                blend_mode = layered_texture.GetTextureBlendMode(k)
                texture_info = get_file_texture(texture, blend_mode)
                texture_info.property = property.GetName()
                texture_info.connected_material = material_idx
                textures.append(texture_info)
        else:
            texture = property.GetSrcObject(j)
            if not isinstance(texture, fbx.FbxTexture):
                continue
            
            texture_info = get_file_texture(texture, -1)
            texture_info.property = property.GetName()
            texture_info.connected_material = material_idx
            textures.append(texture_info)
    
    return textures

def get_file_texture(texture, blend_mode):
    # TODO: Implement this
    info = TextureInfo()