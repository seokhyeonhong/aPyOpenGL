#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
layout(location=0) in vec3  vPosition;
layout(location=1) in vec3  vNormal;
layout(location=2) in vec2  vTexCoord;
layout(location=3) in int   vMaterialID;
layout(location=4) in ivec4 vLbsJointIDs1;
layout(location=5) in vec4  vLbsWeights1;
layout(location=6) in ivec4 vLbsJointIDs2;
layout(location=7) in vec4  vLbsWeights2;

// --------------------------------------------
// output vertex data
// --------------------------------------------
out vec3     fPosition;
out vec3     fNormal;
out vec2     fTexCoord;
flat out int fMaterialID;
out vec4     fPosLightSpace;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 uPV;
uniform mat4 M;
uniform mat4 uLightSpaceMatrix;

void main()
{
    fPosition      = vec3(M * vec4(vPosition, 1.0f));
    fNormal        = normalize(transpose(inverse(mat3(M))) * vNormal);
    fTexCoord      = vTexCoord;
    fPosLightSpace = uLightSpaceMatrix * vec4(fPosition, 1.0f);
    fMaterialID    = vMaterialID;
    gl_Position    = uPV * vec4(fPosition, 1.0f);
}