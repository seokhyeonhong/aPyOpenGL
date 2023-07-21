#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
layout(location=0) in vec3  vPosition;
layout(location=1) in vec3  vNormal;
layout(location=2) in vec2  vTexCoord;
layout(location=3) in vec3  vTangent;
layout(location=4) in vec3  vBitangent;
layout(location=5) in int   vMaterialID;
layout(location=6) in ivec4 vLbsJointIDs1;
layout(location=7) in vec4  vLbsWeights1;
layout(location=8) in ivec4 vLbsJointIDs2;
layout(location=9) in vec4  vLbsWeights2;

// --------------------------------------------
// output vertex data
// --------------------------------------------
out vec3     fPosition;
out vec2     fTexCoord;
out vec3     fNormal;
out mat3     fTBN;
flat out int fMaterialID;
out vec4     fPosLightSpace;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 uPV;
uniform mat4 uModel;
uniform mat4 uLightSpaceMatrix;

#define MAX_INSTANCE_NUM 100
uniform int  uInstanceNum;
uniform mat4 uInstanceModel[MAX_INSTANCE_NUM];

void main()
{
    mat4 M = mat4(1.0f);
    if (uInstanceNum == 1)
    {
        M = uModel;
    }
    else
    {
        M = uInstanceModel[gl_InstanceID];
    }
    fPosition      = vec3(M * vec4(vPosition, 1.0f));
    vec3 T         = normalize(transpose(inverse(mat3(M))) * vTangent);
    vec3 B         = normalize(transpose(inverse(mat3(M))) * vBitangent);
    vec3 N         = normalize(transpose(inverse(mat3(M))) * vNormal);
    fTBN           = mat3(T, B, N);
    fNormal        = N;
    fTexCoord      = vTexCoord;
    fPosLightSpace = uLightSpaceMatrix * vec4(fPosition, 1.0f);
    fMaterialID    = vMaterialID;
    gl_Position    = uPV * vec4(fPosition, 1.0f);
}