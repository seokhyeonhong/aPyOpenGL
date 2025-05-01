#version 430
#define MAX_JOINT_NUM 150
uniform mat4 uLbsJoints[MAX_JOINT_NUM];

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
out vec3     fTangent;
out vec3     fBitangent;
out vec3     fNormal;
flat out int fMaterialID;
out vec4     fPosLightSpace;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 uPV;
uniform mat4 uLightSpaceMatrix;

mat4 GetJointMatrix(ivec4 ids, vec4 weights)
{
    mat4 m = mat4(0.0f);
    for (int i = 0; i < 4 && 0 <= ids[i] && ids[i] < MAX_JOINT_NUM; ++i)
    {
        m += uLbsJoints[ids[i]] * weights[i];
    }
    return m;
}

void main()
{
    // LBS
    mat4 lbsModel  = GetJointMatrix(vLbsJointIDs1, vLbsWeights1) + GetJointMatrix(vLbsJointIDs2, vLbsWeights2);

    fPosition      = vec3(lbsModel * vec4(vPosition, 1.0f));
    fTangent       = normalize(mat3(lbsModel) * vTangent);
    fBitangent     = normalize(mat3(lbsModel) * vBitangent);
    fNormal        = normalize(transpose(inverse(mat3(lbsModel))) * vNormal);
    fTexCoord      = vTexCoord;
    fPosLightSpace = uLightSpaceMatrix * vec4(fPosition, 1.0f);
    fMaterialID    = vMaterialID;

    gl_Position    = uPV * vec4(fPosition, 1.0f);
}