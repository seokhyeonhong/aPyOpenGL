#version 430
#define MAX_JOINT_NUM 100
uniform mat4 uLbsJoints[MAX_JOINT_NUM];

// --------------------------------------------
// input vertex data
// --------------------------------------------
layout(location=0) in vec3  vPosition;
layout(location=4) in ivec4 vLbsJointIDs1;
layout(location=5) in vec4  vLbsWeights1;
layout(location=6) in ivec4 vLbsJointIDs2;
layout(location=7) in vec4  vLbsWeights2;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 uLightSpaceMatrix;
uniform mat4 uModel;
uniform bool uIsSkinned;

#define MAX_INSTANCE_NUM 100
uniform int  uInstanceNum;
uniform mat4 uInstanceModel[MAX_INSTANCE_NUM];

mat4 GetJointMatrix(ivec4 ids, vec4 weights)
{
    mat4 m = mat4(0.0f);
    for (int i = 0; i < 4; ++i)
    {
        if (0 <= ids[i] && ids[i] < MAX_JOINT_NUM)
        {
            m += uLbsJoints[ids[i]] * weights[i];
        }
        else
        {
            break;
        }
    }
    return m;
}

void main()
{
    if (uIsSkinned)
    {
        // LBS
        mat4 lbsModel = GetJointMatrix(vLbsJointIDs1, vLbsWeights1) + GetJointMatrix(vLbsJointIDs2, vLbsWeights2);
        gl_Position = uLightSpaceMatrix * lbsModel * vec4(vPosition, 1.0f);
    }
    else if (uInstanceNum == 1)
    {
        gl_Position = uLightSpaceMatrix * uModel * vec4(vPosition, 1.0f);
    }
    else
    {
        gl_Position = uLightSpaceMatrix * uInstanceModel[gl_InstanceID] * vec4(vPosition, 1.0f);
    }
}