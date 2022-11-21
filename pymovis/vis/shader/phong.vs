#version 430

layout(location=0) in vec4 vPosition;
layout(location=1) in vec4 vColor;
layout(location=2) in vec4 vNormal;
layout(location=3) in vec2 vTexCoord;

out vec4 fPosition;
out vec4 fColor;
out vec4 fNormal;
out vec2 fTexCoord;
out vec4 fPosLightSpace;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform mat4 lightSpaceMatrix;

void main()
{
    fPosition      = M * vPosition;
    fColor         = vColor;
    fNormal        = transpose(inverse(M)) * vNormal;
    fTexCoord      = vTexCoord;
    fPosLightSpace = lightSpaceMatrix * fPosition;
    gl_Position    = P * V * fPosition;
}