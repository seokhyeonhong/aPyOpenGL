#version 430

layout(location=0) in vec4 vPosition;

uniform mat4 lightSpaceMatrix;
uniform mat4 M;

void main()
{
    gl_Position = lightSpaceMatrix * M * vPosition;
}