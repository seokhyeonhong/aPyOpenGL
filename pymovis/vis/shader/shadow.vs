#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
layout(location=0) in vec3 vPosition;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 lightSpaceMatrix;
uniform mat4 M;

void main()
{
    gl_Position = lightSpaceMatrix * M * vec4(vPosition, 1.0);
}