#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
layout(location=0) in vec3 vPosition;

// --------------------------------------------
// output vertex data
// --------------------------------------------
out vec3 fTexCoord;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 P;
uniform mat4 V;

void main()
{
    fTexCoord = vPosition;
    vec4 pos = P * V * vec4(vPosition, 1.0);
    gl_Position = pos.xyww;
}