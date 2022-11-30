#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
layout(location=0) in vec4 vPosition; // vec2 position, vec2 texcoord

// --------------------------------------------
// output vertex data
// --------------------------------------------
out vec2 fTexCoord;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

void main()
{
    gl_Position = P * V * M * vec4(vPosition.xy, 0.0, 1.0);
    fTexCoord = vPosition.zw;
}