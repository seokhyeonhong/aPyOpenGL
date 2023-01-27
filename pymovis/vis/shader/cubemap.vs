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
uniform mat4 PV;

void main()
{
    fTexCoord = vPosition;
    vec4 pos = PV * vec4(vPosition, 1.0f);
    gl_Position = pos.xyww;
}