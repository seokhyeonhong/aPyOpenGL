#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
layout(location=0) in vec3  vPosition;

// --------------------------------------------
// output vertex data
// --------------------------------------------
out vec3     fPosition;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 uProjection;
uniform mat4 uView;

void main()
{
    fPosition = vPosition;
    gl_Position = uProjection * uView * vec4(vPosition, 1.0f);
}