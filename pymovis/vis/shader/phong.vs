#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
layout(location=0) in vec3 vPosition;
layout(location=1) in vec3 vNormal;
layout(location=2) in vec2 vTexCoord;
layout(location=3) in int  vMaterialID;

// --------------------------------------------
// output vertex data
// --------------------------------------------
out vec3 fPosition;
out vec3 fNormal;
out vec2 fTexCoord;
flat out int  fMaterialID;
out vec4 fPosLightSpace;

// --------------------------------------------
// uniform data
// --------------------------------------------
uniform mat4 P;
uniform mat4 V;
uniform mat4 M;
uniform mat4 lightSpaceMatrix;

void main()
{
    fPosition      = vec3(M * vec4(vPosition, 1.0));
    fNormal        = normalize(transpose(inverse(mat3(M))) * vNormal);
    fTexCoord      = vTexCoord;
    fPosLightSpace = lightSpaceMatrix * vec4(fPosition, 1.0);
    fMaterialID    = vMaterialID;
    gl_Position    = P * V * vec4(fPosition, 1.0);
}