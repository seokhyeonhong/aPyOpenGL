#version 430

/**
 * input vertex data
 */
layout(location=0) in vec3 vPosition;
layout(location=1) in vec3 vNormal;
layout(location=2) in vec2 vTexCoord;

/**
 * output vertex data
 */
out vec3 fPosition;
out vec3 fNormal;
out vec2 fTexCoord;
out vec4 fPosLightSpace;

/**
 * uniform data
 */
uniform mat4 P;
uniform mat4 V;
uniform mat4 M;
uniform mat4 lightSpaceMatrix;

void main()
{
    fPosition      = vec3(M * vec4(vPosition, 1.0));
    fNormal        = transpose(inverse(mat3(M))) * vNormal;
    fTexCoord      = vTexCoord;
    fPosLightSpace = lightSpaceMatrix * vec4(fPosition, 1.0);
    gl_Position    = P * V * vec4(fPosition, 1.0);
}