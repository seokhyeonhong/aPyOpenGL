#version 430

in vec4 fPosition;
in vec4 fColor;
in vec4 fNormal;
in vec2 fTexCoord;
in vec4 fPosLightSpace;

out vec4 FragColor;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

uniform sampler2D shadowMap;
uniform int colorMode;
uniform int textureMode;
uniform vec4 uColor;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};
uniform Material material;

struct DiffuseTextureMaterial {
    sampler2D diffuse;
    vec3      specular;
    float     shininess;
};
uniform DiffuseTextureMaterial diffuseTextureMaterial;

struct Light {
    vec4 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};
uniform Light light;

float calculateShadow(vec4 fragPosLightSpace)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float shadow = currentDepth > closestDepth ? 1.0 : 0.0;
    return shadow;
}

vec4 phong(vec3 mAmbient, vec3 mDiffuse, vec3 mSpecular, float shininess)
{
    // ambient
    vec3 ambient = light.ambient * mAmbient;

    // diffuse
    vec3 lightDir = normalize(light.position.xyz - fPosition.xyz);
    vec3 diffuse = max(dot(lightDir, fNormal.xyz), 0.0) * light.diffuse * mDiffuse;

    // specular
    vec3 viewDir = normalize(-fPosition.xyz);
    vec3 reflectDir = reflect(-lightDir, fNormal.xyz);
    vec3 specular = pow(max(dot(viewDir, reflectDir), 0.0), shininess) * light.specular * mSpecular;
    
    // attenuation
    float d = length(light.position.xyz - fPosition.xyz);
    float atten = min(1.0 / (light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d), 1.0);

    float shadow = calculateShadow(fPosLightSpace);
    vec3 result = ambient + (1.0 - shadow) * (diffuse + specular);
    return vec4(result, 1.0);
}

void main()
{
    vec3 mAmbient, mDiffuse, mSpecular;
    float shininess;
    switch(textureMode)
    {
        // no texture
        case 0:
            mAmbient = material.ambient;
            mDiffuse = material.diffuse;
            mSpecular = material.specular;
            shininess = material.shininess;
            break;
        // diffuse texture
        case 1:
            vec3 texColor = texture(diffuseTextureMaterial.diffuse, fTexCoord).rgb;
            mAmbient = texColor;
            mDiffuse = texColor;
            mSpecular = diffuseTextureMaterial.specular;
            shininess = diffuseTextureMaterial.shininess;
            break;
    }
    switch(colorMode)
    {
        // phong shading
        case 0:
            FragColor = phong(mAmbient, mDiffuse, mSpecular, shininess);
            break;
        // vertex color
        case 1:
            FragColor = fColor;
            break;
        // user-defined color
        case 2:
            FragColor = uColor;
            break;
    }
}