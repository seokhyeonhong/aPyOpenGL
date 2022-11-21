#version 430

// input vertex data
in vec3 fPosition;
in vec3 fNormal;
in vec2 fTexCoord;
in vec4 fPosLightSpace;

// output fragment color
out vec4 FragColor;

// uniform
uniform bool uColorMode;
uniform vec3 uColor;
uniform vec2 uvScale;
uniform sampler2D shadowMap;

// material structure
struct Material {
    int       id;
    vec3      albedo;
    vec3      diffuse;
    vec3      specular;
    float     shininess;
    sampler2D albedoMap;
};
uniform Material material;

// light structure
struct Light {
    vec4 vector; // point light if w == 1, directional light if w == 0
    vec3 color;
    vec3 attenuation; // attenuation coefficients
};
uniform Light light;

// camera position
uniform vec3 viewPosition;

float Shadow(vec4 fragPosLightSpace, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // return 0 if outside of light frustum
    if(projCoords.z > 1.0)
    {
        return 0.0;
    }

    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float bias = max(0.0001 * (1.0 - dot(fNormal, lightDir)), 0.00001);

    // if current depth from camera is greater than that of the light source,
    // then the fragment is in shadow
    // float shadow = currentDepth > closestDepth ? 1.0 : 0.0;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int u = -1; u <= 1; ++u)
    {
        for(int v = -1; v <= 1; ++v)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(u, v) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    return shadow;
}

vec4 BlinnPhong(vec3 albedo)
{
    vec3 ambient = light.color * 0.1;

    vec3 N = normalize(fNormal);
    vec3 L = light.vector.w == 1 ? normalize(light.vector.xyz - fPosition) : normalize(-light.vector.xyz);

    vec3 diffuse = max(dot(N, L), 0.0) * material.diffuse * light.color;

    vec3 V = normalize(viewPosition - fPosition);
    // vec3 R = reflect(-L, N); // for phong shading, use R instead of H
    vec3 H = normalize(L + V);
    vec3 specular = pow(max(dot(V, H), 0.0), material.shininess) * material.specular * light.color;
    
    // attenuation
    float atten = 1.0;
    if(light.vector.w == 1)
    {
        float d = length(light.vector.xyz - fPosition.xyz);
        atten = min(1.0 / (light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d), 1.0);
    }

    float shadow = Shadow(fPosLightSpace, L);
    vec3 result = (ambient + atten * (1.0 - shadow) * (diffuse + specular)) * albedo;
    return vec4(result, 1.0);
}

vec3 GammaCorrection(vec3 color, float gamma)
{
    return pow(color, vec3(1.0 / gamma));
}

void main()
{
    vec2 uv = fTexCoord * uvScale;
    if (uColorMode)
    {
        FragColor = vec4(uColor, 1.0);
    }
    else if (material.id >= 0)
    {
        FragColor = BlinnPhong(texture(material.albedoMap, uv).rgb);
    }
    else
    {
        FragColor = BlinnPhong(material.albedo);
    }
    // vec3 fogColor = vec3(0.5);
    // float d = length(fPosition - viewPosition);
    // float fogFactor = clamp((d - 10.0) / 10.0, 0.0, 1.0);
    // fogColor = fogColor * fogFactor;
    FragColor.rgb = GammaCorrection(FragColor.rgb, 0.5);// + fogColor;
}