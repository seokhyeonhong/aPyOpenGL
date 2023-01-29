#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
in vec3     fPosition;
in vec3     fNormal;
in vec2     fTexCoord;
flat in int fMaterialID;
in vec4     fPosLightSpace;

// --------------------------------------------
// output fragment color
// --------------------------------------------
out vec4 FragColor;

// --------------------------------------------
// uniform
// --------------------------------------------
uniform bool      uColorMode;
uniform vec2      uvScale;
uniform sampler2D uShadowMap;
uniform bool      uIsFloor;
uniform vec2      uGridSize;
uniform vec3      uGridColors[2];

// --------------------------------------------
// material structure
// --------------------------------------------
#define MAX_MATERIAL_NUM 5
struct Material {
    ivec4 textureID; // albedo, normal, metalic, emissive
    vec4  albedo;
    vec3  diffuse;
    vec3  specular;
    float shininess;
};
uniform Material uMaterial[MAX_MATERIAL_NUM];

// --------------------------------------------
// textures
// --------------------------------------------
#define MAX_MATERIAL_TEXTURE 25
uniform sampler2D uTextures[MAX_MATERIAL_TEXTURE];

// --------------------------------------------
// light structure
// --------------------------------------------
struct Light {
    vec4 vector; // point light if w == 1, directional light if w == 0
    vec3 color;
    vec3 attenuation; // attenuation coefficients
};
uniform Light uLight;

// --------------------------------------------
// camera position
// --------------------------------------------
uniform vec3 viewPosition;

// --------------------------------------------
float Shadow(vec4 fragPosLightSpace, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5f + 0.5f;

    // return 0 if outside of light frustum
    if(projCoords.z > 1.0f)
    {
        return 0.0f;
    }

    float closestDepth = texture(uShadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float bias = max(0.0001f * (1.0f - dot(fNormal, lightDir)), 0.00001f);

    // if current depth from camera is greater than that of the light source,
    // then the fragment is in shadow
    // float shadow = currentDepth > closestDepth ? 1.0 : 0.0;
    float shadow = 0.0f;
    vec2 texelSize = 1.0f / textureSize(uShadowMap, 0);
    for(int u = -1; u <= 1; ++u)
    {
        for(int v = -1; v <= 1; ++v)
        {
            float pcfDepth = texture(uShadowMap, projCoords.xy + vec2(u, v) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0f : 0.0f;
        }
    }
    shadow /= 9.0f;
    return shadow;
}

// --------------------------------------------
vec4 BlinnPhong(vec3 albedo, vec3 N, vec3 V)
{
    // vec3 ambient = albedo;
    vec3 ambient = uLight.color * 0.1f;

    vec3 L = uLight.vector.w == 1.0f ? normalize(uLight.vector.xyz - fPosition) : normalize(-uLight.vector.xyz);

    vec3 diffuse = max(dot(N, L), 0.0f) * uMaterial[fMaterialID].diffuse * uLight.color;

    // vec3 R = reflect(-L, N); // for phong shading, use R instead of H
    vec3 H = normalize(L + V); // for blinn-phong shading, use H instead of R
    vec3 specular = pow(max(dot(V, H), 0.0f), uMaterial[fMaterialID].shininess) * uMaterial[fMaterialID].specular * uLight.color;
    
    // attenuation
    float atten = 1.0f;
    if(uLight.vector.w == 1.0f)
    {
        float d = length(uLight.vector.xyz - fPosition.xyz);
        atten = min(1.0f / (uLight.attenuation.x + uLight.attenuation.y * d + uLight.attenuation.z * d * d), 1.0f);
    }

    float shadow = Shadow(fPosLightSpace, L);
    vec3 result = (ambient + atten * (1.0f - shadow) * (diffuse + specular)) * albedo;
    return vec4(result, 1.0f);
}

// --------------------------------------------
vec3 GammaCorrection(vec3 color, float gamma)
{
    return pow(color, vec3(1.0f / gamma));
}

// --------------------------------------------
// Reference: https://iquilezles.org/articles/checkerfiltering/
vec3 Grid(vec2 p)
{
    vec2 q = sign(fract(p / uGridSize * 0.5) - 0.5f);
    float t = 0.5f * (1.0f - q.x * q.y);
    return t * uGridColors[1] + (1.0f - t) * uGridColors[0];
}

vec2 Triangular(vec2 p)
{
    vec2 q = fract(p * 0.5f) - 0.5f;
    return 1.0f - 2.0 * abs(q);
}

vec3 FilterGrid(vec2 p)
{
    vec2 q = p / uGridSize;
    vec2 w = max(abs(dFdx(q)), abs(dFdy(q))) + 0.001f;
    vec2 i = (Triangular(q + 0.5f * w) - Triangular(q - 0.5f * w)) / w;
    float t = 0.5f * (1.0f - i.x * i.y);
    return t * uGridColors[1] + (1.0f - t) * uGridColors[0];
}

// --------------------------------------------
// main function
// --------------------------------------------
void main()
{
    // find material texture ID
    int albedoID = uMaterial[fMaterialID].textureID.x;

    // texture scaling
    vec2 uv = fTexCoord * uvScale;

    // find material attributes
    vec3 materialColor = uMaterial[fMaterialID].albedo.rgb;
    float alpha = uMaterial[fMaterialID].albedo.a;

    // set normal
    vec3 N = normalize(fNormal);
    vec3 V = normalize(viewPosition - fPosition);

    // materials
    vec3 albedo = materialColor;

    // --------------------------------------------
    // albedo texture
    if (uMaterial[fMaterialID].textureID.x >= 0)
    {
        albedo = texture(uTextures[albedoID], uv).rgb;
    }

    // --------------------------------------------
    // rendering
    if (uIsFloor)
    {
        FragColor.rgb = pow(FilterGrid(fPosition.xz), vec3(0.8f));
        FragColor.a = 1.0f;
        return;
    }
    else if (uColorMode)
    {
        FragColor = vec4(albedo, 1.0f);
        return;
    }
    else
    {
        FragColor = BlinnPhong(albedo, N, V);
        FragColor.a = alpha;
    }

    // Fog
    // float D = length(viewPosition - fPosition);
    // vec3 fog_color = vec3(0.5);
    // float fog_amount = 1.0f - min(exp(-D * 0.1 + 1.5), 1.0);
    // vec3 color = FragColor.rgb;
    // color = mix(color, fog_color, fog_amount);
    // FragColor.rgb = color;
    // vec3 fogColor = vec3(0.5);
    // float d = length(fPosition - viewPosition);
    // float fogFactor = clamp((d - 10.0) / 10.0, 0.0, 1.0);
    // fogColor = fogColor * fogFactor;
    // FragColor.rgb = GammaCorrection(FragColor.rgb, 1.0 / 2.2);// + fogColor;
}