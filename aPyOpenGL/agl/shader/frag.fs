#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
in vec3     fPosition;
in vec2     fTexCoord;
in mat3     fTBN;
in vec3     fNormal;
flat in int fMaterialID;
in vec4     fPosLightSpace;

// --------------------------------------------
// output fragment color
// --------------------------------------------
out vec4 FragColor;

// --------------------------------------------
// uniform
// --------------------------------------------
uniform bool        uColorMode;
uniform bool        uPBR; // true: pbr shading, false: phong shading
uniform vec2        uvScale;
uniform float       uDispScale;
uniform samplerCube uIrradianceMap; // IBL
uniform float       uIrradianceMapIntensity;
uniform sampler2D   uShadowMap;

uniform bool        uIsFloor;
uniform vec3        uGridColor;
uniform float       uGridWidth;
uniform float       uGridInterval;
uniform bool        uDebug;

uniform vec3        uSkyColor;

// --------------------------------------------
// material structure
// --------------------------------------------
#define MAX_MATERIAL_NUM 5
struct Material
{
    ivec3 textureID;    // albedo, normal, displacement
    ivec3 pbrTextureID; // metallic, roughness, ao
    vec4  albedo;       // RGBA

    // phong shading
    vec3  diffuse;
    vec3  specular;
    float shininess;

    // pbr
    float metallic;
    float roughness;
    float ao;
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
#define MAX_LIGHT_NUM 4
struct Light
{
    vec4 vector; // point light if w == 1, directional light if w == 0
    vec3 color;
    vec3 attenuation; // attenuation coefficients
};
uniform Light uLight[MAX_LIGHT_NUM];
uniform int   uLightNum;

// --------------------------------------------
// camera position
// --------------------------------------------
uniform vec3 uViewPosition;

// --------------------------------------------
// constants
// --------------------------------------------
const float PI = 3.14159265359f;
const float GAMMA = 2.2f;

// --------------------------------------------
float Shadow(vec4 fragPosLightSpace, vec3 lightDir, sampler2D shadowMap)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5f + 0.5f;

    // return 0 if outside of light frustum
    if(projCoords.z > 1.0f)
    {
        return 0.0f;
    }

    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float bias = max(0.001f * (1.0f - dot(fNormal, lightDir)), 0.0001f);

    // if current depth from camera is greater than that of the light source,
    // then the fragment is in shadow
    // float shadow = currentDepth > closestDepth ? 1.0 : 0.0;
    float shadow = 0.0f;
    vec2 texelSize = 1.0f / textureSize(shadowMap, 0);
    int kernelSize = 2;
    for(int u = -kernelSize; u <= kernelSize; ++u)
    {
        for(int v = -kernelSize; v <= kernelSize; ++v)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(u, v) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0f : 0.0f;
        }
    }
    shadow /= float((kernelSize * 2 + 1) * (kernelSize * 2 + 1));
    return shadow;
}

// --------------------------------------------
float GetAttenuation(Light light)
{
    float atten = 1.0f;
    if (light.vector.w == 1.0f)
    {
        float d = length(light.vector.xyz - fPosition.xyz);
        atten = min(1.0f / (light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d), 1.0f);
    }

    return atten;
}

vec3 BlinnPhong(vec3 albedo, vec3 N, vec3 V, vec3 L, Light light, Material material)
{
    // diffuse
    vec3 diffuse = max(dot(N, L), 0.0f) * material.diffuse * light.color;

    // specular
    // vec3 R = reflect(-L, N); // for phong shading, use R instead of H
    vec3 H = normalize(L + V); // for blinn-phong shading, use H instead of R
    vec3 specular = pow(max(dot(V, H), 0.0f), material.shininess) * material.specular * light.color;
    
    // attenuation
    float atten = GetAttenuation(light);

    // final color
    vec3 result = (atten * (diffuse + specular)) * albedo;
    return result;
}

// --------------------------------------------
vec3 ReinhardToneMapping(vec3 color)
{
    vec3 result = color / (color + vec3(1.0f));
    return pow(result, vec3(1.0f / GAMMA));
}

vec3 ACESFilmicToneMapping(vec3 x)
{
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    x = (x * (a * x + b)) / (x * (c * x + d) + e);
    return clamp(x, 0.0f, 1.0f);
}

// --------------------------------------------
vec2 Filter(vec2 p, float q)
{
    return floor(p) + min(fract(p) * q, 1.0f);
}

float FilteredGrid(vec2 p)
{
    p *= uGridInterval;

    // larger the _N, thinner the line
    float _N = 200.0f / uGridWidth;
    vec2 w = max(abs(dFdx(p)), abs(dFdy(p))) + 0.001f;
    w *= uGridInterval;

    vec2 a = p + 0.5f * w;
    vec2 b = p - 0.5f * w;
    vec2 i = (Filter(a, _N) - Filter(b, _N)) / (_N*w);

    return (1.0f - i.x) * (1.0f - i.y);
}

vec2 Triangular(vec2 p)
{
    vec2 q = fract(p * 0.5f) - 0.5f;
    return 1.0f - 2.0f * abs(q);
}

float FilteredChecker(vec2 p)
{
    p *= uGridInterval;

    vec2 w = max(abs(dFdx(p)), abs(dFdy(p))) + 0.001f;
    w *= uGridInterval;

    vec2 a = p + 0.5f * w;
    vec2 b = p - 0.5f * w;
    vec2 i = (Triangular(p + 0.5f * w) - Triangular(p - 0.5f * w)) / w;

    float res = 1.0f - i.x * i.y;
    return res * 0.5f + 0.5f;
}

// --------------------------------------------
vec3 GetNormalFromMap(sampler2D normalMap, vec2 uv)
{
    vec3 N = texture(normalMap, uv).rgb * 2.0f - 1.0f;
    // vec3 N = vec3(0, 0, 1);
    return normalize(fTBN * N);
}

// --------------------------------------------
vec2 ParallaxOcclusionMapping(sampler2D dispMap, vec2 texCoords, vec3 viewDir)
{
    // number of the depth layers
    const float minLayers = 8.0f;
    const float maxLayers = 32.0f;
    const float numLayers = mix(maxLayers, minLayers, max(viewDir.z, 0.0f));

    // calculate the size of each layer
    float layerDepth = 1.0f / numLayers;

    // depth of the current layer
    float currLayerDepth = 0.0f;

    // the amount to shift the texture coordinates per layer from vector P
    vec2 P = viewDir.xy * uDispScale;
    vec2 deltaTexCoords = P * numLayers;
    
    // get initial values
    vec2 currTexCoords = texCoords;
    float currDepthMapValue = texture(dispMap, currTexCoords).r;

    while (currLayerDepth < currDepthMapValue)
    {
        // shift texture coordinates along direction of P
        currTexCoords -= deltaTexCoords;

        // get depthmap value at current texture coordinates
        currDepthMapValue = texture(dispMap, currTexCoords).r;

        // get the depth of the next layer
        currLayerDepth += layerDepth;
    }

    // get texture coordinates before collision
    vec2 prevTexCoords = currTexCoords + deltaTexCoords;

    // get depth after and before collision
    float afterDepth = currDepthMapValue - currLayerDepth;
    float beforeDepth = texture(dispMap, prevTexCoords).r - currLayerDepth + layerDepth;

    // linear interpolation of texture coordinates
    float t = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = mix(currTexCoords, prevTexCoords, t);

    return finalTexCoords;
}

// --------------------------------------------
// PBR functions
// normal distribution function
float TrowbridgeReitzGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0f);

    float num = a2;
    float denom = (NdotH*NdotH) * (a2 - 1.0f) + 1.0f;
    denom = PI * denom * denom;

    return num / denom;
}

// geometry function
float SchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r*r) / 8.0f;

    float num = NdotV;
    float denom = NdotV * (1.0f - k) + k;
    
    return num / denom;
}

float Smith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx1 = SchlickGGX(NdotV, roughness);
    float ggx2 = SchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Fresnel equation
vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0f - roughness), F0) - F0) * pow(max(1.0f - cosTheta, 0.0f), 5.0f);
}

// PBR
vec3 CookTorranceBRDF(vec3 N, vec3 V, vec3 albedo, float metallic, float roughness, float ao, Light light)
{
    // light direction and half vector
    vec3 L = light.vector.w == 1.0f ? normalize(light.vector.xyz - fPosition) : normalize(-light.vector.xyz);
    vec3 H = normalize(V + L);

    // surface reflectance
    vec3 F0 = mix(vec3(0.04f), albedo, metallic);

    // radiance
    float d = length(light.vector.xyz - fPosition);
    float atten = GetAttenuation(light);
    vec3 radiance = light.color * atten;

    // Cook-Torrance BRDF
    float NDF = TrowbridgeReitzGGX(N, H, roughness);
    float G   = Smith(N, V, L, roughness);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0f), F0);

    vec3  numerator   = NDF * G * F;
    float denominator = 4.0f * max(dot(N, V), 0.0f) * max(dot(N, L), 0.0f) + 0.0001f;
    vec3  specular    = numerator / denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0f) - kS;
    kD *= 1.0f - metallic;

    float NdotL = max(dot(N, L), 0.0f);

    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// --------------------------------------------
// main function
// --------------------------------------------
void main()
{
    // find material texture ID
    int albedoID    = uMaterial[fMaterialID].textureID.x;
    int normalID    = uMaterial[fMaterialID].textureID.y;
    int dispID      = uMaterial[fMaterialID].textureID.z;
    
    int metallicID  = uMaterial[fMaterialID].pbrTextureID.x;
    int roughnessID = uMaterial[fMaterialID].pbrTextureID.y;
    int aoID        = uMaterial[fMaterialID].pbrTextureID.z;

    // texture scaling
    vec2 uv = fTexCoord * uvScale;

    // find material attributes
    // vec3  albedo    = uMaterial[fMaterialID].albedo.rgb;
    vec3  albedo    = pow(uMaterial[fMaterialID].albedo.rgb, vec3(GAMMA));
    float alpha     = uMaterial[fMaterialID].albedo.a;
    float metallic  = uMaterial[fMaterialID].metallic;
    float roughness = uMaterial[fMaterialID].roughness;
    float ao        = uMaterial[fMaterialID].ao;

    // normal and view direction
    vec3 N = normalize(fNormal);
    vec3 V = normalize(uViewPosition - fPosition);

    // Textures --------------------------------------------
    // displacement
    if (dispID >= 0)
    {
        mat3 TBN_t = transpose(fTBN);
        vec3 V_ = normalize(TBN_t * V);

        uv = ParallaxOcclusionMapping(uTextures[dispID], uv, V_);
        if (uv.x > 1.0f || uv.y > 1.0f || uv.x < 0.0f || uv.y < 0.0f)
        {
            discard;
        }
    }

    // albedo
    if (albedoID >= 0)
    {
        albedo = pow(texture(uTextures[albedoID], uv).rgb, vec3(GAMMA));
        // albedo = pow(texture(uTextures[normalID], uv).rgb, vec3(GAMMA));
    }

    // normal
    if (normalID >= 0)
    {
        N = GetNormalFromMap(uTextures[normalID], uv);
        // albedo = N * 0.5f + 0.5f;
    }

    // metallic
    if (metallicID >= 0)
    {
        metallic = texture(uTextures[metallicID], uv).r;
    }

    // roughness
    if (roughnessID >= 0)
    {
        roughness = texture(uTextures[roughnessID], uv).r;
    }

    // ambient occlusion
    if (aoID >= 0)
    {
        ao = texture(uTextures[aoID], uv).r;
    }

    // shadow
    vec3 lightVec = uLight[0].vector.w == 1.0f ? normalize(uLight[0].vector.xyz - fPosition) : normalize(-uLight[0].vector.xyz);
    float shadow = Shadow(fPosLightSpace, lightVec, uShadowMap);

    // --------------------------------------------
    // rendering
    vec3 color = vec3(0.0f);
    if (uColorMode)
    {
        color = albedo;
    }
    else if (uPBR)
    {
        vec3 Lo = vec3(0.0f);
        for (int i = 0; i < uLightNum; ++i)
        {
            Lo += CookTorranceBRDF(N, V, albedo, metallic, roughness, ao, uLight[i]);
        }

        vec3 F0 = mix(vec3(0.04f), albedo, metallic);
        vec3 kS = FresnelSchlickRoughness(max(dot(N, V), 0.0f), F0, roughness);
        vec3 kD = vec3(1.0f) - kS;
        kD *= 1.0f - metallic;

        vec3 irradiance = texture(uIrradianceMap, N).rgb;
        vec3 diffuse = irradiance * albedo;
        vec3 ambient = (kD * diffuse) * ao;

        color = (ambient + (1.0f - shadow) * Lo);
    }
    else
    {
        vec3 Lo = vec3(0.0f);
        for (int i = 0; i < uLightNum; ++i)
        {
            vec3 L = uLight[i].vector.w == 1.0f ? normalize(uLight[i].vector.xyz - fPosition) : normalize(-uLight[i].vector.xyz);
            vec3 H = normalize(V + L);

            Lo += BlinnPhong(albedo, N, V, L, uLight[i], uMaterial[fMaterialID]);
        }
        vec3 ambient = vec3(0.03f) * albedo * ao;
        color = ambient + (1.0f - shadow) * Lo;
    }

    // floor
    float floorWeight = 0.0f;
    if(uIsFloor)
    {
        float tile = FilteredGrid(fPosition.xz);
        // float tile = FilteredChecker(fPosition.xz);
        tile = pow(tile, 3.0f);

        floorWeight = 1.0f - tile;
    }
    color = (1.0f - floorWeight) * color + floorWeight * (1.0f - 0.95f * shadow) * uGridColor;
    
    // tone mapping
    color = ReinhardToneMapping(color);
    color = ACESFilmicToneMapping(color);

    // fog
    float D = length(uViewPosition - fPosition);
    vec3 fogColor = uSkyColor;
    float fogFactor = 1.0f - min(exp(-D * 0.03f + 1.5f), 1.0f);
    color = mix(color, fogColor, fogFactor);

    // final color
    FragColor = vec4(color, alpha);
}