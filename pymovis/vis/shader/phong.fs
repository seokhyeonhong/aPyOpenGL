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
uniform bool      uColorMode;
uniform bool      uPBR; // true: pbr shading, false: phong shading
uniform vec2      uvScale;
uniform float     uDispScale;
uniform sampler2D uShadowMap;
uniform bool      uIsFloor;
uniform vec2      uGridSize;
uniform vec3      uGridColors[2];

// --------------------------------------------
// material structure
// --------------------------------------------
#define MAX_MATERIAL_NUM 5
struct Material {
    ivec3 textureID; // albedo, normal, displacement
    ivec3 pbrTextureID; // metallic, roughness, ao
    vec4  albedo; // RGBA

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
struct Light {
    vec4 vector; // point light if w == 1, directional light if w == 0
    vec3 color;
    vec3 attenuation; // attenuation coefficients
};
uniform Light uLight;

// --------------------------------------------
// camera position
// --------------------------------------------
uniform vec3 uViewPosition;

// --------------------------------------------
// constants
// --------------------------------------------
const float PI = 3.14159265359f;

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
    float bias = max(0.0001f * (1.0f - dot(fNormal, lightDir)), 0.00001f);

    // if current depth from camera is greater than that of the light source,
    // then the fragment is in shadow
    // float shadow = currentDepth > closestDepth ? 1.0 : 0.0;
    float shadow = 0.0f;
    vec2 texelSize = 1.0f / textureSize(shadowMap, 0);
    for(int u = -1; u <= 1; ++u)
    {
        for(int v = -1; v <= 1; ++v)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(u, v) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0f : 0.0f;
        }
    }
    shadow /= 9.0f;
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

vec4 BlinnPhong(vec3 albedo, vec3 N, vec3 V, vec3 L, Light light, Material material)
{
    // ambient
    vec3 ambient = albedo * 0.05f;

    // diffuse
    vec3 diffuse = max(dot(N, L), 0.0f) * material.diffuse * light.color;

    // specular
    // vec3 R = reflect(-L, N); // for phong shading, use R instead of H
    vec3 H = normalize(L + V); // for blinn-phong shading, use H instead of R
    vec3 specular = pow(max(dot(V, H), 0.0f), material.shininess) * material.specular * light.color;
    
    // attenuation
    float atten = GetAttenuation(light);

    // shadow
    float shadow = Shadow(fPosLightSpace, L, uShadowMap);

    // final color
    vec3 result = (ambient + atten * (1.0f - shadow) * (diffuse + specular)) * albedo;
    return vec4(result, 1.0f);
}

// --------------------------------------------
vec3 ReinhardToneMapping(vec3 color)
{
    const float gamma = 2.2f;
    vec3 result = color / (color + vec3(1.0f));
    return pow(result, vec3(1.0f / gamma));
}

// --------------------------------------------
// Reference: https://iquilezles.org/articles/checkerfiltering/
// Shadertoy: https://www.shadertoy.com/view/ss3yzr
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
vec3 GetNormalFromMap(sampler2D normalMap, vec2 uv)
{
    vec3 N = texture(normalMap, uv).rgb * 2.0f - 1.0f;
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

// PBR
vec3 CookTorranceBRDF(vec3 N, vec3 V, vec3 L, vec3 H, vec3 albedo, float metallic, float roughness, float ao, Light light)
{
    float d = length(light.vector.xyz - fPosition);
    float atten = GetAttenuation(light);
    vec3 radiance = light.color * atten;

    vec3 F0 = vec3(0.04f);
    F0 = mix(F0, albedo, metallic);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0f), F0);

    float NDF = TrowbridgeReitzGGX(N, H, roughness);
    float G = Smith(N, V, L, roughness);

    vec3 num = NDF * G * F;
    float denom = 4.0f * max(dot(N, V), 0.0f) * max(dot(N, L), 0.0f) + 0.0001f;
    vec3 specular = num / denom;

    vec3 kS = F;
    vec3 kD = vec3(1.0f) - kS;
    kD *= 1.0f - metallic;

    float NdotL = max(dot(N, L), 0.0f);

    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;
    // vec3 ambient = (kD * diffuse) * ao;
    vec3 ambient = vec3(0.03f) * albedo * ao;

    // shadow
    float shadow = Shadow(fPosLightSpace, L, uShadowMap);

    // final color
    vec3 result = ((1.0f - shadow) * (Lo + ambient));

    return result;
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
    vec3  albedo    = uMaterial[fMaterialID].albedo.rgb;
    float alpha     = uMaterial[fMaterialID].albedo.a;
    float metallic  = uMaterial[fMaterialID].metallic;
    float roughness = uMaterial[fMaterialID].roughness;
    float ao        = uMaterial[fMaterialID].ao;

    // normal, view, light and half vectors
    vec3 N = normalize(fNormal);
    vec3 V = normalize(uViewPosition - fPosition);
    vec3 L = uLight.vector.w == 1.0f ? normalize(uLight.vector.xyz - fPosition) : normalize(-uLight.vector.xyz);
    vec3 H = normalize(V + L);

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
        albedo = texture(uTextures[albedoID], uv).rgb;
    }

    // normal
    if (normalID >= 0)
    {
        N = GetNormalFromMap(uTextures[normalID], uv);
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

    // --------------------------------------------
    // rendering
    if (uIsFloor)
    {
        FragColor.rgb = FilterGrid(fPosition.xz);
        FragColor.a = 1.0f;
        return;
    }
    else if (uColorMode)
    {
        FragColor = vec4(albedo, 1.0f);
    }
    else if (uPBR)
    {
        vec3 color = vec3(0.0f);
        for (int i = 0; i < 4; ++i)
        {
            color += CookTorranceBRDF(N, V, L, H, albedo, metallic, roughness, ao, uLight);
        }

        // FragColor.rgb = CookTorranceBRDF(N, V, L, H, albedo, uLight, uMaterial[fMaterialID]);
        FragColor.rgb = color;
        // FragColor.rgb = ReinhardToneMapping(FragColor.rgb);
        FragColor.a = alpha;
    }
    else
    {
        FragColor = BlinnPhong(albedo, N, V, L, uLight, uMaterial[fMaterialID]);
        FragColor.a = alpha;
    }

    // FragColor.rgb = ReinhardToneMapping(FragColor.rgb);

    // Fog
    // float D = length(uViewPosition - fPosition);
    // vec3 fog_color = vec3(0.5);
    // float fog_amount = 1.0f - min(exp(-D * 0.1 + 1.5), 1.0);
    // vec3 color = FragColor.rgb;
    // color = mix(color, fog_color, fog_amount);
    // FragColor.rgb = color;
    // vec3 fogColor = vec3(0.5);
    // float d = length(fPosition - uViewPosition);
    // float fogFactor = clamp((d - 10.0) / 10.0, 0.0, 1.0);
    // fogColor = fogColor * fogFactor;
    // FragColor.rgb = GammaCorrection(FragColor.rgb, 1.0 / 2.2);// + fogColor;
}