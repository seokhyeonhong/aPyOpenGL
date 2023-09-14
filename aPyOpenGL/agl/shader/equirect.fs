#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
in vec3     fPosition;

// --------------------------------------------
// output fragment color
// --------------------------------------------
out vec4 FragColor;

// --------------------------------------------
// uniform
// --------------------------------------------
uniform sampler2D uEquirectangularMap;

// --------------------------------------------
// constants
// --------------------------------------------
const vec2 invAtan = vec2(0.1591f, 0.3183f);

vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5f;

    return uv;
}

void main()
{
    vec2 uv = SampleSphericalMap(normalize(fPosition));
    vec3 color = texture(uEquirectangularMap, uv).rgb;
    // color = pow(color, vec3(2.2f));

    FragColor = vec4(color, 1.0f);
}