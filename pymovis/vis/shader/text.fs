#version 430

// --------------------------------------------
// input vertex data
// --------------------------------------------
in vec2 fTexCoord;

// --------------------------------------------
// output fragment color
// --------------------------------------------
out vec4 FragColor;

// --------------------------------------------
// uniform
// --------------------------------------------
uniform sampler2D uFontTexture;
uniform vec3 uTextColor;

// --------------------------------------------
// main function
// --------------------------------------------
void main()
{
    vec4 sampled = vec4(1.0f, 1.0f, 1.0f, texture(uFontTexture, fTexCoord).r);
    FragColor = vec4(uTextColor, 1.0f) * sampled;
}