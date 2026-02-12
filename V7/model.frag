#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform vec3 uLightColor;
uniform float uLightIntensity;

uniform sampler2D texture_diffuse1; // ovo learnopengl koristi

void main()
{
    vec3 albedo = texture(texture_diffuse1, TexCoords).rgb;

    vec3 N = normalize(Normal);
    vec3 L = normalize(uLightPos - FragPos);
    vec3 V = normalize(uViewPos - FragPos);
    vec3 R = reflect(-L, N);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(R, V), 0.0), 32.0);

    float ambient = 0.20;
    vec3 color = (ambient + 0.85*diff + 0.25*spec) * uLightColor * uLightIntensity * albedo;

    FragColor = vec4(color, 1.0);
}
