#version 330 core
in vec3 vWPos;
in vec3 vWNrm;
in vec2 vUV;
in vec4 vCol;

out vec4 outCol;

uniform vec4  uColor;
uniform float uAlpha;
uniform bool  uUseTex;
uniform sampler2D uTex;

uniform vec3 uCamPos;
uniform vec3 uLightPos;
uniform vec3 uLightColor;
uniform float uLightIntensity;

uniform bool uEmissive; // true => bez osvetljenja (za ekrane/lampicu)

void main() {
    vec4 base = uColor * vCol;
    if (uUseTex) base *= texture(uTex, vUV);
    base.a *= uAlpha;

    vec3 rgb = base.rgb;

    if (!uEmissive) {
        vec3 N = normalize(vWNrm);
        vec3 L = normalize(uLightPos - vWPos);
        vec3 V = normalize(uCamPos - vWPos);
        vec3 R = reflect(-L, N);

        float lambert = max(dot(N, L), 0.0);
        float spec = pow(max(dot(R, V), 0.0), 32.0);

        float ambient = 0.18;
        float diffuse = 0.85 * lambert;
        float specular = 0.25 * spec;

        vec3 lit = (ambient + diffuse + specular) * uLightColor * uLightIntensity;
        rgb *= lit;
    }

    outCol = vec4(rgb, base.a);
}
