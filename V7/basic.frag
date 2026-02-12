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

// Lamp (second light source - red when AC is on)
uniform vec3 uLampPos;
uniform vec3 uLampColor;
uniform float uLampIntensity;

uniform bool uEmissive; // true => no lighting (for screens/lamp)

void main() {
    vec4 base = uColor * vCol;
    if (uUseTex) base *= texture(uTex, vUV);
    base.a *= uAlpha;

    vec3 rgb = base.rgb;

    if (!uEmissive) {
        // TWO-SIDED LIGHTING: flip normal for back faces
        vec3 N = normalize(vWNrm);
        if (!gl_FrontFacing) {
            N = -N;
        }

        // === MAIN LIGHT ===
        vec3 L = normalize(uLightPos - vWPos);
        vec3 V = normalize(uCamPos - vWPos);
        vec3 R = reflect(-L, N);

        // Use abs() for fully two-sided diffuse lighting
        float lambert = abs(dot(N, L));
        float spec = pow(max(dot(R, V), 0.0), 32.0);

        float ambient = 0.10;  // INCREASED from 0.18 to avoid dark interiors
        float diffuse = 0.75 * lambert;  // reduced to compensate for abs()
        float specular = 0.20 * spec;

        vec3 mainLight = (ambient + diffuse + specular) * uLightColor * uLightIntensity;

        // === LAMP LIGHT (additive - weak red light when AC on) ===
        vec3 lampLight = vec3(0.0);
        if (uLampIntensity > 0.001) {
            vec3 L2 = normalize(uLampPos - vWPos);
            float dist = length(uLampPos - vWPos);
            float attenuation = 1.0 / (1.0 + 0.5 * dist + 0.1 * dist * dist);
            float lambert2 = abs(dot(N, L2));  // abs() here too
            lampLight = lambert2 * uLampColor * uLampIntensity * attenuation;
        }

        rgb *= (mainLight + lampLight);
    }

    outCol = vec4(rgb, base.a);
}