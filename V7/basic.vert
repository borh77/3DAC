#version 330 core
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNrm;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inCol;

uniform mat4 uM;
uniform mat4 uVP;

out vec3 vWPos;
out vec3 vWNrm;
out vec2 vUV;
out vec4 vCol;

void main() {
    vec4 w = uM * vec4(inPos, 1.0);
    vWPos = w.xyz;
    mat3 nrmM = transpose(inverse(mat3(uM)));
    vWNrm = normalize(nrmM * inNrm);
    vUV = inUV;
    vCol = inCol;
    gl_Position = uVP * w;
}
