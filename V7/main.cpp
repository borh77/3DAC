#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Util.h"

// stb_image samo za učitavanje PNG kursora (GLFW cursor)
#include "stb_image.h"

// -------------------------------
// Helpers
// -------------------------------

static GLuint preprocessTexture(const char* filepath) {
    // relies on loadImageToTexture implemented in Util.cpp
    GLuint tex = loadImageToTexture(filepath);
    if (tex == 0) return 0;
    glBindTexture(GL_TEXTURE_2D, tex);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return tex;
}
static inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

struct Mesh {
    GLuint vao{ 0 }, vbo{ 0 }, ebo{ 0 };
    GLsizei indexCount{ 0 };
    bool indexed{ false };
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nrm;
    glm::vec2 uv;
    glm::vec4 col;
};

// New helper: ensure majority of triangles are CCW (front faces outward);
// If most triangles are inverted we flip every triangle's winding;
// This runs in all builds and will auto-fix generators that produced inverted meshes.
static void enforceCCW(std::vector<Vertex>& verts, std::vector<unsigned int>& idx, const char* name) {
    if (idx.empty()) return;
    glm::vec3 center(0.0f);
    for (auto& v : verts) center += v.pos;
    center /= (float)verts.size();

    int bad = 0, total = 0;
    for (size_t i = 0; i + 2 < idx.size(); i += 3) {
        glm::vec3 p0 = verts[idx[i + 0]].pos;
        glm::vec3 p1 = verts[idx[i + 1]].pos;
        glm::vec3 p2 = verts[idx[i + 2]].pos;
        glm::vec3 fn = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        glm::vec3 triCent = (p0 + p1 + p2) / 3.0f;
        glm::vec3 out = glm::normalize(triCent - center);
        if (glm::length(out) < 1e-6f) continue;
        if (glm::dot(fn, out) < 0.0f) ++bad;
        ++total;
    }

    if (total == 0) return;

    // If majority of triangles are inward-facing, flip every triangle to make them CCW outward.
    if (bad > total / 2) {
        for (size_t i = 0; i + 2 < idx.size(); i += 3) {
            std::swap(idx[i + 1], idx[i + 2]); // flip winding
        }
        std::cout << "WINDING FIX: mesh '" << name << "' had " << bad << " / " << total << " inward-facing; flipped all triangles to CCW\n";
    }
#ifndef NDEBUG
    else if (bad > 0) {
        // still warn in debug if there are any stray inverted triangles
        std::cout << "WINDING WARNING: mesh '" << name << "' has " << bad << " / " << total << " inward-facing triangles\n";
    }
#endif
}

static Mesh createMesh(const std::vector<Vertex>& verts, const std::vector<unsigned int>& idx) {
    Mesh m;
    glGenVertexArrays(1, &m.vao);
    glBindVertexArray(m.vao);

    glGenBuffers(1, &m.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(Vertex)), verts.data(), GL_STATIC_DRAW);

    if (!idx.empty()) {
        m.indexed = true;
        m.indexCount = (GLsizei)idx.size();
        glGenBuffers(1, &m.ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(idx.size() * sizeof(unsigned int)), idx.data(), GL_STATIC_DRAW);
    }
    else {
        m.indexed = false;
        m.indexCount = (GLsizei)verts.size();
    }

    // layout:
    // 0 pos (vec3)
    // 1 normal (vec3)
    // 2 uv (vec2)
    // 3 color (vec4)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nrm));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, col));

    glBindVertexArray(0);
    return m;
}

static void destroyMesh(Mesh& m) {
    if (m.ebo) glDeleteBuffers(1, &m.ebo);
    if (m.vbo) glDeleteBuffers(1, &m.vbo);
    if (m.vao) glDeleteVertexArrays(1, &m.vao);
    m = {};
}

// Optional debug helper to validate winding of generated indexed meshes.
// Prints a warning in debug builds if many triangles appear flipped.
static void validateWinding(const std::vector<Vertex>& verts, const std::vector<unsigned int>& idx, const char* name) {
#ifndef NDEBUG
    if (idx.empty()) return;
    glm::vec3 center(0.0f);
    for (auto &v : verts) center += v.pos;
    center /= (float)verts.size();

    int bad = 0, total = 0;
    for (size_t i = 0; i + 2 < idx.size(); i += 3) {
        glm::vec3 p0 = verts[idx[i + 0]].pos;
        glm::vec3 p1 = verts[idx[i + 1]].pos;
        glm::vec3 p2 = verts[idx[i + 2]].pos;
        glm::vec3 fn = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        glm::vec3 triCent = (p0 + p1 + p2) / 3.0f;
        glm::vec3 out = glm::normalize(triCent - center);
        if (glm::length(out) < 1e-6f) continue;
        if (glm::dot(fn, out) < 0.0f) ++bad;
        ++total;
    }
    if (total > 0 && bad > 0) {
        std::cout << "WINDING WARNING: mesh '" << name << "' has " << bad << " / " << total << " triangles inward-facing\n";
    }
#endif
}

// -------------------------------
// Mesh generators (consistent CCW winding)
// -------------------------------

static Mesh makeCube(glm::vec4 col = glm::vec4(1, 1, 1, 1)) {
    // Unit cube centered at origin, size 1
    const glm::vec3 p[8] = {
        {-0.5f,-0.5f,-0.5f}, {0.5f,-0.5f,-0.5f}, {0.5f,0.5f,-0.5f}, {-0.5f,0.5f,-0.5f},
        {-0.5f,-0.5f, 0.5f}, {0.5f,-0.5f, 0.5f}, {0.5f,0.5f, 0.5f}, {-0.5f,0.5f, 0.5f}
    };
    struct Face { int a, b, c, d; glm::vec3 n; };
    const Face faces[6] = {
        {4,5,6,7, {0,0,1}},  // front
        {1,0,3,2, {0,0,-1}}, // back
        {0,4,7,3, {-1,0,0}}, // left
        {5,1,2,6, {1,0,0}},  // right
        {3,7,6,2, {0,1,0}},  // top
        {0,1,5,4, {0,-1,0}}  // bottom
    };
    std::vector<Vertex> v;
    v.reserve(24);
    for (auto& f : faces) {
        v.push_back({ p[f.a], f.n, {0,0}, col });
        v.push_back({ p[f.b], f.n, {1,0}, col });
        v.push_back({ p[f.c], f.n, {1,1}, col });
        v.push_back({ p[f.d], f.n, {0,1}, col });
    }
    std::vector<unsigned int> idx;
    idx.reserve(36);
    for (int i = 0; i < 6; i++) {
        unsigned int base = i * 4;
        // triangles CCW when front is facing outward
        idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
        idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
    }
    validateWinding(v, idx, "cube"); // optional debug message
    enforceCCW(v, idx, "cube");      // ensure CCW for runtime/builds
    return createMesh(v, idx);
}

static Mesh makeCylinder(float radius, float height, int slices, glm::vec4 col,
    bool hollow, float innerRadius = 0.0f, bool reverseWinding = false, bool reverseNormals = false)
{
    std::vector<Vertex> v;
    std::vector<unsigned int> idx;

    const float y0 = -height * 0.5f;
    const float y1 = height * 0.5f;

    // Generate vertices in a ring around Y axis
    for (int i = 0; i <= slices; i++) {
        float t = (float)i / (float)slices;
        float a = t * 2.0f * 3.1415926f;
        float x = std::cos(a) * radius;
        float z = std::sin(a) * radius;
        glm::vec3 n = glm::normalize(glm::vec3(x, 0, z));
        if (reverseNormals) n = -n; // flip normal direction for inner walls
        v.push_back({ {x, y0, z}, n, {t, 0}, col });
        v.push_back({ {x, y1, z}, n, {t, 1}, col });
    }

    // Generate indices for the cylinder side
    for (int i = 0; i < slices; i++) {
        int k = i * 2;
        if (!reverseWinding) {
            // CCW winding (outer wall - front faces point outward)
            idx.push_back(k + 0); idx.push_back(k + 1); idx.push_back(k + 3);
            idx.push_back(k + 0); idx.push_back(k + 3); idx.push_back(k + 2);
        } else {
            // CW winding (inner wall - front faces point inward)
            idx.push_back(k + 0); idx.push_back(k + 3); idx.push_back(k + 1);
            idx.push_back(k + 0); idx.push_back(k + 2); idx.push_back(k + 3);
        }
    }

    return createMesh(v, idx);
}   
// Insert this function after makeCylinder(...) in main.cpp

static Mesh makeThickCylinder(float outerRadius, float innerRadius, float height, int slices, glm::vec4 col) {
    // Creates a thick-walled cylinder (closed annular volume) with top/bottom caps.
    // outerRadius > innerRadius required.
    std::vector<Vertex> v;
    std::vector<unsigned int> idx;

    const float y0 = -height * 0.5f;
    const float y1 = height * 0.5f;

    // Generate ring vertices (duplicate last = first for easy indexing)
    for (int i = 0; i <= slices; ++i) {
        float t = (float)i / (float)slices;
        float a = t * 2.0f * 3.1415926f;
        float ox = std::cos(a) * outerRadius;
        float oz = std::sin(a) * outerRadius;
        glm::vec3 on = glm::normalize(glm::vec3(ox, 0.0f, oz));

        float ix = std::cos(a) * innerRadius;
        float iz = std::sin(a) * innerRadius;
        glm::vec3 in = glm::normalize(glm::vec3(ix, 0.0f, iz));

        // outer bottom, outer top, inner bottom, inner top
        v.push_back({ {ox, y0, oz},  on, {t, 0}, col }); // outer bottom
        v.push_back({ {ox, y1, oz},  on, {t, 1}, col }); // outer top
        v.push_back({ {ix, y0, iz}, -in, {t, 0}, col }); // inner bottom (normal toward -radial)
        v.push_back({ {ix, y1, iz}, -in, {t, 1}, col }); // inner top
    }

    // Build side faces (outer and inner)
    for (int i = 0; i < slices; ++i) {
        int base = i * 4;
        int next = (i + 1) * 4;

        int ob0 = base + 0; int ot0 = base + 1;
        int ib0 = base + 2; int it0 = base + 3;
        int ob1 = next + 0; int ot1 = next + 1;
        int ib1 = next + 2; int it1 = next + 3;

        // Outer side (front facing outward) - two triangles
        idx.push_back(ob0); idx.push_back(ot0); idx.push_back(ot1);
        idx.push_back(ob0); idx.push_back(ot1); idx.push_back(ob1);

        // Inner side (front facing inward) - reverse winding so front faces inward
        idx.push_back(ib0); idx.push_back(ib1); idx.push_back(it1);
        idx.push_back(ib0); idx.push_back(it1); idx.push_back(it0);
    }

    // Top cap (annulus) at y1: connect outer top -> inner top
    for (int i = 0; i < slices; ++i) {
        int base = i * 4;
        int next = (i + 1) * 4;

        int ot0 = base + 1;
        int ot1 = next + 1;
        int it0 = base + 3;
        int it1 = next + 3;

        // When looking from +Y, triangles must be CCW -> outer -> outer_next -> inner_next
        idx.push_back(ot0); idx.push_back(ot1); idx.push_back(it1);
        idx.push_back(ot0); idx.push_back(it1); idx.push_back(it0);
    }

    // Bottom cap (annulus) at y0: connect inner bottom -> outer bottom (caps face -Y)
    for (int i = 0; i < slices; ++i) {
        int base = i * 4;
        int next = (i + 1) * 4;

        int ob0 = base + 0;
        int ob1 = next + 0;
        int ib0 = base + 2;
        int ib1 = next + 2;

        // For bottom (looking from -Y), keep CCW when looking along -Y:
        // inner -> inner_next -> outer_next
        idx.push_back(ib0); idx.push_back(ib1); idx.push_back(ob1);
        idx.push_back(ib0); idx.push_back(ob1); idx.push_back(ob0);
    }

    validateWinding(v, idx, "thickCylinder");
    enforceCCW(v, idx, "thickCylinder");
    return createMesh(v, idx);
}
static Mesh makeDisk(float radius, int slices, glm::vec4 col, glm::vec3 normal, float y) {
    std::vector<Vertex> v;
    std::vector<unsigned int> idx;
    v.reserve(slices + 2);
    v.push_back({ {0,y,0}, normal, {0.5f,0.5f}, col });

    // generate ring in increasing angle (CCW around +Y reference)
    for (int i = 0; i <= slices; i++) {
        float a = (float)i / (float)slices * 2.0f * 3.1415926f;
        float x = std::cos(a) * radius;
        float z = std::sin(a) * radius;
        v.push_back({ {x,y,z}, normal, {0.5f + x / (2 * radius), 0.5f + z / (2 * radius)}, col });
    }

    // Decide triangle winding based on provided normal.
    // If normal approximately points up (+Y), emit (0, i+1, i) for CCW when looking along normal.
    glm::vec3 upRef = glm::vec3(0.0f, 1.0f, 0.0f);
    bool normalUp = glm::dot(glm::normalize(normal), upRef) > 0.0f;

    for (int i = 1; i <= slices; i++) {
        if (normalUp) {
            idx.push_back(0);
            idx.push_back(i + 1);
            idx.push_back(i);
        } else {
            idx.push_back(0);
            idx.push_back(i);
            idx.push_back(i + 1);
        }
    }

    validateWinding(v, idx, "disk");
    enforceCCW(v, idx, "disk");
    return createMesh(v, idx);
}

static Mesh makeSphere(float r, int stacks, int slices, glm::vec4 col) {
    std::vector<Vertex> v;
    std::vector<unsigned int> idx;
    for (int i = 0; i <= stacks; i++) {
        float v0 = (float)i / (float)stacks;
        float phi = v0 * 3.1415926f;
        for (int j = 0; j <= slices; j++) {
            float u0 = (float)j / (float)slices;
            float theta = u0 * 2.0f * 3.1415926f;
            float x = std::sin(phi) * std::cos(theta);
            float y = std::cos(phi);
            float z = std::sin(phi) * std::sin(theta);
            glm::vec3 n = glm::normalize(glm::vec3(x, y, z));
            v.push_back({ r * n, n, {u0, v0}, col });
        }
    }
    int cols = slices + 1;
    for (int i = 0; i < stacks; i++) {
        for (int j = 0; j < slices; j++) {
            int a = i * cols + j;
            int b = a + cols;
            // CCW ordering so front faces point outward
            idx.push_back(a);        idx.push_back(a + 1);    idx.push_back(b);
            idx.push_back(a + 1);    idx.push_back(b + 1);    idx.push_back(b);
        }
    }
    validateWinding(v, idx, "sphere");
    enforceCCW(v, idx, "sphere");
    return createMesh(v, idx);
}

// -------------------------------
// Camera + Input
// -------------------------------

static bool firstMouse = true;
static float lastX = 0.0f, lastY = 0.0f;
static glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
static float camSpeed = 2.5f; // units/sec
static float yaw = -90.0f, pitch = -12.0f;     // gledaj malo naniže
static glm::vec3 cameraPos(0.0f, 1.35f, 5.5f); // malo dalje i malo više
static glm::vec3 cameraFront(0.0f, 0.0f, -1.0f);
static float fov = 62.0f;                      // širi ugao


static bool gDepthTest = true;
static bool gCullFace = true;
// New toggles: floor ("pod") and basin wall ("zid")
static bool gShowFloor = false;
static bool gShowBasinWall = true;

static void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) { lastX = (float)xpos; lastY = (float)ypos; firstMouse = false; }
    float xoffset = (float)xpos - lastX;
    float yoffset = lastY - (float)ypos;
    lastX = (float)xpos; lastY = (float)ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;
    pitch = clampf(pitch, -89.0f, 89.0f);

    glm::vec3 dir;
    dir.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
    dir.y = std::sin(glm::radians(pitch));
    dir.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
    cameraFront = glm::normalize(dir);
}

static void scroll_callback(GLFWwindow* window, double, double yoffset) {
    (void)window;
    fov -= (float)yoffset;
    fov = clampf(fov, 25.0f, 75.0f);
}

static void applyGLToggles() {
    if (gDepthTest) glEnable(GL_DEPTH_TEST);
    else glDisable(GL_DEPTH_TEST);
    if (gCullFace) { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); glFrontFace(GL_CCW); }
    else glDisable(GL_CULL_FACE);
    

}

// -------------------------------
// Picking (ray vs sphere)
// -------------------------------

static bool raySphere(const glm::vec3& ro, const glm::vec3& rd, const glm::vec3& c, float r, float& tHit) {
    glm::vec3 oc = ro - c;
    float b = glm::dot(oc, rd);
    float c2 = glm::dot(oc, oc) - r * r;
    float disc = b * b - c2;
    if (disc < 0.0f) return false;
    float t = -b - std::sqrt(disc);
    if (t < 0.0f) t = -b + std::sqrt(disc);
    if (t < 0.0f) return false;
    tHit = t;
    return true;
}

// -------------------------------
// 7-seg (3D as quads)
// -------------------------------

// Segment indices: 0 top,1 top-left,2 top-right,3 mid,4 bot-left,5 bot-right,6 bottom
static const int SEG_MAP[10][7] = {
    {1,1,1,0,1,1,1}, //0
    {0,0,1,0,0,1,0}, //1
    {1,0,1,1,1,0,1}, //2
    {1,0,1,1,0,1,1}, //3
    {0,1,1,1,0,1,0}, //4
    {1,1,0,1,0,1,1}, //5
    {1,1,0,1,1,1,1}, //6
    {1,0,1,0,0,1,0}, //7
    {1,1,1,1,1,1,1}, //8
    {1,1,1,1,0,1,1}  //9
};

struct SegDigit {
    glm::vec3 center;
    float w;
    float h;
};

static void drawSegDigit(GLuint prog, const Mesh& quad, const glm::mat4& VP, const SegDigit& d, int value,
    const glm::vec4& onCol, const glm::vec4& offCol) {
    value = std::abs(value) % 10;
    auto segOn = [&](int s) { return SEG_MAP[value][s] == 1; };

    // segment local transforms
    const float t = 0.12f * d.h; // thickness
    const float pad = 0.06f * d.h;
    const float hw = d.w * 0.5f;
    const float hh = d.h * 0.5f;

    struct Seg { glm::vec3 pos; glm::vec3 scale; };
    Seg segs[7] = {
        {{d.center.x, d.center.y + hh - pad, d.center.z}, {d.w - 2 * pad, t, 1}},
        {{d.center.x - hw + pad, d.center.y + hh * 0.5f, d.center.z}, {t, d.h * 0.5f - pad, 1}},
        {{d.center.x + hw - pad, d.center.y + hh * 0.5f, d.center.z}, {t, d.h * 0.5f - pad, 1}},
        {{d.center.x, d.center.y, d.center.z}, {d.w - 2 * pad, t, 1}},
        {{d.center.x - hw + pad, d.center.y - hh * 0.5f, d.center.z}, {t, d.h * 0.5f - pad, 1}},
        {{d.center.x + hw - pad, d.center.y - hh * 0.5f, d.center.z}, {t, d.h * 0.5f - pad, 1}},
        {{d.center.x, d.center.y - hh + pad, d.center.z}, {d.w - 2 * pad, t, 1}},
    };

    glBindVertexArray(quad.vao);
    glUniform1i(glGetUniformLocation(prog, "uUseTex"), 0);
    glUniform1i(glGetUniformLocation(prog, "uEmissive"), 1);
    glUniform1f(glGetUniformLocation(prog, "uAlpha"), 1.0f);

    for (int s = 0; s < 7; s++) {
        glm::vec4 c = segOn(s) ? onCol : offCol;
        glUniform4fv(glGetUniformLocation(prog, "uColor"), 1, glm::value_ptr(c));
        glm::mat4 M(1.0f);
        M = glm::translate(M, segs[s].pos);
        M = glm::scale(M, segs[s].scale);
        glUniformMatrix4fv(glGetUniformLocation(prog, "uM"), 1, GL_FALSE, glm::value_ptr(M));
        glUniformMatrix4fv(glGetUniformLocation(prog, "uVP"), 1, GL_FALSE, glm::value_ptr(VP));
        if (quad.indexed) glDrawElements(GL_TRIANGLES, quad.indexCount, GL_UNSIGNED_INT, 0);
        else glDrawArrays(GL_TRIANGLES, 0, quad.indexCount);
    }
    glUniform1i(glGetUniformLocation(prog, "uEmissive"), 0);
}

static void drawThickLine2D(
    GLuint prog, const Mesh& quad, const glm::mat4& VP,
    glm::vec3 a, glm::vec3 b,
    float thickness,
    const glm::vec4& col
) {
    glm::vec3 d = b - a;
    float len = glm::length(glm::vec2(d.x, d.y));
    if (len < 1e-5f) return;

    glm::vec3 mid = (a + b) * 0.5f;
    float ang = std::atan2(d.y, d.x); // rotacija u XY

    glUniform1i(glGetUniformLocation(prog, "uUseTex"), 0);
    glUniform1i(glGetUniformLocation(prog, "uEmissive"), 1);
    glUniform1f(glGetUniformLocation(prog, "uAlpha"), 1.0f);
    glUniform4fv(glGetUniformLocation(prog, "uColor"), 1, glm::value_ptr(col));

    glm::mat4 M(1.0f);
    M = glm::translate(M, mid);
    M = glm::rotate(M, ang, glm::vec3(0, 0, 1));
    M = glm::scale(M, glm::vec3(len, thickness, 1.0f));

    glUniformMatrix4fv(glGetUniformLocation(prog, "uM"), 1, GL_FALSE, glm::value_ptr(M));
    glUniformMatrix4fv(glGetUniformLocation(prog, "uVP"), 1, GL_FALSE, glm::value_ptr(VP));

    glBindVertexArray(quad.vao);
    glDrawElements(GL_TRIANGLES, quad.indexCount, GL_UNSIGNED_INT, 0);

    glUniform1i(glGetUniformLocation(prog, "uEmissive"), 0);
}




enum class BasinState { OnFloor, InHands };

struct Droplet {
    glm::vec3 pos;
    glm::vec3 vel;
    float life;
};

int main() {
    glewExperimental = GL_TRUE;
    if (!glfwInit()) {
        std::cout << "GLFW nije mogao da se inicijalizuje.\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    
    // Fullscreen
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    const int wWidth = mode->width;
    const int wHeight = mode->height;

    GLFWwindow* window = glfwCreateWindow(wWidth, wHeight, "3D Klima", monitor, nullptr);
    if (!window) {
        std::cout << "Prozor nije napravljen.\n";
        glfwTerminate();
        return 2;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // frame limiter radimo sami

// Sakrij kursor potpuno
glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

glfwSetCursorPosCallback(window, mouse_callback);
glfwSetScrollCallback(window, scroll_callback);

    if (glewInit() != GLEW_OK) {
        std::cout << "GLEW nije mogao da se ucita.\n";
        return 3;
    }

    // Blending (za providnost)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 
    GLuint prog = createShader("basic.vert", "basic.frag");
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048];
        glGetProgramInfoLog(prog, 2048, nullptr, log);
        std::cout << "PROGRAM LINK ERROR:\n" << log << "\n";
    }

    // Cursor (daljinski) - ZAKOMENTARISANO, koristimo default cursor
/*
{
    int cw, ch, cc;
    unsigned char* px = stbi_load("res/cursor_remote.png", &cw, &ch, &cc, 4);
    if (px) {
        GLFWimage img{ cw, ch, px };
        // laserska tačka treba da bude gore-levo
        GLFWcursor* cur = glfwCreateCursor(&img, 2, 2);
        if (cur) glfwSetCursor(window, cur);
        stbi_image_free(px);
    }
}
*/

    // Meshes
    Mesh cube = makeCube();
    // Quad in XY plane
    Mesh quad;
    {
        std::vector<Vertex> v = {
            {{-0.5f,-0.5f,0},{0,0,1},{0,0},{1,1,1,1}},
            {{ 0.5f,-0.5f,0},{0,0,1},{1,0},{1,1,1,1}},
            {{ 0.5f, 0.5f,0},{0,0,1},{1,1},{1,1,1,1}},
            {{-0.5f, 0.5f,0},{0,0,1},{0,1},{1,1,1,1}},
        };
        std::vector<unsigned int> i = { 0,1,2, 0,2,3 };
        quad = createMesh(v, i);
    }
    Mesh lampSphere = makeSphere(1.0f, 10, 16, { 1,1,1,1 });
    Mesh dropletSphere = makeSphere(1.0f, 8, 12, { 1,1,1,1 });
    // Thick-walled basin (outer + inner + caps)
    Mesh basinWall = makeThickCylinder(1.00f, 0.82f, 1.0f, 32, { 0.88f,0.88f,0.90f,1 }); // solid thickness

    Mesh basinBottom = makeDisk(0.82f, 32, { 0.75f,0.75f,0.75f,1 }, { 0,1,0 }, -0.5f);
    
    Mesh waterTop = makeDisk(0.80f, 32, { 0.35f,0.75f,0.95f,0.55f }, { 0,1,0 }, 0.0f);
    Mesh waterCyl = makeCylinder(0.80f, 1.0f, 32, { 0.35f,0.75f,0.95f,0.55f }, false, 0.0f, false);
    

    // Textures
    GLuint texStudent = preprocessTexture("res/ime.png");
    GLuint texRemote = preprocessTexture("res/remote.png");

    // Scene constants

    const float WATER_SURFACE_MAX_LOCAL = 0.42f; // top water local y (mora biti < 0.5)
    const float WATER_SURFACE_MIN_LOCAL = -0.5f; // basin bottom local y

    const glm::vec3 acPos(0.0f, 2.0f, 0.0f);
    const glm::vec3 ventPos = acPos + glm::vec3(0.0f, -0.28f, 0.55f);

    // lavor ispod vent-a (x,z poravnati)
    const glm::vec3 basinPos(ventPos.x, 0.65f, ventPos.z);

    // WC šolja IZA NAS - tamo gde se prosipa voda (iza klime, u pravcu +Z)
    const glm::vec3 toiletPos(0.0f, 0.0f, 9.0f);  // Dalje na +Z (iza nas kad gledamo klim u)

    const glm::vec3 lampPos = acPos + glm::vec3(0.42f, -0.12f, 0.56f); // lowered Y from -0.02 -> -0.12

    const float lampRadius = 0.04f;


    // AC state
    bool acOn = false;
    bool acLocked = false; // zaključano kad se lavor napuni
    float flap = 0.0f;     // 0 closed -> 1 open
    const float flapSpeed = 1.0f; // per second

    int desiredTemp = 24;
    float measuredTemp = 30.0f;
    const float tempRate = 1.0f; // deg/sec

    BasinState basinState = BasinState::OnFloor;
    bool basinEmptied = false;
    float basinFill = 0.0f;      // 0..1
    float basinFillTimer = 0.0f; // sec

    std::vector<Droplet> droplets;
    float dropletSpawnTimer = 0.0f;

    // Light
    glm::vec3 lightPos(1.8f, 4.5f, 2.0f);
    glm::vec3 lightColor(1.0f, 0.95f, 0.85f);
    float lightIntensity = 1.2f;

    auto setCommonUniforms = [&](const glm::mat4& VP) {
        glUniformMatrix4fv(glGetUniformLocation(prog, "uVP"), 1, GL_FALSE, glm::value_ptr(VP));
        glUniform3fv(glGetUniformLocation(prog, "uCamPos"), 1, glm::value_ptr(cameraPos));
        glUniform3fv(glGetUniformLocation(prog, "uLightPos"), 1, glm::value_ptr(lightPos));
        glUniform3fv(glGetUniformLocation(prog, "uLightColor"), 1, glm::value_ptr(lightColor));
        glUniform1f(glGetUniformLocation(prog, "uLightIntensity"), lightIntensity);
        };

    auto drawMesh = [&](const Mesh& m, const glm::mat4& M, const glm::mat4& VP,
        const glm::vec4& color, bool useTex, GLuint tex, float alpha,
        bool emissive) {
            glUniformMatrix4fv(glGetUniformLocation(prog, "uM"), 1, GL_FALSE, glm::value_ptr(M));
            glUniformMatrix4fv(glGetUniformLocation(prog, "uVP"), 1, GL_FALSE, glm::value_ptr(VP));
            glUniform4fv(glGetUniformLocation(prog, "uColor"), 1, glm::value_ptr(color));
            glUniform1i(glGetUniformLocation(prog, "uUseTex"), useTex ? 1 : 0);
            glUniform1i(glGetUniformLocation(prog, "uEmissive"), emissive ? 1 : 0);
            glUniform1f(glGetUniformLocation(prog, "uAlpha"), alpha);
            if (useTex) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, tex);
            }
            glBindVertexArray(m.vao);
            if (m.indexed) glDrawElements(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, 0);
            else glDrawArrays(GL_TRIANGLES, 0, m.indexCount);
        };

    // Timing
    using clock = std::chrono::high_resolution_clock;
    auto prev = clock::now();

    // Input helpers
    bool upPrev = false, downPrev = false, spacePrev = false, zPrev = false, xPrev = false, pPrev = false, bPrev = false;

    // PATCH 2: pouzdan click-edge polling
    bool mousePrev = false;

    // Main loop
    const double targetFrame = 1.0 / 75.0;
    while (!glfwWindowShouldClose(window)) {
        auto frameStart = clock::now();

        glfwPollEvents();

        // Escape exit
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Toggles: Z depth, X cull
        bool zDown = glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS;
        bool xDown = glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS;
        if (zDown && !zPrev) gDepthTest = !gDepthTest;
        if (xDown && !xPrev) gCullFace = !gCullFace;
        zPrev = zDown; xPrev = xDown;

        // NEW: toggles for floor/basin
        bool pDown = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
        bool bDown = glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS;
        if (pDown && !pPrev) gShowFloor = !gShowFloor;
        if (bDown && !bPrev) gShowBasinWall = !gShowBasinWall;
        pPrev = pDown; bPrev = bDown;

        applyGLToggles();

        // Delta time
        auto now = clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        dt = clampf(dt, 0.0f, 0.05f);

        // ZAKOMENTARISANO: WASD/E/Q kretanje - koristimo samo mouse look
/*
// PATCH 1: kretanje kamere (WASD + SPACE/SHIFT)
{
    // kretanje po "podu" (XZ), da ne letiš kad gledaš gore/dole
    glm::vec3 fwd = glm::normalize(glm::vec3(cameraFront.x, 0.0f, cameraFront.z));
    if (glm::length(fwd) < 0.0001f) fwd = glm::vec3(0, 0, -1);
    glm::vec3 right = glm::normalize(glm::cross(fwd, cameraUp)); // cameraUp je (0,1,0)
    float speed = camSpeed;

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) speed *= 2.0f;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraPos += fwd * speed * dt;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= fwd * speed * dt;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraPos += right * speed * dt;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= right * speed * dt;

    // PATCH 1: kretanje kamere (WASD + E/Q)  -- SPACE je rezervisan za akcije lavora
    {
        // kretanje po "podu" (XZ), da ne letiš kad gledaš gore/dole
        glm::vec3 fwd = glm::normalize(glm::vec3(cameraFront.x, 0.0f, cameraFront.z));
        if (glm::length(fwd) < 0.0001f) fwd = glm::vec3(0, 0, -1);
        glm::vec3 right = glm::normalize(glm::cross(fwd, cameraUp)); // cameraUp je (0,1,0)
        float speed = camSpeed;

        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) speed *= 2.0f;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraPos += fwd * speed * dt;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= fwd * speed * dt;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraPos += right * speed * dt;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= right * speed * dt;

        // vertikalno (E gore, Q dole)
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) cameraPos += cameraUp * speed * dt;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) cameraPos -= cameraUp * speed * dt;

        // mali clamp da ne upadne ispod poda
        cameraPos.y = std::max(0.25f, cameraPos.y);
    }

    // mali clamp da ne upadne ispod poda
    cameraPos.y = std::max(0.25f, cameraPos.y);
}
*/

// Desired temp input
bool upDown = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
bool downDown = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
if (upDown && !upPrev) desiredTemp = std::min(40, desiredTemp + 1);
if (downDown && !downPrev) desiredTemp = std::max(-10, desiredTemp - 1);
upPrev = upDown; downPrev = downDown;

// SPACE actions (edge)
bool spaceDown = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
bool spaceEdge = spaceDown && !spacePrev;
spacePrev = spaceDown;

// Ray picking from camera center
glm::vec3 ro = cameraPos;
glm::vec3 rd = glm::normalize(cameraFront);

// PATCH 2: click edge preko glfwGetMouseButton
bool mouseDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
bool mouseEdge = mouseDown && !mousePrev;
mousePrev = mouseDown;

// Click interactions
if (mouseEdge) {
    float tHit = 0.0f;

    
   // Lamp toggle (only if basin not locked)
    if (!acLocked) {
        bool hitLamp = raySphere(ro, rd, lampPos, lampRadius * 2.2f, tHit); // 2.2x tolerancija

        // fallback: ako si blizu klime i gledaš je (grubo), klik bilo gde pali/gasi
        glm::vec3 toAC = acPos - ro;
        float distAC = glm::length(toAC);
        glm::vec3 toACn = (distAC > 0.0001f) ? (toAC / distAC) : glm::vec3(0, 0, -1);
        float facingAC = glm::dot(rd, toACn);

        bool nearAndFacing = (distAC < 6.0f) && (facingAC > 0.80f); // "olabavljeno" - možeš menjati

        if (hitLamp || nearAndFacing) {
            acOn = !acOn;
        }
    }


    // If locked: click basin to pick it up
    if (acLocked && basinState == BasinState::OnFloor) {
        glm::vec3 basinPickCenter = basinPos + glm::vec3(0, 0.15f, 0);
        if (raySphere(ro, rd, basinPickCenter, 0.6f, tHit)) {
            basinState = BasinState::InHands;
        }
    }
}

// Flap animation
float flapTarget = (acOn ? 1.0f : 0.0f);
if (flap < flapTarget) flap = std::min(flapTarget, flap + flapSpeed * dt);
if (flap > flapTarget) flap = std::max(flapTarget, flap - flapSpeed * dt);

// Temperature simulation
if (acOn) {
    float diff = (float)desiredTemp - measuredTemp;
    if (std::abs(diff) > 0.01f) {
        measuredTemp += clampf(diff, -tempRate * dt, tempRate * dt);
    }
}

// Water fill + droplets
if (acOn) {
    basinFillTimer += dt;
    dropletSpawnTimer += dt;

    if (dropletSpawnTimer >= 0.20f) {
        dropletSpawnTimer = 0.0f;
        Droplet d;
        d.pos = ventPos + glm::vec3((float)((rand() % 100) / 100.0 - 0.5) * 0.08f, 0.0f, 0.0f);
        d.vel = glm::vec3(0.0f, -1.4f, 0.0f);
        d.life = 4.0f;
        droplets.push_back(d);
    }
    if (basinFillTimer >= 1.0f) {
        basinFillTimer = 0.0f;
        basinFill = std::min(1.0f, basinFill + 0.12f);
    }
}

// Compute basin bottom world Y (used for droplet collision/removal)
const float basinScaleY = 0.55f; // same scale used when rendering the basin
float basinBottomY;
if (basinState == BasinState::OnFloor) {
    basinBottomY = basinPos.y + WATER_SURFACE_MIN_LOCAL * basinScaleY;
} else {
    glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));
    glm::vec3 up = glm::normalize(cameraUp);
    glm::vec3 handPos = cameraPos + cameraFront * 1.0f - right * 0.35f - up * 0.30f;
    basinBottomY = handPos.y + WATER_SURFACE_MIN_LOCAL * basinScaleY;
}

const float dropletRadius = 0.05f; // same scale used when rendering droplets (scale 0.05)
// Droplet physics
for (auto& d : droplets) {
    d.vel += glm::vec3(0.0f, -3.0f, 0.0f) * dt;
    d.pos += d.vel * dt;
    d.life -= dt;

    // clamp to just above basin bottom and mark dead to avoid visual overlap
    if (d.pos.y < basinBottomY + dropletRadius) {
        d.pos.y = basinBottomY + dropletRadius;
        d.life = 0.0f; // will be removed below
    }
}

// Remove expired droplets (including those that hit the bottom)
droplets.erase(std::remove_if(droplets.begin(), droplets.end(), [&](const Droplet& d) {
    return d.life <= 0.0f;
    }), droplets.end());

// If basin full -> auto off + lock
if (!acLocked && basinFill >= 1.0f) {
    acOn = false;
    acLocked = true;
    basinEmptied = false;
    basinState = BasinState::OnFloor;
}

// Basin special actions when locked
if (acLocked && basinState == BasinState::InHands && spaceEdge) {
    // PATCH 4: yaw-only facing (XZ)
    glm::vec3 f = glm::normalize(glm::vec3(cameraFront.x, 0.0f, cameraFront.z));
    if (glm::length(f) < 0.0001f) f = glm::vec3(0, 0, -1);
    glm::vec3 to = glm::normalize(glm::vec3(acPos.x - cameraPos.x, 0.0f, acPos.z - cameraPos.z));
    if (glm::length(to) < 0.0001f) to = glm::vec3(0, 0, -1);

    float facing = glm::dot(f, to);
    // facing < -0.8 => okrenut "iza" (oko 180°)
    if (!basinEmptied && facing < -0.8f) {
        basinFill = 0.0f;
        basinEmptied = true;
    }
    // facing > 0.8 => gledamo ka klimi, možemo da vratimo lavor
    else if (basinEmptied && facing > 0.8f) {
        basinState = BasinState::OnFloor;
        basinEmptied = false;
        acLocked = false;
    }
}

// Render
glViewport(0, 0, wWidth, wHeight);
glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

glm::mat4 V = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
glm::mat4 P = glm::perspective(glm::radians(fov), (float)wWidth / (float)wHeight, 0.1f, 100.0f);
glm::mat4 VP = P * V;

setCommonUniforms(VP);

// Lamp as a weak local light (red) when AC is ON.
// Keep lamp sphere emissive as before, but also provide its contribution to shading.
{
    glm::vec3 lampLightColor = acOn ? glm::vec3(1.0f, 0.25f, 0.25f) : glm::vec3(0.0f);
    float lampLightIntensity = acOn ? 0.35f : 0.0f; // "slabo" red light when on

    glUniform3fv(glGetUniformLocation(prog, "uLampPos"), 1, glm::value_ptr(lampPos));
    glUniform3fv(glGetUniformLocation(prog, "uLampColor"), 1, glm::value_ptr(lampLightColor));
    glUniform1f(glGetUniformLocation(prog, "uLampIntensity"), lampLightIntensity);
}

// Compute basinM here once so transparent pass can access it too
glm::mat4 basinM(1.0f);
if (basinState == BasinState::OnFloor) {
    basinM = glm::translate(basinM, basinPos);
}
else {
    // in hands, in front of camera
    glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));
    glm::vec3 up = glm::normalize(cameraUp);
    glm::vec3 handPos = cameraPos + cameraFront * 1.0f - right * 0.35f - up * 0.30f;
    basinM = glm::translate(basinM, handPos);
    basinM = glm::rotate(basinM, glm::radians(-10.0f), right);
}
basinM = glm::scale(basinM, glm::vec3(0.75f, 0.55f, 0.75f));

// OPAQUE PASS (all opaque objects / emissive but depth-writable)
// Floor
if (gShowFloor) {
    glm::mat4 M(1.0f);
    M = glm::translate(M, glm::vec3(0.0f, -0.01f, 0.0f));
    M = glm::scale(M, glm::vec3(12.0f, 1.0f, 12.0f));
    drawMesh(quad, M, VP, { 0.18f,0.18f,0.2f,1 }, false, 0, 1.0f, false);
}

// AC body
{
    glm::mat4 M(1.0f);
    M = glm::translate(M, acPos);
    M = glm::scale(M, glm::vec3(1.6f, 0.55f, 1.0f));
    drawMesh(cube, M, VP, { 0.92f,0.92f,0.92f,1 }, false, 0, 1.0f, false);
}

// AC front panel (slightly darker)
{
    glm::mat4 M(1.0f);
    M = glm::translate(M, acPos + glm::vec3(0, 0, 0.55f));
    M = glm::scale(M, glm::vec3(1.55f, 0.50f, 0.04f));
    drawMesh(cube, M, VP, { 0.86f,0.86f,0.88f,1 }, false, 0, 1.0f, false);
}

// Flap (hinge at bottom of front)
{
    float angle = glm::radians(65.0f * flap);
    glm::mat4 M(1.0f);
    glm::vec3 hinge = acPos + glm::vec3(0.0f, -0.28f, 0.53f);
    M = glm::translate(M, hinge);
    M = glm::rotate(M, angle, glm::vec3(1, 0, 0));
    M = glm::translate(M, glm::vec3(0.0f, -0.05f, 0.05f));
    M = glm::scale(M, glm::vec3(1.3f, 0.10f, 0.25f));
    drawMesh(cube, M, VP, { 0.78f,0.78f,0.80f,1 }, false, 0, 1.0f, false);
}

// Lamp (emissive sphere)
{
    glm::mat4 M(1.0f);
    M = glm::translate(M, lampPos);
    M = glm::scale(M, glm::vec3(lampRadius));

    glm::vec4 c = acOn
        ? glm::vec4(1.0f, 0.25f, 0.25f, 1.0f)     // ON
        : glm::vec4(0.35f, 0.35f, 0.35f, 1.0f);   // OFF

    drawMesh(lampSphere, M, VP, c, false, 0, 1.0f, true);
}


        // Screens (three emissive panels)
        bool screensOn = acOn && !acLocked;
        glm::vec4 screenBase = screensOn ? glm::vec4(0.05f, 0.05f, 0.05f, 1) : glm::vec4(0.02f, 0.02f, 0.02f, 1);
        for (int i = 0; i < 3; i++) {
            glm::mat4 M(1.0f);
            M = glm::translate(M, acPos + glm::vec3(-0.45f + i * 0.45f, 0.10f, 0.57f));
            M = glm::scale(M, glm::vec3(0.32f, 0.18f, 0.02f));
            drawMesh(cube, M, VP, screenBase, false, 0, 1.0f, screensOn);
        }

        // 7-seg digits on first two screens when on (emissive shapes; still depth-writable)
        if (screensOn) {
            glm::vec4 onCol(0.15f, 1.0f, 0.35f, 1.0f);
            glm::vec4 offCol(0.03f, 0.12f, 0.05f, 1.0f);

            // Desired temp (two digits + sign for negative)
            int dtemp = desiredTemp;
            int absd = std::abs(dtemp);
            int tens = absd / 10;
            int ones = absd % 10;
            float z = acPos.z + 0.585f;
            float y = acPos.y + 0.10f;
            float x0 = acPos.x - 0.45f;

            // minus sign if needed
            if (dtemp < 0) {
                glm::mat4 M(1.0f);
                M = glm::translate(M, glm::vec3(x0 - 0.12f, y, z));
                M = glm::scale(M, glm::vec3(0.09f, 0.02f, 1));
                drawMesh(quad, M, VP, onCol, false, 0, 1.0f, true);
            }

            drawSegDigit(prog, quad, VP, { {x0 - 0.03f, y, z}, 0.13f, 0.16f }, tens, onCol, offCol);
            drawSegDigit(prog, quad, VP, { {x0 + 0.10f, y, z}, 0.13f, 0.16f }, ones, onCol, offCol);

            // Measured temp
            int mtemp = (int)std::round(measuredTemp);
            int absm = std::abs(mtemp);
            int mt = absm / 10;
            int mo = absm % 10;
            float x1 = acPos.x;
            if (mtemp < 0) {
                glm::mat4 M(1.0f);
                M = glm::translate(M, glm::vec3(x1 - 0.12f, y, z));
                M = glm::scale(M, glm::vec3(0.09f, 0.02f, 1));
                drawMesh(quad, M, VP, onCol, false, 0, 1.0f, true);
            }
            drawSegDigit(prog, quad, VP, { {x1 - 0.03f, y, z}, 0.13f, 0.16f }, mt, onCol, offCol);
            drawSegDigit(prog, quad, VP, { {x1 + 0.10f, y, z}, 0.13f, 0.16f }, mo, onCol, offCol);

            // Icon on third screen: flame / snow / check (simple shapes)
            float x2 = acPos.x + 0.45f;
            float iconZ = z;
            float iconScale = 0.75f; // 20% manje

            glm::vec4 iconCol(0.9f, 0.3f, 0.1f, 1);
            if (desiredTemp < (int)std::round(measuredTemp)) {
                iconCol = glm::vec4(0.3f, 0.7f, 1.0f, 1); // snow
                // snowflake: two crossing rectangles
                glm::mat4 M1(1.0f);
                M1 = glm::translate(M1, glm::vec3(x2, y, iconZ));
                M1 = glm::scale(M1, glm::vec3(0.16f, 0.02f, 1));
                drawMesh(quad, M1, VP, iconCol, false, 0, 1.0f, true);
                glm::mat4 M2(1.0f);
                M2 = glm::translate(M2, glm::vec3(x2, y, iconZ));
                M2 = glm::rotate(M2, glm::radians(60.0f), glm::vec3(0, 0, 1));
                M2 = glm::scale(M2, glm::vec3(0.16f, 0.02f, 1));
                drawMesh(quad, M2, VP, iconCol, false, 0, 1.0f, true);
                glm::mat4 M3(1.0f);
                M3 = glm::translate(M3, glm::vec3(x2, y, iconZ));
                M3 = glm::rotate(M3, glm::radians(-60.0f), glm::vec3(0, 0, 1));
                M3 = glm::scale(M3, glm::vec3(0.16f, 0.02f, 1));
                drawMesh(quad, M3, VP, iconCol, false, 0, 1.0f, true);
            }
            
            else if (desiredTemp > (int)std::round(measuredTemp)) {
                glm::vec4 flameCol(1.0f, 0.45f, 0.10f, 1.0f);

                // Donji deo
                {
                    glm::mat4 M(1.0f);
                    M = glm::translate(M, glm::vec3(x2, y - 0.02f * iconScale, iconZ));
                    M = glm::scale(M, glm::vec3(0.12f * iconScale, 0.10f * iconScale, 1));
                    drawMesh(quad, M, VP, flameCol, false, 0, 1.0f, true);
                }

                // Srednji deo
                {
                    glm::mat4 M(1.0f);
                    M = glm::translate(M, glm::vec3(x2, y + 0.03f * iconScale, iconZ));
                    M = glm::scale(M, glm::vec3(0.08f * iconScale, 0.10f * iconScale, 1));
                    drawMesh(quad, M, VP, flameCol, false, 0, 1.0f, true);
                }

                // Vrh
                {
                    glm::mat4 M(1.0f);
                    M = glm::translate(M, glm::vec3(x2, y + 0.08f * iconScale, iconZ));
                    M = glm::rotate(M, glm::radians(45.0f), glm::vec3(0, 0, 1));
                    M = glm::scale(M, glm::vec3(0.05f * iconScale, 0.05f * iconScale, 1));
                    drawMesh(quad, M, VP, flameCol, false, 0, 1.0f, true);
                }
            }


            else {
                glm::vec4 okCol(0.20f, 1.0f, 0.35f, 1.0f);

                // draw bars without forcing culling state
                auto drawBar = [&](glm::vec3 a, glm::vec3 b, float thick, float depth) {
                    glm::vec3 d = b - a;
                    float len = glm::length(glm::vec2(d.x, d.y));
                    if (len < 1e-5f) return;

                    glm::vec3 mid = (a + b) * 0.5f;
                    float ang = std::atan2(d.y, d.x); // rotacija u XY (panel ravn)

                    glm::mat4 M(1.0f);
                    M = glm::translate(M, mid);
                    M = glm::rotate(M, ang, glm::vec3(0, 0, 1));
                    // cube je 1x1x1, pa ga skaliramo u štapić
                    M = glm::scale(M, glm::vec3(len, thick, depth));

                    drawMesh(cube, M, VP, okCol, false, 0, 1.0f, true); // emissive
                    };

                float thick = 0.060f * iconScale;
                float depth = 0.020f * iconScale;

                glm::vec3 p0(x2 - 0.12f * iconScale, y - 0.00f * iconScale, iconZ + 0.010f);
                glm::vec3 p1(x2 - 0.02f * iconScale, y - 0.09f * iconScale, iconZ + 0.010f);
                glm::vec3 p2(x2 + 0.14f * iconScale, y + 0.08f * iconScale, iconZ + 0.010f);


                drawBar(p0, p1, thick, depth); // kratki potez
                drawBar(p1, p2, thick, depth); // dugi potez
            }
        }

        // Basin (thick cylinder) - either on floor or in hands (opaque geometry - walls & bottom)
        {
            if (gShowBasinWall) {
                // Basin needs two-sided rendering (inner + outer walls) ONLY when depth test is enabled.
                // When depth test is OFF, X toggle should work normally on basin too.
                GLboolean cullWas = glIsEnabled(GL_CULL_FACE);
                if (gDepthTest && cullWas) {
                    // Hack: disable culling for basin only when Z is ON
                    glDisable(GL_CULL_FACE);
                }
                
                drawMesh(basinWall, basinM, VP, { 0.88f,0.88f,0.90f,1 }, false, 0, 1.0f, false);
                
                // Restore culling state (only if we disabled it)
                if (gDepthTest && cullWas) {
                    glEnable(GL_CULL_FACE);
                }
            }
            // bottom always drawn with normal culling
            drawMesh(basinBottom, basinM, VP, { 0.75f,0.75f,0.75f,1 }, false, 0, 1.0f, false);
        }

        // Toilet behind (simple block model)
        {
            glm::mat4 M(1.0f);
            M = glm::translate(M, toiletPos);
            M = glm::scale(M, glm::vec3(0.9f, 0.5f, 1.2f));
            drawMesh(cube, M, VP, { 0.92f,0.92f,0.94f,1 }, false, 0, 1.0f, false);
            glm::mat4 M2(1.0f);
            M2 = glm::translate(M2, toiletPos + glm::vec3(0, 0.55f, -0.25f));
            M2 = glm::scale(M2, glm::vec3(0.6f, 0.6f, 0.6f));
            drawMesh(cube, M2, VP, { 0.92f,0.92f,0.94f,1 }, false, 0, 1.0f, false);
        }

        // Remote model - prikazan SAMO kada lavor NIJE u rukama, UVEK OKRENUT KA KLIMI
        if (basinState != BasinState::InHands) {
            glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));
            glm::vec3 up = glm::normalize(cameraUp);
            glm::vec3 pos = cameraPos + cameraFront * 0.9f + right * 0.35f - up * 0.35f;
            
            // Izračunaj vektor od daljinskog ka klimi
            glm::vec3 toAC = acPos - pos;
            float distToAC = glm::length(toAC);
            if (distToAC > 0.0001f) {
                toAC /= distToAC; // normalize
                
                // Izračunaj yaw i pitch iz vektora toAC
                float remoteYaw = std::atan2(toAC.z, toAC.x) - glm::radians(90.0f); // -90° jer model gleda duž +Y
                float remoteHorizDist = std::sqrt(toAC.x * toAC.x + toAC.z * toAC.z);
                float remotePitch = std::atan2(toAC.y, remoteHorizDist);
                
                glm::mat4 M(1.0f);
                M = glm::translate(M, pos);
                // Prvo rotiraj oko Y (yaw)
                M = glm::rotate(M, remoteYaw, glm::vec3(0, 1, 0));
                // Zatim rotiraj oko lokalne X ose (pitch)
                M = glm::rotate(M, remotePitch, glm::vec3(1, 0, 0));
                M = glm::scale(M, glm::vec3(0.18f, 0.42f, 0.06f));
                drawMesh(cube, M, VP, { 1,1,1,1 }, true, texRemote, 1.0f, false);
            }
        }

        // TRANSPARENT PASS (draw after opaque geometry)
        {
            // Save culling state
            GLboolean cullWas = glIsEnabled(GL_CULL_FACE);
            // Ensure we do not write depth for transparent objects
            glDepthMask(GL_FALSE);

            // Water inside basin (transparent) - draw here after opaque geometry
            if (basinFill > 0.001f) {

                // clamp water surface inside basin (local coords)
                float surfaceLocal = WATER_SURFACE_MIN_LOCAL + (0.9f * basinFill);
                surfaceLocal = std::min(surfaceLocal, WATER_SURFACE_MAX_LOCAL);

                // derived height (0..)
                float h = std::max(0.0f, surfaceLocal - WATER_SURFACE_MIN_LOCAL);

                // side volume
                glm::mat4 MW = basinM;
                MW = glm::translate(MW, glm::vec3(0, WATER_SURFACE_MIN_LOCAL + h * 0.5f, 0));
                MW = glm::scale(MW, glm::vec3(1, h, 1));
                drawMesh(
                    waterCyl, MW, VP,
                    { 0.35f,0.75f,0.95f,0.55f },
                    false, 0, 0.55f, false
                );

                // top disk at EXACT surfaceLocal
                glm::mat4 MT = basinM;
                MT = glm::translate(MT, glm::vec3(0, surfaceLocal, 0));
                drawMesh(
                    waterTop, MT, VP,
                    { 0.35f,0.75f,0.95f,0.55f },
                    false, 0, 0.55f, false
                );
            }

            // Droplets (transparent) - sort back-to-front and draw
            if (!droplets.empty()) {
                std::vector<int> order(droplets.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    glm::vec3 da = cameraPos - droplets[a].pos;
                    glm::vec3 db = cameraPos - droplets[b].pos;
                    float da2 = glm::dot(da, da);
                    float db2 = glm::dot(db, db);
                    return da2 > db2; // farthest first
                });

                for (int idx : order) {
                    auto& d = droplets[idx];
                    glm::mat4 M(1.0f);
                    M = glm::translate(M, d.pos);
                    M = glm::scale(M, glm::vec3(0.05f));
                    drawMesh(dropletSphere, M, VP, { 0.35f,0.75f,0.95f,0.45f }, false, 0, 0.45f, false);
                }
            }

            // restore depth write
            glDepthMask(GL_TRUE);
            // restore culling state
            if (!cullWas) glDisable(GL_CULL_FACE);
            else glEnable(GL_CULL_FACE);
        }

        // Overlay (student name/index) - screen space (bigger nametag)
        {
            glDisable(GL_DEPTH_TEST);
            glm::mat4 ortho = glm::ortho(0.0f, (float)wWidth, 0.0f, (float)wHeight);
            glm::mat4 VP2 = ortho;
            glm::mat4 M(1.0f);

            // Scale factor for nametag size (tweak this value to make it larger/smaller)
            const float nameTagScale = 1.6f; // <-- increase to make nametag bigger

            // Keep the original top-left corner of the nametag fixed while scaling.
            // Original values: center (170, h-70), size (300,90).
            // Compute base left/top in screen space and derive new center from scaled size.
            const float baseLeft = 170.0f - 300.0f * 0.5f;         // = 20.0f
            const float baseTop  = (float)wHeight - 25.0f;         // original top = h - 25.0f

            float sizeX = 300.0f * nameTagScale;
            float sizeY =  90.0f * nameTagScale;

            float centerX = baseLeft + sizeX * 0.5f;
            float centerY = baseTop  - sizeY * 0.5f;

            M = glm::translate(M, glm::vec3(centerX, centerY, 0.0f));
            M = glm::scale(M, glm::vec3(sizeX, sizeY, 1.0f));

            drawMesh(quad, M, VP2, { 2,2,2,2 }, true, texStudent, 0.7f, false);
            if (gDepthTest) glEnable(GL_DEPTH_TEST);
        }

        glfwSwapBuffers(window);

        // Frame limiter to 75 FPS
        auto frameEnd = clock::now();
        double elapsed = std::chrono::duration<double>(frameEnd - frameStart).count();
        if (elapsed < targetFrame) {
            std::this_thread::sleep_for(std::chrono::duration<double>(targetFrame - elapsed));
        }
    }
    
    // Cleanup
    destroyMesh(cube);
    destroyMesh(quad);
    destroyMesh(lampSphere);
    destroyMesh(dropletSphere);
    //destroyMesh(basinWallOuter);
    destroyMesh(basinWall);
    destroyMesh(basinBottom);
    destroyMesh(waterCyl);
    destroyMesh(waterTop);

    glDeleteProgram(prog);
    glfwTerminate();
    return 0;
}
