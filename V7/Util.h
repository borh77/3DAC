#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
unsigned int createShader(const char* vsSource, const char* fsSource);
unsigned loadImageToTexture(const char* filePath);
GLFWcursor* loadImageToCursor(const char* filePath);