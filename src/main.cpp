#if defined(__APPLE__)
#   define GL_SILENCE_DEPRECATION
#   include <OpenGL/gl3.h>
#endif

#include <GLFW/glfw3.h>
#include <stdio.h>

#include <deque>
#include <vector>
#include <map>
#include <random>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define FILE_NAME __FILE_NAME__
#define LOG_ERROR(msg, ...) fprintf(stderr, "(error:%s:%d) " msg "\n", FILE_NAME, __LINE__ ,##__VA_ARGS__)
#define GL_CHECK(call) \
do \
{ \
    call; \
    GLenum result = glGetError(); \
    static bool first = true; \
    if (first && result != GL_NO_ERROR) \
    { \
        first = false; \
        LOG_ERROR("%d", result); \
    } \
} while (0) 

struct Vertex 
{
    glm::vec3 position;
    glm::vec3 normal;
};

const char *vertexSource = R"(
#version 330 core

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;

out vec3 v_normal;

uniform mat4 u_modelToWorld;
uniform mat4 u_worldToClip;

void main(void) 
{
    v_normal = transpose(inverse(mat3(u_modelToWorld))) * a_normal;
    gl_Position = u_worldToClip * u_modelToWorld * vec4(a_position, 1.0f);
}

)";

const char *fragmentSource = R"( 
#version 330 core

in vec3 v_normal;

out vec4 FragColor;

#define PI 3.141592

void main(void)
{
    vec3 baseColor = vec3(1.0f);
    vec3 sunDirection = vec3(1.0f, 1.0f, 0.0f);
    vec3 sunColor = vec3(1.0f);

    vec3 L = normalize(sunDirection);
    vec3 N = normalize(v_normal);

    vec3 diffuse = (baseColor/PI) * (sunColor * max(dot(L, N), 0.0f));
    vec3 ambient = baseColor * 0.01f;

    FragColor = vec4(diffuse + ambient, 1.0f);
}

)";

static uint32_t compileShader(GLenum type, const char *source) 
{
    uint32_t shader;
    GL_CHECK(shader = glCreateShader(type));
    GL_CHECK(glShaderSource(shader, 1, &source, nullptr));
    GL_CHECK(glCompileShader(shader));

    char infoLog[1024];
    int infoLogLength = 0;
    glGetShaderInfoLog(shader, sizeof(infoLog), &infoLogLength, infoLog);
    if (infoLogLength > 0) 
    {
        infoLog[infoLogLength - 1] = 0;
        LOG_ERROR("%s", infoLog);
    }

    return shader;
}

static uint32_t compileProgram(const char *vertexSource, const char *fragmentSource) 
{
    uint32_t program;
    GL_CHECK(program = glCreateProgram());

    uint32_t vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    uint32_t fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GL_CHECK(glAttachShader(program, vertexShader));
    GL_CHECK(glAttachShader(program, fragmentShader));
    GL_CHECK(glLinkProgram(program));

    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    return program;
}

static int32_t getUniformLocation(uint32_t program, const char *uname) 
{
    int32_t location;
    GL_CHECK(location = glGetUniformLocation(program, uname));
    if (location == -1) 
        LOG_ERROR("Location of uniform %s was null", uname);

    return location;
}

static void setUniform(uint32_t program, const char *uname, const glm::mat4 &value) 
{
    int32_t location = getUniformLocation(program, uname);
    if (location == -1) 
        return;

    GL_CHECK(glUniformMatrix4fv(location, 1, GL_FALSE, &value[0][0]));
}

static std::random_device rd;
static std::mt19937 gen(rd());
static std::normal_distribution<float> d(0.5, 0.17);
// static std::uniform_real_distribution<float> d(0.0f, 1.0f);

static float randomNormal(void)
{
    // return (float)rand()/RAND_MAX;
    return d(gen);
}

static float randomInRange(float min, float max)
{
    return min + (max-min) * randomNormal();
}

struct Mesh 
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

static Mesh generateMesh(glm::vec2 min, glm::vec2 max, size_t maxDepth)
{
    std::vector<glm::vec3> positions;
    std::deque<uint32_t> q;

    positions.push_back({ min.x, 0.0f, min.y });
    positions.push_back({ min.x, 0.0f, max.y });
    positions.push_back({ max.x, 0.0f, max.y });
    positions.push_back({ max.x, 0.0f, min.y });

    // For the algorithm it is more convenient to work with quads,
    // will convert to triangles later.

    q.push_back(0);
    q.push_back(1);
    q.push_back(2);
    q.push_back(3);

    // If I work on indices instead of positions, then I don't get gaps, ie 
    // v0, v1, v2, v3
    // 0, 1, 2, 3
    // 
    // During one iteration also apply a random offset either to all the positions or just the new one.
    // v0, v1, v2, v3, center, v01, v12, v23, v30
    // 0, 5, 4, 1,
    // 1, 6, 4, 2,
    // 2, 7, 4, 3,
    // 3, 8, 4, 0,
    //
    // Since literally the same positions are being references, gaps are not possible.

    float yChangeFactor = 1.0f;

    size_t depth = maxDepth;
    while (depth--)
    {
        // Working in terms of quads. So 4 points per.
        size_t n = q.size()/4;
        while (n--) 
        {
            const auto v0 = q.front(); 
            q.pop_front();
            const auto v1 = q.front(); 
            q.pop_front();
            const auto v2 = q.front(); 
            q.pop_front();
            const auto v3 = q.front(); 
            q.pop_front();

            positions.push_back(0.25f * (positions[v0] + positions[v1] + positions[v2] + positions[v3]));
            const auto center = positions.size() - 1;

            positions.push_back(0.5f * (positions[v0] + positions[v1]));
            const auto v01 = positions.size() - 1;

            positions.push_back(0.5f * (positions[v1] + positions[v2]));
            const auto v12  = positions.size() - 1;

            positions.push_back(0.5f * (positions[v2] + positions[v3]));
            const auto v23 = positions.size() - 1;

            positions.push_back(0.5f * (positions[v3] + positions[v0]));
            const auto v30  = positions.size() - 1;

            //
            // v1-------v2
            // |    |    |
            // |----+----|
            // |    |    |
            // v0-------v3
            //
            // As can be seen from the picture above, the center changes from being or not being 
            // the higher than the vertex  it is compared to, so for some sub-quads the order needs 
            // to be inverted to achieve the same winding on all positions in the mesh.

            q.push_back(v0);
            q.push_back(v30);
            q.push_back(center);
            q.push_back(v01);

            q.push_back(v1);
            q.push_back(v01);
            q.push_back(center);
            q.push_back(v12);

            q.push_back(v2);
            q.push_back(v12);
            q.push_back(center);
            q.push_back(v23);

            q.push_back(v3);
            q.push_back(v23);
            q.push_back(center);
            q.push_back(v30);

            positions[center].x += randomInRange(-yChangeFactor, yChangeFactor);
            positions[center].y += randomInRange(-yChangeFactor, yChangeFactor);
            positions[center].z += randomInRange(-yChangeFactor, yChangeFactor);
        }

        yChangeFactor *= 0.5f;
    }

    // HACK: Should just be generated in a way where this is not a problem rather than fixed post 
    // fact with a map.
    const auto comp = [](const glm::vec3 &a, const glm::vec3 &b) -> bool
    { 
        if (a.x == b.x)
            return a.z < b.z;
        return a.x < b.x;
    };

    std::map<glm::vec3, uint32_t, decltype(comp)> seen(comp);
    for (auto &index : q) 
    {
        if (seen.find(positions[index]) == seen.end())
        {
            seen[positions[index]] = index;
        }
        else 
        {
            index = seen[positions[index]];
        }
    }

    size_t n = q.size()/4; 
    while (n--) 
    {
        const auto v0 = q.front(); 
        q.pop_front();
        const auto v1 = q.front(); 
        q.pop_front();
        const auto v2 = q.front(); 
        q.pop_front();
        const auto v3 = q.front(); 
        q.pop_front();

        // The winding order seems to change every depth iteration, so this should push back 
        // in clockwise for odd and counterclockwise for even.

        if (maxDepth & 1) 
        {
            q.push_back(v0);
            q.push_back(v3);
            q.push_back(v2);

            q.push_back(v2);
            q.push_back(v1);
            q.push_back(v0);
        }
        else 
        {
            q.push_back(v0);
            q.push_back(v1);
            q.push_back(v2);

            q.push_back(v2);
            q.push_back(v3);
            q.push_back(v0);
        }
        
    }

    Mesh mesh = {};
    for (const auto position : positions) 
        mesh.vertices.push_back({ position, glm::vec3(0) });
    mesh.indices = { q.begin(), q.end() };

    return mesh;
}

static void generateHarshNormals(Mesh &mesh) 
{
    std::vector<Vertex> vertices;
    for (const auto index : mesh.indices) 
        vertices.push_back(mesh.vertices[index]);

    const size_t triangleCount =  vertices.size()/3;
    for (size_t i = 0; i < triangleCount; ++i)
    {
        const auto p0 = vertices[i*3+0].position;
        const auto p1 = vertices[i*3+1].position;
        const auto p2 = vertices[i*3+2].position;

        const auto u = p1 - p0;
        const auto v = p2 - p0;

        const auto normal = glm::normalize(glm::cross(u, v));

        vertices[i*3+0].normal = normal;
        vertices[i*3+1].normal = normal;
        vertices[i*3+2].normal = normal;
    }

    mesh.indices.clear();
    for (uint32_t i = 0; i < vertices.size(); ++i) 
        mesh.indices.push_back(i);
    mesh.vertices = vertices;
}

static void generateSmoothNormals(Mesh &mesh) 
{
    const size_t triangleCount =  mesh.indices.size()/3;
    for (size_t i = 0; i < triangleCount; ++i)
    {
        const auto p0 = mesh.vertices[mesh.indices[i*3+0]].position;
        const auto p1 = mesh.vertices[mesh.indices[i*3+1]].position;
        const auto p2 = mesh.vertices[mesh.indices[i*3+2]].position;

        const auto u = p1 - p0;
        const auto v = p2 - p0;

        // const auto triangleSize = glm::length(u) * glm::length(v) / 2.0f;
        const auto triangleSize = 1.0f; // This would be a good idea in general, 
                                        // but with this particular terrain algo all the triangles have 
                                        // exactly the same size and this does nothing except waste cycles.

        const auto normal = glm::normalize(glm::cross(u, v));

        mesh.vertices[mesh.indices[i*3+0]].normal += triangleSize * normal;
        mesh.vertices[mesh.indices[i*3+1]].normal += triangleSize * normal;
        mesh.vertices[mesh.indices[i*3+2]].normal += triangleSize * normal;
    }

    for (auto &vertex  : mesh.vertices)
        vertex.normal = glm::normalize(vertex.normal);
}

int main(void) 
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow *window = glfwCreateWindow(800, 600, "", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    uint32_t program = compileProgram(vertexSource, fragmentSource);

    uint32_t vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    float meshScale = 1.0f;
    auto mesh = generateMesh({ -meshScale, -meshScale }, { meshScale, meshScale }, 11);
    generateSmoothNormals(mesh);

    const auto *vertices = mesh.vertices.data();
    const size_t vertexCount = mesh.vertices.size();

    const auto *indices = mesh.indices.data();
    const size_t indexCount = mesh.indices.size();

    printf("vertex count %zu\n", vertexCount);
    printf("index count %zu\n", indexCount);

    // const Vertex vertices[] = {
    //     { glm::vec3(-0.5f, -0.5f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f) },
    //     { glm::vec3( 0.5f, -0.5f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f) },
    //     { glm::vec3( 0.5f,  0.5f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f) },
    // };

    uint32_t vertexBuffer;
    glGenBuffers(1, &vertexBuffer);
    {
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*vertexCount, vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    uint32_t indexBuffer;
    glGenBuffers(1, &indexBuffer);
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t)*indexCount, indices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    float yaw = -M_PI_2;
    float pitch = 0.0f;

    double lastTime = glfwGetTime();
    glm::vec3 cameraPosition(0, 2, 4);
    glm::vec3 cameraUp(0, 1, 0);

    while (!glfwWindowShouldClose(window)) 
    {
        glfwPollEvents();

        double now = glfwGetTime();
        double dt = now - lastTime;
        lastTime = now;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glm::vec3 movement(0, 0, 0);

        if (glfwGetKey(window, GLFW_KEY_W))
            movement.z += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_S))
            movement.z -= 1.0f;
        if (glfwGetKey(window, GLFW_KEY_A))
            movement.x -= 1.0f;
        if (glfwGetKey(window, GLFW_KEY_D))
            movement.x += 1.0f;

        float lookSpeed = 1.0f * dt;

        glm::vec3 front = glm::normalize(glm::vec3(
            glm::cos(yaw) * glm::cos(pitch),
            glm::sin(pitch),
            glm::sin(yaw) * glm::cos(pitch)));
        glm::vec3 right = glm::normalize(glm::cross(front, cameraUp));
        glm::vec3 up = glm::cross(right, front);

        movement *= dt;
        cameraPosition += front*movement.z + right*movement.x + up*movement.y;

        if (glfwGetKey(window, GLFW_KEY_UP))
            pitch += lookSpeed;
        if (glfwGetKey(window, GLFW_KEY_DOWN))
            pitch -= lookSpeed;
        if (glfwGetKey(window, GLFW_KEY_LEFT))
            yaw -= lookSpeed;
        if (glfwGetKey(window, GLFW_KEY_RIGHT))
            yaw += lookSpeed;

        glm::mat4 P = glm::perspective((float)M_PI_4, (float)width/height, 0.1f, 1000.0f);
        glm::mat4 V = glm::lookAt(cameraPosition, cameraPosition + front, cameraUp);
        glm::mat4 M = glm::mat4(1);
        M[3][3] = 0.5f;

        glm::mat4 modelToWorld = M;
        glm::mat4 worldToClip = P * V;

        {
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_FRAMEBUFFER_SRGB);
            glEnable(GL_CULL_FACE);
            glClearColor(0.01f, 0.02f, 0.03f, 1.0f);

            GL_CHECK(glUseProgram(program));
            setUniform(program, "u_modelToWorld", modelToWorld);
            setUniform(program, "u_worldToClip", worldToClip);
            
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer));

            GL_CHECK(glEnableVertexAttribArray(0));
            GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void *)offsetof(Vertex, position)));
            GL_CHECK(glEnableVertexAttribArray(1));
            GL_CHECK(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void *)offsetof(Vertex, normal)));

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            GL_CHECK(glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, (const void *)0));

            GL_CHECK(glDisableVertexAttribArray(0));
            GL_CHECK(glDisableVertexAttribArray(1));

            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

            GL_CHECK(glUseProgram(0));

            glDisable(GL_FRAMEBUFFER_SRGB);
            glDisable(GL_CULL_FACE);
        }

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
