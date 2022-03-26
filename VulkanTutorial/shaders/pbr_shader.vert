#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    
    vec4 cameraPos;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragWorldPos;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragCameraPos;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    
    fragTexCoord = inTexCoord;
    fragWorldPos = vec3(ubo.model * vec4(inPosition, 1.0f));
    fragNormal = vec3(ubo.model * vec4(inNormal, 1.0f));
    fragCameraPos = vec3(ubo.cameraPos);
}
