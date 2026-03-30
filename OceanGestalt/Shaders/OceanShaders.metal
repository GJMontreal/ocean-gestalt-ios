#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

#include "../Renderer/ShaderTypes.h"

// Vertex input from mesh
struct VertexIn {
    float3 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 fragPos;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 fragUV;
    float3 color;
};

// ---- Gerstner wave math (ported from gerstner.vert) ----

constant float PI = 3.14159265358979323;
constant float GRAVITY = 9.81;
constant float VELOCITY_SCALE = 1.0;
constant float NORMAL_OFFSET = 0.01;

float3 waveOffset(float time, float3 aPosition, WaveUniform wave) {
    float safeWavelength = max(wave.wavelength, 0.01);
    float k = 2.0 * PI / safeWavelength;
    float w = VELOCITY_SCALE * sqrt(GRAVITY * k);
    float2 D = normalize(wave.direction);

    float phase = dot(D * k, aPosition.xz) - fmod(w * time, 2.0 * PI) + wave.phase;

    float S = sin(phase);
    float C = cos(phase);

    float y = wave.amplitude * C;
    float2 xz = -wave.steepness * D * S * wave.amplitude;

    return float3(xz.x, y, xz.y);
}

float3 calcNewPosition(float3 aPosition, constant SceneUniforms& uniforms) {
    float3 offset = float3(0.0);
    for (int i = 0; i < uniforms.numWaves; i++) {
        offset += waveOffset(uniforms.time, aPosition, uniforms.waves[i]);
    }
    return aPosition + offset;
}

float3 calcNormal(float3 originalPosition,
                  float3 newPosition,
                  float offset,
                  constant SceneUniforms& uniforms,
                  thread float3& tangent,
                  thread float3& bitangent) {
    float3 posOffsetX = float3(originalPosition.x + offset, 0.0, originalPosition.z);
    float3 displacedX = calcNewPosition(posOffsetX, uniforms);

    float3 posOffsetZ = float3(originalPosition.x, 0.0, originalPosition.z + offset);
    float3 displacedZ = calcNewPosition(posOffsetZ, uniforms);

    tangent   = displacedX - newPosition;
    bitangent = displacedZ - newPosition;

    return normalize(cross(bitangent, tangent));
}

vertex VertexOut oceanVertex(VertexIn in [[stage_in]],
                             constant SceneUniforms& uniforms [[buffer(1)]]) {
    VertexOut out;

    float3 newPosition = calcNewPosition(in.position, uniforms);

    float3 tangent;
    float3 bitangent;
    float3 normal = calcNormal(in.position, newPosition, NORMAL_OFFSET, uniforms, tangent, bitangent);

    float4x4 mvp = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix;
    out.position  = mvp * float4(newPosition, 1.0);
    out.fragPos   = newPosition;
    out.normal    = normalize(normal);
    out.tangent   = tangent;
    out.bitangent = bitangent;
    out.fragUV    = in.texCoord;
    out.color     = float3(0.0, 0.04, 0.03);

    return out;
}

// ---- Fragment shader (ported from water.frag, no textures) ----

float hash2(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float valueNoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float a = hash2(fmod(i,               289.0));
    float b = hash2(fmod(i + float2(1.0, 0.0), 289.0));
    float c = hash2(fmod(i + float2(0.0, 1.0), 289.0));
    float d = hash2(fmod(i + float2(1.0, 1.0), 289.0));
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(float2 p) {
    float sum = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 5; ++i) {
        sum += amp * valueNoise(fmod(p * freq, 289.0));
        freq *= 2.0;
        amp  *= 0.5;
    }
    return sum;
}

float3 fresnelSchlick(float cosTheta, float f0) {
    return float3(f0) + (1.0 - float3(f0)) * pow(1.0 - cosTheta, 5.0);
}

fragment float4 oceanFragment(VertexOut in [[stage_in]],
                               constant SceneUniforms& uniforms [[buffer(1)]],
                               constant SurfaceUniforms& surface [[buffer(2)]]) {
    float3 normal   = normalize(in.normal);
    float3 viewDir  = normalize(uniforms.cameraPos - in.fragPos);
    float3 lightDir = normalize(surface.lightPos - in.fragPos);

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);

    // Fresnel
    float cosTheta = max(dot(viewDir, normal), 0.0);
    float3 fresnel = fresnelSchlick(cosTheta, surface.fresnelF0);

    // Approximate sky reflection
    float3 skyColor = float3(0.4, 0.6, 0.7);
    float3 reflection = fresnel * skyColor;

    // Diffuse color
    float3 baseCol = in.color;
    float3 diffuse = (1.0 - fresnel) * diff * baseCol;

    // Depth-based tint
    float depth = length(uniforms.cameraPos - in.fragPos);
    float depthFade = smoothstep(surface.depthFadeNear, surface.depthFadeFar, depth);
    float3 deepColor = baseCol * surface.deepWaterTint;
    float3 shiftedColor = mix(baseCol, deepColor, depthFade);
    shiftedColor = pow(shiftedColor, float3(surface.gamma));

    // Caustics
    float causticStrength = smoothstep(surface.causticTroughMin, surface.causticTroughMax, -in.fragPos.y);
    float2 flickerUV = in.fragPos.xz * surface.causticScale + uniforms.time * surface.causticSpeed;
    float causticFlicker = fbm(flickerUV);
    causticFlicker = smoothstep(surface.causticThresholdMin, surface.causticThresholdMax, causticFlicker);
    float NdotL = max(dot(normalize(in.normal), normalize(surface.lightPos - in.fragPos)), 0.0);
    causticFlicker *= NdotL;
    causticFlicker = pow(causticFlicker, surface.causticSharpness);
    float3 causticLight = surface.causticColor * causticFlicker * causticStrength * surface.causticIntensity;

    // Foam
    float2 dPos = float2(dfdx(in.fragPos.y), dfdy(in.fragPos.y));
    float slope = length(dPos) * surface.foamSlopeAmplifier;
    float2 slopeDir = normalize(dPos);
    float2 foamUV = in.fragPos.xz + slopeDir * uniforms.time * surface.foamScrollSpeed;
    float foamNoise = fbm(foamUV * surface.foamScale);
    float foamMask = smoothstep(surface.foamSlopeMin, surface.foamSlopeMax, slope);
    float foam = pow(foamNoise * foamMask, surface.foamPower);

    float3 finalColor = reflection + diffuse + causticLight;
    finalColor = mix(finalColor, float3(1.0), foam);

    // Fog
    float fogFactor = clamp(exp(-surface.fogDensity * depth), 0.0, 1.0);
    finalColor = mix(surface.fogColor, finalColor, fogFactor);

    return float4(finalColor, 1.0);
}

// ---- Buoy shaders ----

struct BuoyVertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

struct BuoyVertexOut {
    float4 position [[position]];
    float3 fragPos;
    float3 normal;
};

vertex BuoyVertexOut buoyVertex(BuoyVertexIn in [[stage_in]],
                                constant SceneUniforms& scene     [[buffer(1)]],
                                constant float4x4&      model     [[buffer(2)]]) {
    float4 worldPos    = model * float4(in.position, 1.0);
    float3 worldNormal = normalize((model * float4(in.normal, 0.0)).xyz);

    float4x4 vp = scene.projectionMatrix * scene.viewMatrix;
    BuoyVertexOut out;
    out.position = vp * worldPos;
    out.fragPos  = worldPos.xyz;
    out.normal   = worldNormal;
    return out;
}

fragment float4 buoyFragment(BuoyVertexOut       in      [[stage_in]],
                              constant SurfaceUniforms& surface [[buffer(2)]]) {
    float3 normal   = normalize(in.normal);
    float3 lightDir = normalize(surface.lightPos - in.fragPos);
    float  diff     = max(dot(normal, lightDir), 0.0);
    float3 color    = float3(0.85, 0.31, 0.0) * (0.25 + 0.75 * diff);
    return float4(color, 1.0);
}
