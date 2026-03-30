#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

#include "../Renderer/ShaderTypes.h"

// ---- Shared Gerstner wave math ----

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

float2 calcScrollUV(float3 position, float2 dir, float2 origin, float speed, float scale, float time) {
    float2 offset = normalize(dir) * speed * time;
    float2 uv = (position.xz - origin) * scale;
    uv += offset;
    return fract(uv);
}

// ---- Ocean vertex ----

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
    float2 fragUV;        // normal map UV (scrolling)
    float3 color;
};

vertex VertexOut oceanVertex(VertexIn in [[stage_in]],
                             constant SceneUniforms&  uniforms [[buffer(1)]],
                             constant SurfaceUniforms& surface [[buffer(2)]],
                             texture2d<float> gustNoiseTex [[texture(0)]],
                             sampler repeatSampler [[sampler(0)]]) {
    VertexOut out;

    // Gust UV
    float2 gustUV = calcScrollUV(in.position,
                                 surface.gustDirection,
                                 float2(-60.0),
                                 surface.gustSpeed,
                                 surface.gustScale,
                                 uniforms.time);

    // Normal map UV
    float2 normalUV = calcScrollUV(in.position,
                                   surface.normalMappingDirection,
                                   float2(-60.0),
                                   surface.normalMappingSpeed,
                                   surface.normalMappingScale,
                                   uniforms.time);

    float3 newPosition = calcNewPosition(in.position, uniforms);

    // Gust displacement
    float gustSample = gustNoiseTex.sample(repeatSampler, gustUV).r;
    float gustDisp = (gustSample - 0.5) * 2.0 * surface.gustStrength;
    newPosition.y += gustDisp;

    float3 tangent;
    float3 bitangent;
    float3 normal = calcNormal(in.position, newPosition, NORMAL_OFFSET, uniforms, tangent, bitangent);

    float4x4 mvp = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix;
    out.position  = mvp * float4(newPosition, 1.0);
    out.fragPos   = newPosition;
    out.normal    = normalize(normal);
    out.tangent   = tangent;
    out.bitangent = bitangent;
    out.fragUV    = normalUV;
    out.color     = surface.baseColor.rgb;

    return out;
}

// ---- Shared fragment utilities ----

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

// ---- Ocean fragment ----

fragment float4 oceanFragment(VertexOut in [[stage_in]],
                               constant SceneUniforms&  uniforms [[buffer(1)]],
                               constant SurfaceUniforms& surface [[buffer(2)]],
                               texture2d<float>  normalMapTex  [[texture(0)]],
                               texturecube<float> envMapTex    [[texture(1)]],
                               texture2d<float>  reflectionTex [[texture(2)]],
                               sampler repeatSampler  [[sampler(0)]],
                               sampler clampSampler   [[sampler(1)]]) {

    // Clip below water in reflection pass
    if (uniforms.isReflectionPass == 1 && in.fragPos.y < -0.1) {
        discard_fragment();
    }

    // Normal map via TBN
    float3 T = normalize(in.tangent);
    float3 B = normalize(in.bitangent);
    float3 N = normalize(in.normal);
    float3x3 TBN = float3x3(T, B, N);

    float3 sampledNormalTS = normalMapTex.sample(repeatSampler, in.fragUV).rgb;
    sampledNormalTS = normalize(sampledNormalTS * 2.0 - 1.0);
    float3 normal = normalize(TBN * sampledNormalTS);

    float3 viewDir  = normalize(uniforms.cameraPos - in.fragPos);
    float3 lightDir = normalize(surface.lightPos - in.fragPos);

    float diff     = max(dot(normal, lightDir), 0.0);
    float cosTheta = max(dot(viewDir, normal), 0.0);
    float3 fresnel = fresnelSchlick(cosTheta, surface.fresnelF0);

    // Environment reflection (cubemap)
    float3 reflectedDir = reflect(-viewDir, normal);
    float3 envReflection = envMapTex.sample(repeatSampler, reflectedDir).rgb;

    // Planar reflection
    float4 clipSpaceRefl = uniforms.reflectionMatrix * float4(in.fragPos, 1.0);
    float2 reflUV = (clipSpaceRefl.xy / clipSpaceRefl.w) * 0.5 + 0.5;
    reflUV += float2(normal.x, normal.z) * surface.reflectionDistortion;
    reflUV = clamp(reflUV, 0.001, 0.999);
    float3 planarRefl = reflectionTex.sample(clampSampler, reflUV).rgb;

    // Blend: planar more visible at grazing angles
    float planarWeight = 1.0 - cosTheta;
    float3 combinedReflection = mix(envReflection, planarRefl, planarWeight);
    float3 reflection = fresnel * combinedReflection;

    // Diffuse
    float3 diffuse = (1.0 - fresnel) * diff * in.color;

    // Depth-based tint
    float depth = length(uniforms.cameraPos - in.fragPos);
    float depthFade = smoothstep(surface.depthFadeNear, surface.depthFadeFar, depth);
    float3 deepColor = in.color * surface.deepWaterTint;
    float3 shiftedColor = mix(in.color, deepColor, depthFade);
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

// ---- Wireframe ----

fragment float4 wireframeFragment(VertexOut in [[stage_in]]) {
    return float4(0.0, 0.5, 0.5, 1.0); // teal, matches web/native wireframe_shader color
}

// ---- Skybox ----

constant float3 skyboxPositions[36] = {
    // +X face
    float3( 1, -1, -1), float3( 1,  1, -1), float3( 1,  1,  1),
    float3( 1, -1, -1), float3( 1,  1,  1), float3( 1, -1,  1),
    // -X face
    float3(-1, -1,  1), float3(-1,  1,  1), float3(-1,  1, -1),
    float3(-1, -1,  1), float3(-1,  1, -1), float3(-1, -1, -1),
    // +Y face
    float3(-1,  1, -1), float3(-1,  1,  1), float3( 1,  1,  1),
    float3(-1,  1, -1), float3( 1,  1,  1), float3( 1,  1, -1),
    // -Y face
    float3(-1, -1,  1), float3(-1, -1, -1), float3( 1, -1, -1),
    float3(-1, -1,  1), float3( 1, -1, -1), float3( 1, -1,  1),
    // +Z face
    float3( 1, -1,  1), float3( 1,  1,  1), float3(-1,  1,  1),
    float3( 1, -1,  1), float3(-1,  1,  1), float3(-1, -1,  1),
    // -Z face
    float3(-1, -1, -1), float3(-1,  1, -1), float3( 1,  1, -1),
    float3(-1, -1, -1), float3( 1,  1, -1), float3( 1, -1, -1),
};

struct SkyboxOut {
    float4 position [[position]];
    float3 texCoord;
};

vertex SkyboxOut skyboxVertex(uint vid [[vertex_id]],
                               constant SceneUniforms& scene [[buffer(1)]]) {
    float3 pos = skyboxPositions[vid];

    // Strip translation from view matrix
    float4x4 viewRot = scene.viewMatrix;
    viewRot[3] = float4(0, 0, 0, 1);

    float4 clip = scene.projectionMatrix * viewRot * float4(pos, 1.0);

    SkyboxOut out;
    out.position = clip.xyww; // force depth = 1.0
    out.texCoord = pos;
    return out;
}

fragment float4 skyboxFragment(SkyboxOut in [[stage_in]],
                                texturecube<float> envMap [[texture(0)]],
                                sampler envSampler [[sampler(0)]]) {
    return float4(envMap.sample(envSampler, in.texCoord).rgb, 1.0);
}

// ---- Buoy shaders (PBR with textures) ----

struct BuoyVertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

struct BuoyVertexOut {
    float4 position [[position]];
    float3 fragPos;
    float3 T;
    float3 B;
    float3 N;
    float2 texCoord;
};

// 3D value noise for hull displacement
float hash3f(float3 p) {
    p = fract(p * float3(127.1, 311.7, 74.7));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

float valueNoise3(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash3f(i),               hash3f(i + float3(1,0,0)), u.x),
            mix(hash3f(i + float3(0,1,0)), hash3f(i + float3(1,1,0)), u.x), u.y),
        mix(mix(hash3f(i + float3(0,0,1)), hash3f(i + float3(1,0,1)), u.x),
            mix(hash3f(i + float3(0,1,1)), hash3f(i + float3(1,1,1)), u.x), u.y), u.z);
}

float hullNoise(float3 p) {
    float n = valueNoise3(p * 2.1) * 0.65 + valueNoise3(p * 4.3) * 0.35;
    return n * 2.0 - 1.0;
}

vertex BuoyVertexOut buoyVertex(BuoyVertexIn in [[stage_in]],
                                constant SceneUniforms& scene [[buffer(1)]],
                                constant float4x4&      model [[buffer(2)]],
                                constant BuoyUniforms&  buoy  [[buffer(3)]]) {
    float3 localNormal = normalize(in.position);

    // Build local TBN
    float3 up      = abs(localNormal.y) < 0.999 ? float3(0,1,0) : float3(1,0,0);
    float3 localT  = normalize(cross(up, localNormal));
    float3 localB  = cross(localNormal, localT);

    // Hull displacement
    float eps = 0.02;
    float  d  = hullNoise(in.position);
    float  dT = hullNoise(in.position + localT * eps);
    float  dB = hullNoise(in.position + localB * eps);

    float3 displacedPos = in.position + localNormal * d * buoy.hullDisplacementAmt;

    float3 dpT = localT * eps + localNormal * (dT - d) * buoy.hullDisplacementAmt;
    float3 dpB = localB * eps + localNormal * (dB - d) * buoy.hullDisplacementAmt;
    float3 perturbedLocalNormal = normalize(cross(dpT, dpB));

    float3x3 modelRot = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
    float3 N = normalize(modelRot * perturbedLocalNormal);
    float3 T = normalize(modelRot * localT);
    float3 B = cross(N, T);

    // Spherical UV
    float2 uv = float2(atan2(localNormal.z, localNormal.x) / (2.0 * PI) + 0.5,
                       acos(clamp(localNormal.y, -1.0, 1.0)) / PI);

    float4 worldPos = model * float4(displacedPos, 1.0);
    float4x4 vp = scene.projectionMatrix * scene.viewMatrix;

    BuoyVertexOut out;
    out.position = vp * worldPos;
    out.fragPos  = worldPos.xyz;
    out.T        = T;
    out.B        = B;
    out.N        = N;
    out.texCoord = uv;
    return out;
}

// Wave height evaluation (CPU mirror in Swift does the same, but needed in fragment for waterline)
float waveHeightAtXZ(float2 xz, constant SceneUniforms& scene) {
    float h = 0.0;
    for (int i = 0; i < scene.numWaves; i++) {
        float k = 2.0 * PI / max(scene.waves[i].wavelength, 0.01);
        float w = sqrt(GRAVITY * k);
        float2 D = normalize(scene.waves[i].direction);
        float phase = dot(D * k, xz) - fmod(w * scene.time, 2.0 * PI) + scene.waves[i].phase;
        h += scene.waves[i].amplitude * cos(phase);
    }
    return h;
}

fragment float4 buoyFragment(BuoyVertexOut         in      [[stage_in]],
                              constant SceneUniforms&  scene   [[buffer(1)]],
                              constant SurfaceUniforms& surface [[buffer(2)]],
                              constant BuoyUniforms&   buoy    [[buffer(3)]],
                              texture2d<float>  colorMap    [[texture(0)]],
                              texture2d<float>  normalMap   [[texture(1)]],
                              texture2d<float>  bumpMap     [[texture(2)]],
                              texture2d<float>  roughnessTex [[texture(3)]],
                              texturecube<float> envMap     [[texture(4)]],
                              sampler repeatSampler [[sampler(0)]]) {

    if (scene.isReflectionPass == 1 && in.fragPos.y < 0.0) {
        discard_fragment();
    }

    float3 albedo = colorMap.sample(repeatSampler, in.texCoord).rgb;

    float3x3 TBN = float3x3(in.T, in.B, in.N);
    float3 sampledNormal = normalMap.sample(repeatSampler, in.texCoord).rgb;
    sampledNormal = normalize(sampledNormal * 2.0 - 1.0);
    float3 normal = normalize(TBN * sampledNormal);

    float height = bumpMap.sample(repeatSampler, in.texCoord).r;
    normal = normalize(normal + float3(0, 0, height * buoy.bumpFactor));

    float3 lightDir  = normalize(surface.lightPos - in.fragPos);
    float3 viewDir   = normalize(scene.cameraPos - in.fragPos);
    float3 reflDir   = reflect(-viewDir, normal);

    float diff = max(dot(normal, lightDir), 0.0);
    float roughness  = roughnessTex.sample(repeatSampler, in.texCoord).r;
    float shininess  = pow(1.0 - roughness, 2.0);

    float3 envLighting = envMap.sample(repeatSampler, normal).rgb;
    float3 ambient     = 0.2 * albedo * envLighting;
    float3 specSample  = envMap.sample(repeatSampler, reflDir).rgb;
    float3 specular    = buoy.specularFactor * shininess * specSample;
    float3 diffuse     = 0.6 * diff * albedo;

    // Waterline
    float surfaceY = waveHeightAtXZ(in.fragPos.xz, scene);
    float belowWater    = smoothstep(surfaceY + 0.2, surfaceY - 0.5, in.fragPos.y);
    float bandCenter    = surfaceY - buoy.waterlineBias;
    float waterlineBand = 1.0 - smoothstep(0.0, buoy.waterlineWidth, abs(in.fragPos.y - bandCenter));
    float bandNoise     = fbm(in.fragPos.xz * 2.5 + scene.time * 0.05);
    float noiseMask     = mix(1.0, smoothstep(0.25, 0.75, bandNoise), buoy.waterlineNoise);
    waterlineBand      *= noiseMask;

    float3 wetTint = float3(0.05, 0.08, 0.1);
    float3 color   = mix(ambient + diffuse + specular, wetTint, belowWater * buoy.wetStrength);
    color          = mix(color, float3(0.9, 0.95, 1.0), waterlineBand * buoy.waterlineStrength);

    return float4(color, 1.0);
}
