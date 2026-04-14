#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Structs — must match ShaderTypes.h exactly
// ---------------------------------------------------------------------------

#define MAX_WAVES 10

struct WaveUniform {
    float2 direction;
    float  amplitude;
    float  wavelength;
    float  steepness;
    float  phase;
    float  _pad0;
    float  _pad1;
};

struct SceneUniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 reflectionMatrix;
    float4x4 lightSpaceMatrix;
    float4   cameraPos;
    float4   lightPos;
    float    time;
    int      isReflectionPass;
    int      numWaves;
    float    _pad0;
    WaveUniform waves[MAX_WAVES];
};

struct ModelUniforms {
    float4 wireframeColor;
    float  waterlineClip;
    float  waterlineBias;
    float  waterlineWidth;
    float  waterlineStrength;
    float  waterlineNoise;
    float  wetStrength;
    float  specularFactor;
    float  bumpFactor;
};

// ---------------------------------------------------------------------------
// Vertex layout — packed to match ModelDrawable.mtlVertexDescriptor
// position: float3 (12), texCoord: float2 (8), normal: float3 (12),
// tangent: float4 (16) — stride 48
// ---------------------------------------------------------------------------

struct GltfVertexIn {
    float3 position  [[attribute(0)]];
    float2 texCoord  [[attribute(1)]];
    float3 normal    [[attribute(2)]];
    float4 tangent   [[attribute(3)]];  // .w = handedness
};

struct GltfVertexOut {
    float4 clipPos  [[position]];
    float3 fragPos;
    float2 texCoord;
    float3 tangent;
    float3 bitangent;
    float3 normal;
};

// ---------------------------------------------------------------------------
// Wave helpers (matches evaluateGerstnerWaves in C++ reference)
// ---------------------------------------------------------------------------

float2 waveXZDispAt(float2 xz, constant WaveUniform* waves, int numWaves, float time) {
    float2 disp = 0;
    for (int i = 0; i < numWaves; ++i) {
        float k = 2.0 * M_PI_F / max(waves[i].wavelength, 0.01f);
        float w = sqrt(9.81f * k);
        float2 D = normalize(waves[i].direction);
        float phase = dot(D * k, xz) - fmod(w * time, 2.0f * M_PI_F) + waves[i].phase;
        disp -= waves[i].steepness * D * sin(phase) * waves[i].amplitude;
    }
    return disp;
}

float waveHeightAt(float2 worldXZ, constant WaveUniform* waves, int numWaves, float time) {
    // Approximate rest XZ by subtracting XZ displacement at the query point
    float2 restXZ = worldXZ - waveXZDispAt(worldXZ, waves, numWaves, time);
    float h = 0;
    for (int i = 0; i < numWaves; ++i) {
        float k = 2.0 * M_PI_F / max(waves[i].wavelength, 0.01f);
        float w = sqrt(9.81f * k);
        float2 D = normalize(waves[i].direction);
        float phase = dot(D * k, restXZ) - fmod(w * time, 2.0f * M_PI_F) + waves[i].phase;
        h += waves[i].amplitude * cos(phase);
    }
    return h;
}

// ---------------------------------------------------------------------------
// Noise helpers (used for waterline noise)
// ---------------------------------------------------------------------------

float hashVal(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float valueNoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float a = hashVal(fmod(i,               289.0f));
    float b = hashVal(fmod(i + float2(1, 0), 289.0f));
    float c = hashVal(fmod(i + float2(0, 1), 289.0f));
    float d = hashVal(fmod(i + float2(1, 1), 289.0f));
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbmNoise(float2 p) {
    float sum = 0, amp = 0.5, freq = 1;
    for (int i = 0; i < 5; ++i) {
        sum += amp * valueNoise(fmod(p * freq, 289.0f));
        freq *= 2; amp *= 0.5;
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

vertex GltfVertexOut modelVertex(
    GltfVertexIn     in      [[stage_in]],
    constant SceneUniforms& scene [[buffer(1)]],
    constant float4x4&      modelMatrix [[buffer(2)]])
{
    float3x3 normalMat = float3x3(
        modelMatrix[0].xyz,
        modelMatrix[1].xyz,
        modelMatrix[2].xyz);

    float3 N = normalize(normalMat * in.normal);
    float3 T = normalize(normalMat * in.tangent.xyz);
    float3 B = cross(N, T) * in.tangent.w;

    GltfVertexOut out;
    out.fragPos   = (modelMatrix * float4(in.position, 1.0)).xyz;
    out.texCoord  = in.texCoord;
    out.tangent   = T;
    out.bitangent = B;
    out.normal    = N;
    out.clipPos   = scene.projectionMatrix * scene.viewMatrix * float4(out.fragPos, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — port of normal_bump.frag
// ---------------------------------------------------------------------------

fragment float4 modelFragment(
    GltfVertexOut        in           [[stage_in]],
    constant SceneUniforms&  scene    [[buffer(1)]],
    constant ModelUniforms&  model    [[buffer(3)]],
    texture2d<float>     colorMap     [[texture(0)]],
    texture2d<float>     normalMap    [[texture(1)]],
    texture2d<float>     mrMap        [[texture(2)]],  // G=roughness, B=metallic
    texturecube<float>   envMap       [[texture(3)]],
    sampler              samp         [[sampler(0)]])
{
    // Waterline clip
    float surfaceY = waveHeightAt(in.fragPos.xz, scene.waves, scene.numWaves, scene.time)
                   - model.waterlineBias;
    if (model.waterlineClip >  0.5 && in.fragPos.y <  surfaceY) discard_fragment();
    if (model.waterlineClip < -0.5 && in.fragPos.y >= surfaceY) discard_fragment();

    // Wireframe pass: just output the wireframe colour
    if (model.waterlineClip < -0.5) {
        return float4(model.wireframeColor.xyz, 1.0);
    }

    // Base colour
    float3 albedo = colorMap.sample(samp, in.texCoord).rgb;

    // Normal (tangent → world)
    float3x3 TBN    = float3x3(normalize(in.tangent), normalize(in.bitangent), normalize(in.normal));
    float3 sampledN = normalMap.sample(samp, in.texCoord).rgb * 2.0 - 1.0;
    float3 normal   = normalize(TBN * sampledN);

    // Bump perturbation
    float height = mrMap.sample(samp, in.texCoord).r;
    normal = normalize(normal + float3(0, 0, height * model.bumpFactor));

    // Lighting directions
    float3 lightDir = normalize(scene.lightPos.xyz - in.fragPos);
    float3 viewDir  = normalize(scene.cameraPos.xyz - in.fragPos);
    float3 reflDir  = reflect(-viewDir, normal);

    float diff    = max(dot(normal, lightDir), 0.0f);
    float roughness = mrMap.sample(samp, in.texCoord).g;
    float shininess = pow(1.0 - roughness, 2.0);

    float3 ambient  = 0.2  * albedo;
    float3 diffuse  = 0.6  * diff * albedo;
    float3 specSamp = envMap.sample(samp, reflDir).rgb;
    float3 specular = model.specularFactor * shininess * specSamp;

    float waterFacing = max(dot(normal, float3(0, -1, 0)), 0.0f);
    float3 waterBounce = 0.25 * albedo * envMap.sample(samp, float3(0, -1, 0)).rgb * waterFacing;

    // Waterline foam ring
    float bandCenter  = waveHeightAt(in.fragPos.xz, scene.waves, scene.numWaves, scene.time)
                      - model.waterlineBias;
    float waterlineBand = 1.0 - smoothstep(0.0, model.waterlineWidth,
                                            abs(in.fragPos.y - bandCenter));
    float bandNoise  = fbmNoise(in.fragPos.xz * 2.5 + scene.time * 0.05);
    float noiseMask  = mix(1.0, smoothstep(0.25, 0.75, bandNoise), model.waterlineNoise);
    waterlineBand   *= noiseMask;

    float3 color = ambient + diffuse + specular + waterBounce;
    color = mix(color, float3(0.9, 0.95, 1.0), waterlineBand * model.waterlineStrength);

    // Gamma correction
    color = pow(color, 1.0 / 2.2);

    return float4(color, 1.0);
}
