#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared structs — must match ShaderTypes.h exactly
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

struct SurfaceUniforms {
    float4  baseColor;
    float4  fogColor;
    float   fogDensity;
    float   _padFog0, _padFog1, _padFog2;
    float4  causticColor;
    float   causticScale;
    float   causticSpeed;
    float   causticIntensity;
    float   causticTroughMin;
    float   causticTroughMax;
    float   causticThresholdMin;
    float   causticThresholdMax;
    float   causticSharpness;
    float   foamScale;
    float   foamScrollSpeed;
    float   foamSlopeMin;
    float   foamSlopeMax;
    float   foamSlopeAmplifier;
    float   foamPower;
    float   depthFadeNear;
    float   depthFadeFar;
    float4  deepWaterTint;
    float   fresnelF0;
    float   gamma;
    float   reflectionDistortion;
    float   _padRend0;
    float2  gustDirection;
    float   gustSpeed;
    float   gustScale;
    float   gustStrength;
    float   reflectionStrength;
    float   foamEdgeNoise;
    float   _padGust2;
    float   normalMappingScale;
    float   normalMappingSpeed;
    float2  normalMappingDir;
};

// ---------------------------------------------------------------------------
// Debug cube — unit cube at origin.
// Uses the same buffer bindings as ModelDrawable: SceneUniforms at 1, model matrix at 2.
// ---------------------------------------------------------------------------

struct CubeVertOut {
    float4 position [[position]];
    float3 localPos;
};

vertex CubeVertOut cubeVertex(
    uint                    vertexID    [[vertex_id]],
    constant float4*        positions   [[buffer(0)]],
    constant SceneUniforms& scene       [[buffer(1)]],
    constant float4x4&      modelMatrix [[buffer(2)]])
{
    float3 p   = positions[vertexID].xyz;
    float4x4 mvp = scene.projectionMatrix * scene.viewMatrix * modelMatrix;
    CubeVertOut out;
    out.position = mvp * float4(p, 1.0);
    out.localPos = p;
    return out;
}

fragment float4 cubeFragment(CubeVertOut in [[stage_in]])
{
    return float4(in.localPos + 0.5, 1.0);
}

// ---------------------------------------------------------------------------
// Ocean vertex input
// Layout: packed float3 position (12) + float2 texCoord (8) = stride 20
// ---------------------------------------------------------------------------

struct OceanVertexIn {
    float3 position  [[attribute(0)]];
    float2 texCoord  [[attribute(1)]];
};

struct OceanVertexOut {
    float4 clipPos   [[position]];
    float3 fragPos;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 fragUV;
};

// ---------------------------------------------------------------------------
// Gerstner wave helpers — mirrors waveOffset() in gerstner.vert
// ---------------------------------------------------------------------------

float3 gerstnerOffset(float3 basePos, constant WaveUniform* waves, int n, float time) {
    float3 offset = 0;
    for (int i = 0; i < n; ++i) {
        float k = 2.0 * M_PI_F / max(waves[i].wavelength, 0.01f);
        float w = sqrt(9.81f * k);
        float2 D = normalize(waves[i].direction);
        float phase = dot(D * k, basePos.xz) - fmod(w * time, 2.0f * M_PI_F) + waves[i].phase;
        offset.y   += waves[i].amplitude * cos(phase);
        float2 xz   = -waves[i].steepness * D * sin(phase) * waves[i].amplitude;
        offset.x   += xz.x;
        offset.z   += xz.y;
    }
    return offset;
}

float3 displacedPos(float3 base, constant WaveUniform* waves, int n, float time) {
    return base + gerstnerOffset(base, waves, n, time);
}

float2 calcUV(float3 pos, float2 dir, float speed, float scale, float time) {
    float2 offset = normalize(dir) * speed * time;
    float2 uv = (pos.xz - float2(-60)) * scale + offset;
    return fract(uv);
}

// ---------------------------------------------------------------------------
// Gust
// ---------------------------------------------------------------------------

float gustDisplacement(float2 uv, texture2d<float> gustTex, sampler samp,
                       float strength) {
    float v = gustTex.sample(samp, uv).r;
    return (v - 0.5) * 2.0 * strength;
}

// ---------------------------------------------------------------------------
// Noise / FBM helpers (shared with fragment)
// ---------------------------------------------------------------------------

float hashOcean(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}
float valueNoiseOcean(float2 p) {
    float2 i = floor(p), f = fract(p);
    float a = hashOcean(fmod(i,               289.0f));
    float b = hashOcean(fmod(i + float2(1,0), 289.0f));
    float c = hashOcean(fmod(i + float2(0,1), 289.0f));
    float d = hashOcean(fmod(i + float2(1,1), 289.0f));
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbmOcean(float2 p) {
    float s = 0, a = 0.5, fr = 1;
    for (int i = 0; i < 5; ++i) {
        s += a * valueNoiseOcean(fmod(p * fr, 289.0f));
        fr *= 2; a *= 0.5;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Shadow depth vertex shader — same displacement as oceanVertex, but outputs
// into light clip space.  No fragment function needed; depth writes happen
// during rasterisation.
// ---------------------------------------------------------------------------

vertex float4 oceanShadowVertex(
    OceanVertexIn             in      [[stage_in]],
    constant SceneUniforms&   scene   [[buffer(1)]],
    constant SurfaceUniforms& surface [[buffer(2)]],
    texture2d<float>          gustTex [[texture(0)]],
    sampler                   samp    [[sampler(0)]])
{
    float3 base   = in.position;
    float3 newPos = displacedPos(base, scene.waves, scene.numWaves, scene.time);
    float2 gustUV = calcUV(base, surface.gustDirection,
                           surface.gustSpeed, surface.gustScale, scene.time);
    newPos.y += gustDisplacement(gustUV, gustTex, samp, surface.gustStrength);
    float4 worldPos = scene.modelMatrix * float4(newPos, 1.0);
    return scene.lightSpaceMatrix * worldPos;
}

// ---------------------------------------------------------------------------
// Ocean vertex shader
// ---------------------------------------------------------------------------

vertex OceanVertexOut oceanVertex(
    OceanVertexIn             in      [[stage_in]],
    constant SceneUniforms&   scene   [[buffer(1)]],
    constant SurfaceUniforms& surface [[buffer(2)]],
    texture2d<float>          gustTex [[texture(0)]],
    sampler                   samp    [[sampler(0)]])
{
    float3 base = in.position;

    float2 gustUV   = calcUV(base, surface.gustDirection,
                             surface.gustSpeed, surface.gustScale, scene.time);
    float2 normalUV = calcUV(base, surface.normalMappingDir,
                             surface.normalMappingSpeed, surface.normalMappingScale, scene.time);

    float3 newPos = displacedPos(base, scene.waves, scene.numWaves, scene.time);

    // Finite-difference normals from Gerstner only. Gust must not be included
    // here: its UV delta would need to be eps*gustScale in texture space, but
    // eps is a world-space value — using it directly as a UV offset is 111×
    // too large, causing fract() discontinuities that corrupt the normal.
    const float eps = 0.01;
    float3 posX = displacedPos(base + float3(eps, 0, 0), scene.waves, scene.numWaves, scene.time);
    float3 posZ = displacedPos(base + float3(0, 0, eps), scene.waves, scene.numWaves, scene.time);

    float3 tangent   = posX - newPos;
    float3 bitangent = posZ - newPos;
    float3 normal    = normalize(cross(bitangent, tangent));

    // Gust applied after normals — it's a gentle vertical offset, not a surface deformation.
    float  gust   = gustDisplacement(gustUV, gustTex, samp, surface.gustStrength);
    newPos.y += gust;

    float4 worldPos = scene.modelMatrix * float4(newPos, 1.0);

    OceanVertexOut out;
    out.clipPos   = scene.projectionMatrix * scene.viewMatrix * worldPos;
    out.fragPos   = worldPos.xyz;
    out.normal    = normal;
    out.tangent   = tangent;
    out.bitangent = bitangent;
    out.fragUV    = normalUV;
    return out;
}

// ---------------------------------------------------------------------------
// Wireframe fragment shader (solid colour, no lighting)
// ---------------------------------------------------------------------------

fragment float4 wireframeFragment(OceanVertexOut in [[stage_in]],
                                  constant SurfaceUniforms& surface [[buffer(2)]]) {
    // Subtle blue-green tint matching the reference wireframe colour
    return float4(0.15, 0.36, 0.25, 1.0);
}

// ---------------------------------------------------------------------------
// Shadow PCF — port of shadowPCF() in water.frag.
// Metal NDC z is already in [0,1] (no *0.5+0.5 needed for z).
// y is flipped relative to texture v: NDC +1 = top, UV v=0 = top.
// ---------------------------------------------------------------------------

float shadowPCF(float3 fragPos, float3 normal, float3 lightDir,
                depth2d<float>          shadowMap,
                sampler                 shadowSamp,
                constant SceneUniforms& scene)
{
    float4 fragLS     = scene.lightSpaceMatrix * float4(fragPos, 1.0);
    float3 proj       = fragLS.xyz / fragLS.w;

    if (proj.z > 1.0) return 1.0;

    float2 shadowUV   = proj.xy * float2(0.5, -0.5) + 0.5;
    if (any(shadowUV < 0.0) || any(shadowUV > 1.0)) return 1.0;

    float bias         = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    float currentDepth = proj.z - bias;

    float2 texelSize   = 1.0 / float2(shadowMap.get_width(), shadowMap.get_height());
    float  shadow      = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float pcfDepth = shadowMap.sample(shadowSamp,
                                              shadowUV + float2(x, y) * texelSize);
            shadow += currentDepth < pcfDepth ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

// ---------------------------------------------------------------------------
// Fresnel helpers
// ---------------------------------------------------------------------------

float3 fresnelSchlick(float cosTheta, float f0) {
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// ---------------------------------------------------------------------------
// Ocean fragment shader — port of water.frag
// ---------------------------------------------------------------------------

fragment float4 oceanFragment(
    OceanVertexOut            in          [[stage_in]],
    constant SceneUniforms&   scene       [[buffer(1)]],
    constant SurfaceUniforms& surface     [[buffer(2)]],
    texture2d<float>          normalMap   [[texture(0)]],
    texturecube<float>        envMap      [[texture(1)]],
    texture2d<float>          reflTex     [[texture(2)]],
    depth2d<float>            shadowMap   [[texture(3)]],
    sampler                   repeatSamp  [[sampler(0)]],
    sampler                   clampSamp   [[sampler(1)]],
    sampler                   shadowSamp  [[sampler(2)]])
{
    float3x3 TBN = float3x3(normalize(in.tangent),
                             normalize(in.bitangent),
                             normalize(in.normal));

    float3 sampledN = normalMap.sample(repeatSamp, in.fragUV).rgb * 2.0 - 1.0;
    float3 normal   = normalize(TBN * sampledN);

    float3 lightDir = normalize(scene.lightPos.xyz - in.fragPos);
    float3 viewDir  = normalize(scene.cameraPos.xyz - in.fragPos);

    float diff = max(dot(normal, lightDir), 0.0f);

    float3 reflectedDir = reflect(-viewDir, normal);
    float3 envRefl = envMap.sample(clampSamp, reflectedDir).rgb;

    // Planar reflection — project frag pos into reflection UV
    float4 clipRefl = scene.reflectionMatrix * float4(in.fragPos, 1.0);
    // Metal textures have v=0 at the top; OpenGL has v=0 at the bottom.
    // Negate y so that NDC +1 (top of screen) maps to v=0 (top of texture).
    float2 reflUV   = (clipRefl.xy / clipRefl.w) * float2(0.5, -0.5) + 0.5;
    reflUV += float2(normal.x, normal.z) * surface.reflectionDistortion;
    reflUV  = clamp(reflUV, 0.001, 0.999);
    float3 planarRefl = reflTex.sample(clampSamp, reflUV).rgb;

    float  cosTheta       = max(dot(viewDir, normal), 0.0f);
    float  planarWeight   = 1.0 - cosTheta;
    float3 combinedRefl   = mix(envRefl, planarRefl, planarWeight);
    float3 fresnel        = fresnelSchlick(cosTheta, surface.fresnelF0);
    float3 reflection     = fresnel * combinedRefl;

    // Depth-based colour
    float  depth     = length(scene.cameraPos.xyz - in.fragPos);
    float  depthFade = smoothstep(surface.depthFadeNear, surface.depthFadeFar, depth);
    float3 baseColor = surface.baseColor.rgb;
    float3 deepColor = baseColor * surface.deepWaterTint.xyz;
    float3 color     = mix(baseColor, deepColor, depthFade);
    color = pow(color, surface.gamma);

    float  shadow  = shadowPCF(in.fragPos, normal, lightDir, shadowMap, shadowSamp, scene);
    float3 diffuse = (1.0 - fresnel) * diff * shadow * color;

    // Caustics
    float causticStr = smoothstep(surface.causticTroughMin, surface.causticTroughMax,
                                   -in.fragPos.y);
    float2 flickUV   = in.fragPos.xz * surface.causticScale + scene.time * surface.causticSpeed;
    float  flicker   = fbmOcean(flickUV);
    flicker = smoothstep(surface.causticThresholdMin, surface.causticThresholdMax, flicker);
    float  NdotL     = max(dot(normalize(in.normal), normalize(scene.lightPos.xyz - in.fragPos)), 0.0f);
    flicker *= NdotL;
    flicker  = pow(flicker, surface.causticSharpness);
    float3 caustic = surface.causticColor.xyz * flicker * causticStr * surface.causticIntensity;

    float3 finalColor = reflection + diffuse + caustic;

    // Foam
    float slope    = length(float2(dfdx(in.fragPos.y), dfdy(in.fragPos.y))) * surface.foamSlopeAmplifier;
    float2 slopeDir = normalize(float2(dfdx(in.fragPos.y), dfdy(in.fragPos.y)));
    float2 foamUV  = in.fragPos.xz + slopeDir * scene.time * surface.foamScrollSpeed;
    float  foam    = fbmOcean(foamUV * surface.foamScale);
    // Perturb the lower threshold with the foam noise so the bottom edge is ragged
    // rather than a clean geometric line.
    float  foamMask = smoothstep(surface.foamSlopeMin - foam * surface.foamEdgeNoise, surface.foamSlopeMax, slope);
    foam = pow(foam * foamMask, surface.foamPower);
    finalColor = mix(finalColor, float3(1.0), foam);

    // Fog
    float fogFactor = clamp(exp(-surface.fogDensity * depth), 0.0f, 1.0f);
    finalColor = mix(surface.fogColor.xyz, finalColor, fogFactor);

    return float4(finalColor, 1.0);
}

// ---------------------------------------------------------------------------
// Skybox — fullscreen unit-cube trick; no vertex buffer needed
// ---------------------------------------------------------------------------

constant float3 skyboxVerts[36] = {
    // -Z face
    {-1,-1,-1},{-1, 1,-1},{ 1, 1,-1},{ 1, 1,-1},{ 1,-1,-1},{-1,-1,-1},
    // +Z face
    {-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{ 1, 1, 1},{-1, 1, 1},{-1,-1, 1},
    // -X face
    {-1, 1, 1},{-1, 1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1, 1},{-1, 1, 1},
    // +X face
    { 1, 1,-1},{ 1, 1, 1},{ 1,-1, 1},{ 1,-1, 1},{ 1,-1,-1},{ 1, 1,-1},
    // -Y face
    {-1,-1,-1},{ 1,-1,-1},{ 1,-1, 1},{ 1,-1, 1},{-1,-1, 1},{-1,-1,-1},
    // +Y face
    {-1, 1,-1},{-1, 1, 1},{ 1, 1, 1},{ 1, 1, 1},{ 1, 1,-1},{-1, 1,-1},
};

struct SkyboxOut {
    float4 clipPos [[position]];
    float3 texDir;
};

vertex SkyboxOut skyboxVertex(uint vid [[vertex_id]],
                              constant SceneUniforms& scene [[buffer(1)]]) {
    float3 pos = skyboxVerts[vid];
    // Strip translation from view matrix
    float4x4 viewNoTranslation = float4x4(
        scene.viewMatrix[0],
        scene.viewMatrix[1],
        scene.viewMatrix[2],
        float4(0, 0, 0, 1));
    float4 clipPos = scene.projectionMatrix * viewNoTranslation * float4(pos, 1.0);
    // Push to far plane: set z = w so depth = 1.0 after divide
    SkyboxOut out;
    out.clipPos = clipPos.xyww;
    out.texDir  = pos;
    return out;
}

fragment float4 skyboxFragment(SkyboxOut       in     [[stage_in]],
                               texturecube<float> envMap [[texture(0)]],
                               sampler            samp   [[sampler(0)]]) {
    return float4(envMap.sample(samp, normalize(in.texDir)).rgb, 1.0);
}

// ---------------------------------------------------------------------------
// Surface-normals visualisation — draws one yellow line per ocean vertex,
// originating at the Gerstner-displaced surface position (same displacement
// as oceanVertex so lines sit exactly on the wave surface).
//
// Metal has no geometry shaders; the vertex_id trick is used instead:
//   vertexID / 2  → ocean vertex index
//   vertexID % 2  → 0 = base, 1 = tip
// Draw vertexCount*2 vertices as .line primitives (pairs → line segments).
// ---------------------------------------------------------------------------

/// Packed layout matching OceanVertex in MeshBuilder.swift:
/// position (12 bytes) + texCoord (8 bytes) = stride 20.
/// packed_float2 is used (not float2) to keep 4-byte alignment and
/// match the Swift struct stride exactly — float2 has 8-byte alignment
/// which would pad the struct to 24 bytes and mis-index every vertex.
struct OceanVertexData {
    packed_float3 position;
    packed_float2 texCoord;
};

struct NormalsVertOut {
    float4 clipPos [[position]];
    float3 color;
};

vertex NormalsVertOut normalsVertex(
    uint                        vertexID [[vertex_id]],
    constant OceanVertexData*   vertices [[buffer(0)]],
    constant SceneUniforms&     scene    [[buffer(1)]])
{
    // One normal line per ocean vertex: even vertexID = base, odd = tip.
    uint  vi    = vertexID / 2;
    bool  isTip = (vertexID & 1) == 1;

    float3 base = float3(vertices[vi].position);

    // Gerstner displacement — identical to oceanVertex (gust excluded so that
    // normals reflect the true wave surface, not the gentle vertical gust offset).
    float3 newPos = displacedPos(base, scene.waves, scene.numWaves, scene.time);

    const float eps = 0.01;
    float3 posX = displacedPos(base + float3(eps, 0, 0), scene.waves, scene.numWaves, scene.time);
    float3 posZ = displacedPos(base + float3(0, 0, eps), scene.waves, scene.numWaves, scene.time);

    float3 tangent   = posX - newPos;
    float3 bitangent = posZ - newPos;
    float3 normal    = normalize(cross(bitangent, tangent));

    float4 worldPos    = scene.modelMatrix * float4(newPos, 1.0);
    float3 worldNormal = normalize((scene.modelMatrix * float4(normal, 0.0)).xyz);

    const float MAGNITUDE = 0.5;
    float3 pos = isTip ? worldPos.xyz + worldNormal * MAGNITUDE : worldPos.xyz;

    NormalsVertOut out;
    out.clipPos = scene.projectionMatrix * scene.viewMatrix * float4(pos, 1.0);
    out.color   = float3(1.0, 1.0, 0.0);  // yellow — matches C++ reference gerstner_normal.gs
    return out;
}

fragment float4 normalsFragment(NormalsVertOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
