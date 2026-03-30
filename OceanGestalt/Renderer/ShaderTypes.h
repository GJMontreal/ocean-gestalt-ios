#pragma once
#include <simd/simd.h>

#define MAX_WAVES 10

typedef struct {
    simd_float2 direction;
    float amplitude;
    float wavelength;
    float steepness;
    float phase;
    float _pad0;
    float _pad1;
} WaveUniform;

typedef struct {
    simd_float4x4 modelMatrix;
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 reflectionMatrix;
    simd_float3 cameraPos;
    float time;
    int numWaves;
    int isReflectionPass;
    float _pad0;
    float _pad1;
    WaveUniform waves[MAX_WAVES];
} SceneUniforms;

typedef struct {
    // Gust
    simd_float2 gustDirection;
    float gustSpeed;
    float gustScale;
    float gustStrength;
    // Normal mapping
    float normalMappingScale;
    float normalMappingSpeed;
    simd_float2 normalMappingDirection;
    // Foam
    float foamScale;
    float foamScrollSpeed;
    float foamSlopeMin;
    float foamSlopeMax;
    float foamSlopeAmplifier;
    float foamPower;
    // Fog
    float fogDensity;
    simd_float3 fogColor;
    // Caustics
    float causticScale;
    float causticSpeed;
    float causticIntensity;
    simd_float3 causticColor;
    float causticTroughMin;
    float causticTroughMax;
    float causticThresholdMin;
    float causticThresholdMax;
    float causticSharpness;
    // Depth
    float depthFadeNear;
    float depthFadeFar;
    simd_float3 deepWaterTint;
    // Fresnel / gamma
    float fresnelF0;
    float gamma;
    // Reflection
    float reflectionDistortion;
    // Light
    simd_float3 lightPos;
    simd_float4 baseColor;
} SurfaceUniforms;

typedef struct {
    float hullDisplacementAmt;
    float waterlineBias;
    float waterlineWidth;
    float waterlineStrength;
    float waterlineNoise;
    float wetStrength;
    float specularFactor;
    float bumpFactor;
} BuoyUniforms;
