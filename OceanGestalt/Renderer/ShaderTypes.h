#pragma once
#include <simd/simd.h>

// Shared C structs between Swift and Metal.
//
// simd_float3 is NOT used here — it is 12 bytes in C but 16 bytes in MSL,
// which breaks struct alignment when shared via a buffer.  All vec3-sized
// fields use simd_float4 instead (shaders access .xyz).

// ---------------------------------------------------------------------------
// Wave
// ---------------------------------------------------------------------------

#define MAX_WAVES 10

typedef struct {
    simd_float2 direction;   //  8 bytes  offset  0
    float       amplitude;   //  4 bytes  offset  8
    float       wavelength;  //  4 bytes  offset 12
    float       steepness;   //  4 bytes  offset 16
    float       phase;       //  4 bytes  offset 20
    float       _pad0;       //  4 bytes  offset 24
    float       _pad1;       //  4 bytes  offset 28
} WaveUniform;               // 32 bytes total

// ---------------------------------------------------------------------------
// SceneUniforms — per-frame matrices and scene state; bound to every shader
// ---------------------------------------------------------------------------

typedef struct {
    simd_float4x4 modelMatrix;       //  64 bytes  offset   0
    simd_float4x4 viewMatrix;        //  64 bytes  offset  64
    simd_float4x4 projectionMatrix;  //  64 bytes  offset 128
    simd_float4x4 reflectionMatrix;  //  64 bytes  offset 192
    simd_float4x4 lightSpaceMatrix;  //  64 bytes  offset 256
    simd_float4   cameraPos;         //  16 bytes  offset 320  (.xyz = world pos)
    simd_float4   lightPos;          //  16 bytes  offset 336  (.xyz = world pos)
    float         time;              //   4 bytes  offset 352
    int           isReflectionPass;  //   4 bytes  offset 356
    int           numWaves;          //   4 bytes  offset 360
    float         _pad0;             //   4 bytes  offset 364
    WaveUniform   waves[MAX_WAVES];  // 320 bytes  offset 368
} SceneUniforms;                     // 688 bytes total

// ---------------------------------------------------------------------------
// SurfaceUniforms — ocean surface appearance parameters
// ---------------------------------------------------------------------------

typedef struct {
    // Base appearance
    simd_float4 baseColor;          // .rgba              offset   0

    // Fog
    simd_float4 fogColor;           // .xyz               offset  16
    float       fogDensity;         //                    offset  32
    float       _padFog0;
    float       _padFog1;
    float       _padFog2;           //                    offset  44 → 48

    // Caustics
    simd_float4 causticColor;       // .xyz               offset  48
    float       causticScale;       //                    offset  64
    float       causticSpeed;       //                    offset  68
    float       causticIntensity;   //                    offset  72
    float       causticTroughMin;   //                    offset  76
    float       causticTroughMax;   //                    offset  80
    float       causticThresholdMin;//                    offset  84
    float       causticThresholdMax;//                    offset  88
    float       causticSharpness;   //                    offset  92

    // Foam
    float       foamScale;          //                    offset  96
    float       foamScrollSpeed;    //                    offset 100
    float       foamSlopeMin;       //                    offset 104
    float       foamSlopeMax;       //                    offset 108
    float       foamSlopeAmplifier; //                    offset 112
    float       foamPower;          //                    offset 116

    // Depth fade
    float       depthFadeNear;      //                    offset 120
    float       depthFadeFar;       //                    offset 124

    // Deep water
    simd_float4 deepWaterTint;      // .xyz               offset 128

    // Fresnel / rendering
    float       fresnelF0;          //                    offset 144
    float       gamma;              //                    offset 148
    float       reflectionDistortion;//                   offset 152
    float       _padRend0;          //                    offset 156 → 160

    // Gust
    simd_float2 gustDirection;      //                    offset 160
    float       gustSpeed;          //                    offset 168
    float       gustScale;          //                    offset 172
    float       gustStrength;       //                    offset 176
    float       reflectionStrength; //                    offset 180
    float       foamEdgeNoise;      //                    offset 184
    float       _padGust2;          //                    offset 188 → 192

    // Normal mapping
    float       normalMappingScale; //                    offset 192
    float       normalMappingSpeed; //                    offset 196
    simd_float2 normalMappingDir;   //                    offset 200
                                    //                    offset 204 → 208
} SurfaceUniforms;                  // 208 bytes total

// ---------------------------------------------------------------------------
// ModelUniforms — per-model state for the GLTF PBR shader
// ---------------------------------------------------------------------------

typedef struct {
    simd_float4 wireframeColor;   // .xyz = color           offset  0
    float waterlineClip;          // 1=above, -1=below, 0=none  offset 16
    float waterlineBias;          //                        offset 20
    float waterlineWidth;         //                        offset 24
    float waterlineStrength;      //                        offset 28
    float waterlineNoise;         //                        offset 32
    float wetStrength;            //                        offset 36
    float specularFactor;         //                        offset 40
    float bumpFactor;             //                        offset 44
} ModelUniforms;                  // 48 bytes total
