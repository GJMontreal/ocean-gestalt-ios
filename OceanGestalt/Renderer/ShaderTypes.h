#pragma once
#include <simd/simd.h>

// Shared C structs between Swift and Metal.
// simd_float3 fields must sit at 16-byte-aligned offsets — add explicit
// padding when needed and verify alignment after any struct change.
