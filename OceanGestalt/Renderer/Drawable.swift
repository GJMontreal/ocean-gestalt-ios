import Metal
import simd

/// A scene object that encodes its own draw calls. Owns mesh buffers and
/// holds a reference to its pipeline state(s).
/// Mirrors the draw() / preDraw interface in src/core/include/Drawable.hpp.
protocol Drawable: AnyObject {
    /// Encode draw calls for the main pass.
    func encode(
        into encoder: MTLRenderCommandEncoder,
        modelMatrix: simd_float4x4,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    )

    /// Encode draw calls for the reflection pass. Default: no-op.
    func encodeReflection(
        into encoder: MTLRenderCommandEncoder,
        modelMatrix: simd_float4x4,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    )

    /// Encode a full-body wireframe overlay for diagnostic use ('m' key).
    /// Default: no-op. Override in concrete drawables to draw edge geometry.
    func encodeDebugMesh(
        into encoder: MTLRenderCommandEncoder,
        modelMatrix: simd_float4x4,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    )
}

extension Drawable {
    func encodeReflection(
        into encoder: MTLRenderCommandEncoder,
        modelMatrix: simd_float4x4,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {}

    func encodeDebugMesh(
        into encoder: MTLRenderCommandEncoder,
        modelMatrix: simd_float4x4,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {}
}
