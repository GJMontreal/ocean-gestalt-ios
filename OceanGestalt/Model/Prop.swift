import Metal
import simd

/// Composes a Moveable and a Drawable. Runs per-frame wave displacement,
/// surface normal alignment, and angular dynamics (spin torque, angular
/// damping, torsional stiffness) before handing off to the drawable.
/// Mirrors src/core/Prop.cpp.
@MainActor
final class Prop {
    var moveable: Moveable
    let drawable: any Drawable

    // Physics config (loaded from scene.json)
    var rightingWeight: Float       // 0 = rigid upright, 1 = fully follows wave
    var spinTorqueScale: Float
    var angularDamping: Float
    var torsionalStiffness: Float

    // Base transform from scene.json (applied before physics rotation)
    var baseRotation: simd_quatf
    var scale: Float

    // Angular dynamics state
    private var angularVelocity: Float = 0
    private var currentYaw: Float     = 0
    private var lastTime: Float       = -1

    // Tilt state — tracks orientation incrementally so normal changes are smooth
    private var lastRotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    init(
        moveable: Moveable,
        drawable: any Drawable,
        baseRotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
        scale: Float = 1,
        rightingWeight: Float = 0.5,
        spinTorqueScale: Float = 2,
        angularDamping: Float = 1.5,
        torsionalStiffness: Float = 0.5
    ) {
        self.moveable          = moveable
        self.drawable          = drawable
        self.baseRotation      = baseRotation
        self.scale             = scale
        self.rightingWeight    = rightingWeight
        self.spinTorqueScale   = spinTorqueScale
        self.angularDamping    = angularDamping
        self.torsionalStiffness = torsionalStiffness
    }

    // MARK: - Per-frame update

    /// Evaluates wave physics and returns the world model matrix for this
    /// frame. `waveOffset` returns pure Gerstner displacement (no base
    /// position added) at a given XZ point — matches evaluateGerstnerWaves()
    /// in the C++ reference.
    func modelMatrix(time: Float, waveOffset: (SIMD2<Float>) -> SIMD3<Float>) -> simd_float4x4 {
        let xz = SIMD2<Float>(moveable.position.x, moveable.position.z)

        var displacement = SIMD3<Float>.zero
        var targetNormal = SIMD3<Float>(0, 1, 0)

        if moveable.isFloating {
            let eps: Float = 0.1
            displacement = waveOffset(xz)
            let dx = waveOffset(xz + SIMD2(eps, 0))
            let dz = waveOffset(xz + SIMD2(0, eps))

            // Full surface tangents: base step + displacement delta.
            // cross(Z-tangent, X-tangent) gives upward normal in Y-up convention.
            let tx = SIMD3<Float>(eps, 0, 0) + (dx - displacement)
            let tz = SIMD3<Float>(0, 0, eps) + (dz - displacement)
            let rawNormal = simd_cross(tz, tx)
            let len = simd_length(rawNormal)
            if len > 1e-5 {
                let waveNormal = rawNormal / len
                targetNormal = simd_normalize(simd_mix(
                    SIMD3<Float>(0, 1, 0), waveNormal, SIMD3<Float>(repeating: rightingWeight)))
            }
        }

        // Incremental tilt rotation from previous up toward targetNormal.
        let prevUp = lastRotation.act(SIMD3<Float>(0, 1, 0))
        let rAlign = simd_quatf(from: prevUp, to: targetNormal)
        let tiltRotation = simd_normalize(rAlign * lastRotation)
        lastRotation = tiltRotation

        // dt guard — skip first frame and large gaps (e.g. resume after pause)
        var dt: Float = 0
        if lastTime >= 0 && time > lastTime && (time - lastTime) < 0.5 {
            dt = time - lastTime
        }
        lastTime = time

        // Noise-based spin torque: varies smoothly with XZ position and time
        var spinTorque: Float = 0
        if moveable.isFloating {
            let phase1 = xz.x * 0.07 + time * 0.13
            let phase2 = xz.y * 0.05 + time * 0.09
            let waveStrength = simd_length(SIMD2<Float>(displacement.x, displacement.z))
            spinTorque = sin(phase1) * cos(phase2) * waveStrength
        }

        // Angular equation of motion
        let torque = spinTorque * spinTorqueScale
                   - angularDamping * angularVelocity
                   - torsionalStiffness * currentYaw
        angularVelocity += torque * dt
        currentYaw      += angularVelocity * dt

        let yawRot = simd_quatf(angle: currentYaw, axis: SIMD3<Float>(0, 1, 0))
        let physicsRotation = simd_normalize(tiltRotation * yawRot)

        // World position: base XZ + displacement, base Y offset along normal
        let worldPos = SIMD3<Float>(
            moveable.position.x + displacement.x,
            moveable.position.y + displacement.y,
            moveable.position.z + displacement.z
        )

        return translationMatrix(worldPos)
             * simd_float4x4(simd_normalize(physicsRotation * baseRotation))
             * scaleMatrix(scale)
    }

    // MARK: - Encode

    func encode(
        into encoder: MTLRenderCommandEncoder,
        time: Float,
        waveOffset: (SIMD2<Float>) -> SIMD3<Float>,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {
        let mdl = modelMatrix(time: time, waveOffset: waveOffset)
        drawable.encode(into: encoder, modelMatrix: mdl, scene: scene, surface: surface)
    }

    func encodeReflection(
        into encoder: MTLRenderCommandEncoder,
        time: Float,
        waveOffset: (SIMD2<Float>) -> SIMD3<Float>,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {
        let mdl = modelMatrix(time: time, waveOffset: waveOffset)
        drawable.encodeReflection(into: encoder, modelMatrix: mdl, scene: scene, surface: surface)
    }

    func encodeDebugMesh(
        into encoder: MTLRenderCommandEncoder,
        time: Float,
        waveOffset: (SIMD2<Float>) -> SIMD3<Float>,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {
        let mdl = modelMatrix(time: time, waveOffset: waveOffset)
        drawable.encodeDebugMesh(into: encoder, modelMatrix: mdl, scene: scene, surface: surface)
    }

    // MARK: - Matrix helpers

    private func translationMatrix(_ t: SIMD3<Float>) -> simd_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4<Float>(t.x, t.y, t.z, 1)
        return m
    }

    private func scaleMatrix(_ s: Float) -> simd_float4x4 {
        simd_float4x4(diagonal: SIMD4<Float>(s, s, s, 1))
    }
}
