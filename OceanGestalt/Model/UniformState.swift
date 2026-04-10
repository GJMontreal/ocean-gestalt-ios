import Foundation
import simd
import Observation

struct WaveState {
    var amplitude:  Float
    var wavelength: Float
    var steepness:  Float
    var phase:      Float
    var direction:  SIMD2<Float>  // normalised 2D; loaded from [x,y,z] in JSON (z ignored)
}

/// All tweakable rendering parameters. Loads from uniforms.json in the app
/// bundle. Mirrors the key/value system in Configuration / UniformAnimator.
/// Mirrors src/api/UniformState.cpp and src/core/Configuration.cpp.
@Observable
@MainActor
final class UniformState {

    // MARK: - Waves (10 slots matching MAX_WAVES in ShaderTypes.h)

    var waves: [WaveState] = (0..<10).map { _ in
        WaveState(amplitude: 0, wavelength: 30, steepness: 0.5,
                  phase: 0, direction: SIMD2<Float>(1, 0))
    }

    // MARK: - Surface appearance

    var baseColor: SIMD4<Float>        = SIMD4<Float>(0, 0.04, 0.03, 1)
    var lightPos: SIMD3<Float>         = SIMD3<Float>(50, 80, 30)   // overwritten by scene.json
    var fogDensity: Float              = 0.01
    var fogColor: SIMD3<Float>         = SIMD3<Float>(0.4, 0.6, 0.7)
    var causticScale: Float            = 2.0
    var causticSpeed: Float            = 0.3
    var causticIntensity: Float        = 1.0
    var causticColor: SIMD3<Float>     = SIMD3<Float>(1, 0.9, 0.7)
    var causticTroughMin: Float        = -0.6
    var causticTroughMax: Float        = 0.1
    var causticThresholdMin: Float     = 0.55
    var causticThresholdMax: Float     = 0.8
    var causticSharpness: Float        = 6.0
    var foamScale: Float               = 0.4
    var foamScrollSpeed: Float         = 0.001
    var foamSlopeMin: Float            = 0.71
    var foamSlopeMax: Float            = 1.97
    var foamSlopeAmplifier: Float      = 72.68
    var foamPower: Float               = 0.5
    var depthFadeNear: Float           = 10.0
    var depthFadeFar: Float            = 80.0
    var deepWaterTint: SIMD3<Float>    = SIMD3<Float>(0.1, 0.2, 0.3)
    var fresnelF0: Float               = 0.04
    var gamma: Float                   = 0.45
    var reflectionDistortion: Float    = 0.03

    // Gust
    var gustDirection: SIMD2<Float>    = SIMD2<Float>(1, 0)
    var gustSpeed: Float               = 0.008
    var gustScale: Float               = 0.009
    var gustStrength: Float            = 0.008

    // Normal mapping
    var normalMappingScale: Float      = 1.0
    var normalMappingSpeed: Float      = 0.0
    var normalMappingDir: SIMD2<Float> = SIMD2<Float>(1, 1)

    // MARK: - Animator

    let animator = UniformAnimator()
    private var lastTime: Float = -1

    // MARK: - Init

    init() {
        loadUniforms()
    }

    // MARK: - Per-frame

    func tick(time: Float) {
        var dt: Float = 0
        if lastTime >= 0 && time > lastTime && (time - lastTime) < 1 { dt = time - lastTime }
        lastTime = time
        animator.tick(deltaTime: dt)
    }

    // MARK: - Uniform struct builders

    func buildSceneUniforms(
        time: Float,
        model: simd_float4x4,
        view: simd_float4x4,
        projection: simd_float4x4,
        reflectionMatrix: simd_float4x4,
        lightSpaceMatrix: simd_float4x4 = matrix_identity_float4x4,
        cameraPos: SIMD3<Float>,
        lightPos: SIMD3<Float>,
        isReflectionPass: Bool = false
    ) -> SceneUniforms {
        var u = SceneUniforms()
        u.modelMatrix       = model
        u.viewMatrix        = view
        u.projectionMatrix  = projection
        u.reflectionMatrix  = reflectionMatrix
        u.lightSpaceMatrix  = lightSpaceMatrix
        u.cameraPos         = SIMD4<Float>(cameraPos.x, cameraPos.y, cameraPos.z, 1)
        u.lightPos          = SIMD4<Float>(lightPos.x,  lightPos.y,  lightPos.z,  1)
        u.time              = time
        u.isReflectionPass  = isReflectionPass ? 1 : 0
        u.numWaves          = Int32(min(waves.count, Int(MAX_WAVES)))
        withUnsafeMutablePointer(to: &u.waves) { ptr in
            let wp = UnsafeMutableRawPointer(ptr).bindMemory(
                to: WaveUniform.self, capacity: Int(MAX_WAVES))
            for i in 0..<Int(u.numWaves) {
                wp[i].direction  = waves[i].direction
                wp[i].amplitude  = waves[i].amplitude
                wp[i].wavelength = waves[i].wavelength
                wp[i].steepness  = waves[i].steepness
                wp[i].phase      = waves[i].phase
                wp[i]._pad0      = 0
                wp[i]._pad1      = 0
            }
        }
        return u
    }

    func buildSurfaceUniforms() -> SurfaceUniforms {
        var s = SurfaceUniforms()
        s.baseColor             = baseColor
        s.fogColor              = SIMD4<Float>(fogColor.x, fogColor.y, fogColor.z, 0)
        s.fogDensity            = fogDensity
        s.causticColor          = SIMD4<Float>(causticColor.x, causticColor.y, causticColor.z, 0)
        s.causticScale          = causticScale
        s.causticSpeed          = causticSpeed
        s.causticIntensity      = causticIntensity
        s.causticTroughMin      = causticTroughMin
        s.causticTroughMax      = causticTroughMax
        s.causticThresholdMin   = causticThresholdMin
        s.causticThresholdMax   = causticThresholdMax
        s.causticSharpness      = causticSharpness
        s.foamScale             = foamScale
        s.foamScrollSpeed       = foamScrollSpeed
        s.foamSlopeMin          = foamSlopeMin
        s.foamSlopeMax          = foamSlopeMax
        s.foamSlopeAmplifier    = foamSlopeAmplifier
        s.foamPower             = foamPower
        s.depthFadeNear         = depthFadeNear
        s.depthFadeFar          = depthFadeFar
        s.deepWaterTint         = SIMD4<Float>(deepWaterTint.x, deepWaterTint.y, deepWaterTint.z, 0)
        s.fresnelF0             = fresnelF0
        s.gamma                 = gamma
        s.reflectionDistortion  = reflectionDistortion
        s.gustDirection         = gustDirection
        s.gustSpeed             = gustSpeed
        s.gustScale             = gustScale
        s.gustStrength          = gustStrength
        s.normalMappingScale    = normalMappingScale
        s.normalMappingSpeed    = normalMappingSpeed
        s.normalMappingDir      = normalMappingDir
        return s
    }

    // MARK: - JSON loading

    private func loadUniforms() {
        guard let url = Bundle.main.url(forResource: "uniforms", withExtension: "json",
                                        subdirectory: "data/config"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }

        for (key, rawValue) in json {
            apply(key: key, rawValue: rawValue)
        }
    }

    private func apply(key: String, rawValue: Any) {
        // waves[N].param — may be plain float or {"value": X, "duration": Y}
        if let (idx, param) = parseBracketKey("waves", key), idx < waves.count {
            switch param {
            case "amplitude":
                if let (val, dur) = floatOrAnimated(rawValue) {
                    if dur > 0 {
                        let from = waves[idx].amplitude
                        animator.animateTo(from, val, duration: dur, key: key) {
                            [weak self] v in self?.waves[idx].amplitude = v
                        }
                    } else {
                        waves[idx].amplitude = val
                    }
                }
            case "wavelength": if let v = floatValue(rawValue) { waves[idx].wavelength = v }
            case "steepness":  if let v = floatValue(rawValue) { waves[idx].steepness  = v }
            case "phase":      if let v = floatValue(rawValue) { waves[idx].phase       = v }
            case "direction":
                if let arr = rawValue as? [Double], arr.count >= 2 {
                    let xy = SIMD2<Float>(Float(arr[0]), Float(arr[1]))
                    let len = simd_length(xy)
                    waves[idx].direction = len > 1e-5 ? xy / len : SIMD2<Float>(1, 0)
                }
            default: break
            }
            return
        }

        switch key {
        case "mesh_shader.baseColor":
            if let v = float4Value(rawValue) { baseColor = v }
        case "mesh_shader.fogDensity":
            if let v = floatValue(rawValue) { fogDensity = v }
        case "mesh_shader.fogColor":
            if let v = float3Value(rawValue) { fogColor = v }
        case "mesh_shader.causticIntensity":
            if let v = floatValue(rawValue) { causticIntensity = v }
        case "mesh_shader.causticColor":
            if let v = float3Value(rawValue) { causticColor = v }
        case "mesh_shader.causticScale":
            if let v = floatValue(rawValue) { causticScale = v }
        case "mesh_shader.causticSpeed":
            if let v = floatValue(rawValue) { causticSpeed = v }
        case "mesh_shader.causticTroughMin":
            if let v = floatValue(rawValue) { causticTroughMin = v }
        case "mesh_shader.causticTroughMax":
            if let v = floatValue(rawValue) { causticTroughMax = v }
        case "mesh_shader.causticThresholdMin":
            if let v = floatValue(rawValue) { causticThresholdMin = v }
        case "mesh_shader.causticThresholdMax":
            if let v = floatValue(rawValue) { causticThresholdMax = v }
        case "mesh_shader.causticSharpness":
            if let v = floatValue(rawValue) { causticSharpness = v }
        case "mesh_shader.foamScale":
            if let v = floatValue(rawValue) { foamScale = v }
        case "mesh_shader.foamScrollSpeed":
            if let v = floatValue(rawValue) { foamScrollSpeed = v }
        case "mesh_shader.foamSlopeMin":
            if let v = floatValue(rawValue) { foamSlopeMin = v }
        case "mesh_shader.foamSlopeMax":
            if let v = floatValue(rawValue) { foamSlopeMax = v }
        case "mesh_shader.foamSlopeAmplifier":
            if let v = floatValue(rawValue) { foamSlopeAmplifier = v }
        case "mesh_shader.foamPower":
            if let v = floatValue(rawValue) { foamPower = v }
        case "mesh_shader.depthFadeNear":
            if let v = floatValue(rawValue) { depthFadeNear = v }
        case "mesh_shader.depthFadeFar":
            if let v = floatValue(rawValue) { depthFadeFar = v }
        case "mesh_shader.deepWaterTint":
            if let v = float3Value(rawValue) { deepWaterTint = v }
        case "mesh_shader.fresnelF0":
            if let v = floatValue(rawValue) { fresnelF0 = v }
        case "mesh_shader.gamma":
            if let v = floatValue(rawValue) { gamma = v }
        case "mesh_shader.reflectionDistortion":
            if let v = floatValue(rawValue) { reflectionDistortion = v }
        case "mesh_shader.gust.direction":
            if let arr = rawValue as? [Double], arr.count >= 2 {
                gustDirection = SIMD2<Float>(Float(arr[0]), Float(arr[1]))
            }
        case "mesh_shader.gust.speed":    if let v = floatValue(rawValue) { gustSpeed    = v }
        case "mesh_shader.gust.scale":    if let v = floatValue(rawValue) { gustScale    = v }
        case "mesh_shader.gust.strength": if let v = floatValue(rawValue) { gustStrength = v }
        case "mesh_shader.normalMapping.scale":
            if let v = floatValue(rawValue) { normalMappingScale = v }
        case "mesh_shader.normalMapping.speed":
            if let v = floatValue(rawValue) { normalMappingSpeed = v }
        case "mesh_shader.normalMapping.direction":
            if let arr = rawValue as? [Double], arr.count >= 2 {
                normalMappingDir = SIMD2<Float>(Float(arr[0]), Float(arr[1]))
            }
        default: break
        }
    }

    // MARK: - Helpers

    /// Parses "prefix[N].param" → (N, "param")
    private func parseBracketKey(_ prefix: String, _ key: String) -> (Int, String)? {
        let open = prefix + "["
        guard key.hasPrefix(open),
              let close = key.firstIndex(of: "]") else { return nil }
        let idxStart = key.index(key.startIndex, offsetBy: open.count)
        guard let idx = Int(key[idxStart..<close]) else { return nil }
        let afterClose = key.index(close, offsetBy: 2)
        guard afterClose < key.endIndex else { return nil }
        return (idx, String(key[afterClose...]))
    }

    /// Returns (value, duration) — duration is 0 for plain-float entries.
    private func floatOrAnimated(_ v: Any) -> (Float, Float)? {
        if let f = floatValue(v) { return (f, 0) }
        if let d = v as? [String: Any],
           let val = d["value"].flatMap(floatValue),
           let dur = d["duration"].flatMap(floatValue) { return (val, dur) }
        return nil
    }

    private func floatValue(_ v: Any) -> Float? {
        if let d = v as? Double { return Float(d) }
        if let f = v as? Float  { return f }
        if let i = v as? Int    { return Float(i) }
        return nil
    }

    private func float3Value(_ v: Any) -> SIMD3<Float>? {
        guard let a = v as? [Double], a.count >= 3 else { return nil }
        return SIMD3<Float>(Float(a[0]), Float(a[1]), Float(a[2]))
    }

    private func float4Value(_ v: Any) -> SIMD4<Float>? {
        guard let a = v as? [Double], a.count >= 4 else { return nil }
        return SIMD4<Float>(Float(a[0]), Float(a[1]), Float(a[2]), Float(a[3]))
    }
}
