import Foundation
import simd
import Observation

struct WaveState {
    var amplitude: Float
    var wavelength: Float
    var steepness: Float
    var phase: Float
    var direction: SIMD2<Float>  // normalised 2D direction
}

@Observable
@MainActor
final class UniformState {
    // Waves
    var waves: [WaveState] = []

    // Surface
    var baseColor: SIMD4<Float>         = SIMD4<Float>(0, 0.04, 0.03, 1)
    var lightPos: SIMD3<Float>          = SIMD3<Float>(50, 80, 30)
    var fogDensity: Float               = 0.01
    var fogColor: SIMD3<Float>          = SIMD3<Float>(0.4, 0.6, 0.7)
    var causticScale: Float             = 2.0
    var causticSpeed: Float             = 0.3
    var causticIntensity: Float         = 1.0
    var causticColor: SIMD3<Float>      = SIMD3<Float>(1, 0.9, 0.7)
    var causticTroughMin: Float         = -0.6
    var causticTroughMax: Float         = 0.1
    var causticThresholdMin: Float      = 0.55
    var causticThresholdMax: Float      = 0.8
    var causticSharpness: Float         = 6.0
    var foamScale: Float                = 0.4
    var foamScrollSpeed: Float          = 0.001
    var foamSlopeMin: Float             = 0.71
    var foamSlopeMax: Float             = 1.97
    var foamSlopeAmplifier: Float       = 72.68
    var foamPower: Float                = 0.5
    var depthFadeNear: Float            = 10.0
    var depthFadeFar: Float             = 80.0
    var deepWaterTint: SIMD3<Float>     = SIMD3<Float>(0.1, 0.2, 0.3)
    var fresnelF0: Float                = 0.04
    var gamma: Float                    = 0.45
    var reflectionDistortion: Float     = 0.03

    // Gust
    var gustDirection: SIMD2<Float>     = SIMD2<Float>(1, 0)
    var gustSpeed: Float                = 0.008
    var gustScale: Float                = 0.009
    var gustStrength: Float             = 0.008

    // Normal mapping
    var normalMappingScale: Float       = 1.0
    var normalMappingSpeed: Float       = 0.0
    var normalMappingDirection: SIMD2<Float> = SIMD2<Float>(1, 1)

    // Buoy
    var buoyHullDisplacement: Float     = 0.04
    var buoyWaterlineBias: Float        = 0.15
    var buoyWaterlineWidth: Float       = 0.5
    var buoyWaterlineStrength: Float    = 0.45
    var buoyWaterlineNoise: Float       = 0.8
    var buoyWetStrength: Float          = 0.5
    var buoySpecularFactor: Float       = 1.0
    var buoyBumpFactor: Float           = 0.5

    let animator = UniformAnimator()

    init() {
        // Default 10 zero waves — will be overwritten by loadUniforms
        waves = (0..<10).map { _ in
            WaveState(amplitude: 0, wavelength: 30, steepness: 0.5,
                      phase: 0, direction: SIMD2<Float>(1, 0))
        }
        loadUniforms()
    }

    func tick(time: Float) {
        animator.tick(time: time)
    }

    func buildSceneUniforms(time: Float,
                             model: simd_float4x4,
                             view: simd_float4x4,
                             projection: simd_float4x4,
                             reflectionMatrix: simd_float4x4,
                             cameraPos: SIMD3<Float>,
                             isReflectionPass: Bool = false) -> SceneUniforms {
        var u = SceneUniforms()
        u.modelMatrix       = model
        u.viewMatrix        = view
        u.projectionMatrix  = projection
        u.reflectionMatrix  = reflectionMatrix
        u.cameraPos         = cameraPos
        u.time              = time
        u.isReflectionPass  = isReflectionPass ? 1 : 0
        let numWaves = min(waves.count, Int(MAX_WAVES))
        u.numWaves = Int32(numWaves)

        withUnsafeMutablePointer(to: &u.waves) { ptr in
            let wavePtr = UnsafeMutableRawPointer(ptr).bindMemory(to: WaveUniform.self, capacity: Int(MAX_WAVES))
            for i in 0..<numWaves {
                wavePtr[i].direction  = waves[i].direction
                wavePtr[i].amplitude  = waves[i].amplitude
                wavePtr[i].wavelength = waves[i].wavelength
                wavePtr[i].steepness  = waves[i].steepness
                wavePtr[i].phase      = waves[i].phase
                wavePtr[i]._pad0      = 0
                wavePtr[i]._pad1      = 0
            }
        }
        return u
    }

    func buildSurfaceUniforms() -> SurfaceUniforms {
        var s = SurfaceUniforms()
        s.gustDirection           = gustDirection
        s.gustSpeed               = gustSpeed
        s.gustScale               = gustScale
        s.gustStrength            = gustStrength
        s.normalMappingScale      = normalMappingScale
        s.normalMappingSpeed      = normalMappingSpeed
        s.normalMappingDirection  = normalMappingDirection
        s.lightPos                = lightPos
        s.fogDensity              = fogDensity
        s.fogColor                = fogColor
        s.causticScale            = causticScale
        s.causticSpeed            = causticSpeed
        s.causticIntensity        = causticIntensity
        s.causticColor            = causticColor
        s.causticTroughMin        = causticTroughMin
        s.causticTroughMax        = causticTroughMax
        s.causticThresholdMin     = causticThresholdMin
        s.causticThresholdMax     = causticThresholdMax
        s.causticSharpness        = causticSharpness
        s.foamScale               = foamScale
        s.foamScrollSpeed         = foamScrollSpeed
        s.foamSlopeMin            = foamSlopeMin
        s.foamSlopeMax            = foamSlopeMax
        s.foamSlopeAmplifier      = foamSlopeAmplifier
        s.foamPower               = foamPower
        s.depthFadeNear           = depthFadeNear
        s.depthFadeFar            = depthFadeFar
        s.deepWaterTint           = deepWaterTint
        s.fresnelF0               = fresnelF0
        s.gamma                   = gamma
        s.reflectionDistortion    = reflectionDistortion
        s.baseColor               = baseColor
        return s
    }

    func buildBuoyUniforms() -> BuoyUniforms {
        var b = BuoyUniforms()
        b.hullDisplacementAmt = buoyHullDisplacement
        b.waterlineBias       = buoyWaterlineBias
        b.waterlineWidth      = buoyWaterlineWidth
        b.waterlineStrength   = buoyWaterlineStrength
        b.waterlineNoise      = buoyWaterlineNoise
        b.wetStrength         = buoyWetStrength
        b.specularFactor      = buoySpecularFactor
        b.bumpFactor          = buoyBumpFactor
        return b
    }

    // MARK: - JSON loading

    private func loadUniforms() {
        guard let url = Bundle.main.url(forResource: "uniforms", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }

        let duration = json["duration"] as? Float ?? 0
        let animatePatterns = json["animate"] as? [String] ?? []

        for (key, rawValue) in json {
            if key == "duration" || key == "animate" { continue }
            apply(key: key, rawValue: rawValue, duration: duration, animatePatterns: animatePatterns)
        }
    }

    private func apply(key: String, rawValue: Any, duration: Float, animatePatterns: [String]) {
        // waves[N].param
        if let (idx, param) = parseWaveKey(key) {
            guard idx < waves.count else { return }
            switch param {
            case "amplitude":
                if let v = floatValue(rawValue) {
                    let shouldAnimate = duration > 0 && matchesAnyPattern(animatePatterns, key)
                    if shouldAnimate {
                        let from = waves[idx].amplitude
                        animator.animateTo(.scalar(from), .scalar(v), duration: duration,
                                           apply: { [weak self] val in
                            if case .scalar(let f) = val { self?.waves[idx].amplitude = f }
                        }, key: key)
                    } else {
                        waves[idx].amplitude = v
                    }
                }
            case "wavelength":  if let v = floatValue(rawValue) { waves[idx].wavelength = v }
            case "steepness":   if let v = floatValue(rawValue) { waves[idx].steepness  = v }
            case "phase":       if let v = floatValue(rawValue) { waves[idx].phase       = v }
            case "direction":
                if let arr = rawValue as? [Double], arr.count >= 2 {
                    waves[idx].direction = SIMD2<Float>(Float(arr[0]), Float(arr[1]))
                }
            default: break
            }
            return
        }

        switch key {
        case "mesh_shader.baseColor":
            if let arr = float4Value(rawValue) { baseColor = arr }
        case "mesh_shader.lightPos":
            if let arr = float3Value(rawValue) { lightPos = arr }
        case "mesh_shader.fogDensity":
            if let v = floatValue(rawValue) { fogDensity = v }
        case "mesh_shader.fogColor":
            if let arr = float3Value(rawValue) { fogColor = arr }
        case "mesh_shader.causticIntensity":
            if let v = floatValue(rawValue) { causticIntensity = v }
        case "mesh_shader.causticColor":
            if let arr = float3Value(rawValue) { causticColor = arr }
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
            if let arr = float3Value(rawValue) { deepWaterTint = arr }
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
        case "mesh_shader.gust.speed":
            if let v = floatValue(rawValue) { gustSpeed = v }
        case "mesh_shader.gust.scale":
            if let v = floatValue(rawValue) { gustScale = v }
        case "mesh_shader.gust.strength":
            if let v = floatValue(rawValue) { gustStrength = v }
        case "mesh_shader.normalMapping.scale":
            if let v = floatValue(rawValue) { normalMappingScale = v }
        case "mesh_shader.normalMapping.speed":
            if let v = floatValue(rawValue) { normalMappingSpeed = v }
        case "mesh_shader.normalMapping.direction":
            if let arr = rawValue as? [Double], arr.count >= 2 {
                normalMappingDirection = SIMD2<Float>(Float(arr[0]), Float(arr[1]))
            }
        case "buoy_mesh.hullDisplacementAmt":
            if let v = floatValue(rawValue) { buoyHullDisplacement = v }
        case "buoy_mesh.waterlineBias":
            if let v = floatValue(rawValue) { buoyWaterlineBias = v }
        case "buoy_mesh.waterlineWidth":
            if let v = floatValue(rawValue) { buoyWaterlineWidth = v }
        case "buoy_mesh.waterlineStrength":
            if let v = floatValue(rawValue) { buoyWaterlineStrength = v }
        case "buoy_mesh.waterlineNoise":
            if let v = floatValue(rawValue) { buoyWaterlineNoise = v }
        case "buoy_mesh.wetStrength":
            if let v = floatValue(rawValue) { buoyWetStrength = v }
        case "buoy_mesh.specularFactor":
            if let v = floatValue(rawValue) { buoySpecularFactor = v }
        case "buoy_mesh.bumpFactor":
            if let v = floatValue(rawValue) { buoyBumpFactor = v }
        default: break
        }
    }

    // MARK: - Helpers

    private func parseWaveKey(_ key: String) -> (Int, String)? {
        guard key.hasPrefix("waves[") else { return nil }
        guard let closeBracket = key.firstIndex(of: "]") else { return nil }
        let idxStart = key.index(key.startIndex, offsetBy: 6)
        guard let idx = Int(key[idxStart..<closeBracket]) else { return nil }
        let afterBracket = key.index(closeBracket, offsetBy: 2)
        guard afterBracket < key.endIndex else { return nil }
        return (idx, String(key[afterBracket...]))
    }

    private func floatValue(_ v: Any) -> Float? {
        if let d = v as? Double { return Float(d) }
        if let f = v as? Float { return f }
        if let i = v as? Int { return Float(i) }
        return nil
    }

    private func float3Value(_ v: Any) -> SIMD3<Float>? {
        guard let arr = v as? [Double], arr.count >= 3 else { return nil }
        return SIMD3<Float>(Float(arr[0]), Float(arr[1]), Float(arr[2]))
    }

    private func float4Value(_ v: Any) -> SIMD4<Float>? {
        guard let arr = v as? [Double], arr.count >= 4 else { return nil }
        return SIMD4<Float>(Float(arr[0]), Float(arr[1]), Float(arr[2]), Float(arr[3]))
    }

    private func matchesAnyPattern(_ patterns: [String], _ key: String) -> Bool {
        patterns.contains { matchesPattern($0, key) }
    }

    private func matchesPattern(_ pattern: String, _ key: String) -> Bool {
        var ki = key.startIndex
        var pi = pattern.startIndex
        var starPos: String.Index? = nil
        var matchPos = key.startIndex

        while ki < key.endIndex {
            if pi < pattern.endIndex && pattern[pi] == "*" {
                starPos = pi
                pattern.formIndex(after: &pi)
                matchPos = ki
            } else if pi < pattern.endIndex && pattern[pi] == key[ki] {
                pattern.formIndex(after: &pi)
                key.formIndex(after: &ki)
            } else if let sp = starPos {
                pi = pattern.index(after: sp)
                key.formIndex(after: &matchPos)
                ki = matchPos
            } else {
                return false
            }
        }
        while pi < pattern.endIndex && pattern[pi] == "*" { pattern.formIndex(after: &pi) }
        return pi == pattern.endIndex
    }
}
