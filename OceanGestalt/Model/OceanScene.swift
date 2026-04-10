import Metal
import simd

// MARK: - Scene data types (decoded from scene.json)

struct CameraConfig {
    var position: SIMD3<Float>
    var yaw: Float      // degrees
    var pitch: Float    // degrees
    var zoom: Float     // degrees (field of view)
}

struct ModelConfig {
    var name: String
    var file: String
    var shader: String
    var position: SIMD3<Float>
    var rotation: SIMD3<Float>  // Euler degrees (X, Y, Z)
    var scale: Float
    var floating: Bool
    var tethered: Bool
    var rightingWeight: Float
    var spinTorqueScale: Float
    var angularDamping: Float
    var torsionalStiffness: Float
    var waterlineWidth: Float?
}

struct MeshConfig {
    var size: Int
    var subdivisions: Int
}

// MARK: - Scene

/// Loads scene.json and owns the Prop list. Nothing about rendering pipelines
/// lives here — each Prop holds a Drawable that handles that.
/// Mirrors src/core/include/Configuration.hpp (sceneModels, camera, light, mesh).
@MainActor
final class OceanScene {
    let camera: CameraConfig
    let lightPosition: SIMD3<Float>
    let mesh: MeshConfig
    let reflectionSize: Int
    let shadowSize: Int

    /// The floating/visible props loaded from scene.json "models".
    let props: [Prop]

    /// Factory closure: given a ModelConfig and device, produce a Drawable.
    /// Allows GLTF loading to be wired in independently of scene parsing.
    typealias DrawableFactory = (ModelConfig, MTLDevice) throws -> any Drawable

    init(device: MTLDevice, drawableFactory: DrawableFactory) throws {
        guard let url = Bundle.main.url(forResource: "scene", withExtension: "json",
                                        subdirectory: "data/config"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw OceanSceneError.fileNotFound
        }

        camera       = try OceanScene.parseCamera(json)
        lightPosition = try OceanScene.parseLight(json)
        mesh         = try OceanScene.parseMesh(json)
        reflectionSize = (json["reflection"] as? [String: Any]).flatMap { $0["size"] as? Int } ?? 512
        shadowSize     = (json["shadow"]     as? [String: Any]).flatMap { $0["size"] as? Int } ?? 2048

        let modelConfigs = try OceanScene.parseModels(json)
        props = try modelConfigs.map { config in
            let drawable = try drawableFactory(config, device)
            let moveable = Moveable(
                name: config.name,
                position: config.position,
                isFloating: config.floating,
                isTethered: config.tethered
            )
            return Prop(
                moveable: moveable,
                drawable: drawable,
                baseRotation: simd_quatf(eulerDegrees: config.rotation),
                scale: config.scale,
                rightingWeight: config.rightingWeight,
                spinTorqueScale: config.spinTorqueScale,
                angularDamping: config.angularDamping,
                torsionalStiffness: config.torsionalStiffness
            )
        }
    }

    // MARK: - Per-frame encode

    func encode(
        into encoder: MTLRenderCommandEncoder,
        time: Float,
        waveOffset: (SIMD2<Float>) -> SIMD3<Float>,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {
        for prop in props {
            prop.encode(into: encoder, time: time, waveOffset: waveOffset,
                        scene: scene, surface: surface)
        }
    }

    func encodeReflection(
        into encoder: MTLRenderCommandEncoder,
        time: Float,
        waveOffset: (SIMD2<Float>) -> SIMD3<Float>,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {
        for prop in props {
            prop.encodeReflection(into: encoder, time: time, waveOffset: waveOffset,
                                  scene: scene, surface: surface)
        }
    }

    // MARK: - JSON parsing

    private static func parseCamera(_ json: [String: Any]) throws -> CameraConfig {
        guard let cam = json["camera"] as? [String: Any] else { throw OceanSceneError.missingKey("camera") }
        return CameraConfig(
            position: try parseVec3(cam, key: "position"),
            yaw:   (cam["yaw"]   as? Double).map(Float.init) ?? -90,
            pitch: (cam["pitch"] as? Double).map(Float.init) ?? 0,
            zoom:  (cam["zoom"]  as? Double).map(Float.init) ?? 45
        )
    }

    private static func parseLight(_ json: [String: Any]) throws -> SIMD3<Float> {
        guard let l = json["light"] as? [String: Any] else { throw OceanSceneError.missingKey("light") }
        return SIMD3<Float>(
            (l["x"] as? Double).map(Float.init) ?? 0,
            (l["y"] as? Double).map(Float.init) ?? 0,
            (l["z"] as? Double).map(Float.init) ?? 0
        )
    }

    private static func parseMesh(_ json: [String: Any]) throws -> MeshConfig {
        guard let m = json["mesh"] as? [String: Any] else { throw OceanSceneError.missingKey("mesh") }
        return MeshConfig(
            size:         (m["size"]         as? Int) ?? 60,
            subdivisions: (m["subdivisions"] as? Int) ?? 2
        )
    }

    private static func parseModels(_ json: [String: Any]) throws -> [ModelConfig] {
        guard let models = json["models"] as? [[String: Any]] else { return [] }
        return try models.map { m in
            guard let name   = m["name"]   as? String,
                  let file   = m["file"]   as? String,
                  let shader = m["shader"] as? String
            else { throw OceanSceneError.malformedModel }
            return ModelConfig(
                name:               name,
                file:               file,
                shader:             shader,
                position:           (try? parseVec3(m, key: "position")) ?? .zero,
                rotation:           (try? parseVec3(m, key: "rotation")) ?? .zero,
                scale:              (m["scale"]              as? Double).map(Float.init) ?? 1,
                floating:           (m["floating"]           as? Bool)   ?? true,
                tethered:           (m["tethered"]           as? Bool)   ?? false,
                rightingWeight:     (m["rightingWeight"]     as? Double).map(Float.init) ?? 0.5,
                spinTorqueScale:    (m["spinTorqueScale"]    as? Double).map(Float.init) ?? 2,
                angularDamping:     (m["angularDamping"]     as? Double).map(Float.init) ?? 1.5,
                torsionalStiffness: (m["torsionalStiffness"] as? Double).map(Float.init) ?? 0.5,
                waterlineWidth:     (m["waterlineWidth"]     as? Double).map(Float.init)
            )
        }
    }

    /// Parses {"x": …, "y": …, "z": …} or [x, y, z] as SIMD3<Float>.
    private static func parseVec3(_ dict: [String: Any], key: String) throws -> SIMD3<Float> {
        if let sub = dict[key] as? [String: Any] {
            return SIMD3<Float>(
                (sub["x"] as? Double).map(Float.init) ?? 0,
                (sub["y"] as? Double).map(Float.init) ?? 0,
                (sub["z"] as? Double).map(Float.init) ?? 0
            )
        }
        if let arr = dict[key] as? [Double], arr.count >= 3 {
            return SIMD3<Float>(Float(arr[0]), Float(arr[1]), Float(arr[2]))
        }
        throw OceanSceneError.missingKey(key)
    }
}

// MARK: - Errors

enum OceanSceneError: Error {
    case fileNotFound
    case missingKey(String)
    case malformedModel
}

// MARK: - simd_quatf from Euler degrees

private extension simd_quatf {
    /// Initialises from Euler angles in degrees (X pitch, Y yaw, Z roll),
    /// applied in Y → X → Z order matching the C++ reference convention.
    init(eulerDegrees v: SIMD3<Float>) {
        let toRad = Float.pi / 180
        let qX = simd_quatf(angle: v.x * toRad, axis: SIMD3<Float>(1, 0, 0))
        let qY = simd_quatf(angle: v.y * toRad, axis: SIMD3<Float>(0, 1, 0))
        let qZ = simd_quatf(angle: v.z * toRad, axis: SIMD3<Float>(0, 0, 1))
        self = simd_normalize(qY * qX * qZ)
    }
}
