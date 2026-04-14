import Metal
import MetalKit
import simd
import GestureCamera

// ---------------------------------------------------------------------------
// Renderer — sequences the reflection pre-pass, main pass (ocean + skybox +
// props), and per-frame CPU work (uniforms, camera floating, audio).
// Owns the command queue, passes, scene, and uniform state.
// ---------------------------------------------------------------------------

@MainActor
final class OceanRenderer {

    // MARK: - Public

    let device:     MTLDevice
    var isRunning:  Bool = true
    var audio:      SurfAudio? = nil

    /// 'm' key: when false draws all props as full-body wireframe instead of shaded.
    var showMesh:    Bool = true
    /// 'n' key: when true draws yellow surface-normal lines on the ocean mesh.
    var showNormals: Bool = false

    var initialCameraTransform: CameraTransform { .identity }

    // MARK: - Private

    private let commandQueue: MTLCommandQueue
    private let oceanPass:    OceanPass
    private let skyboxPass:   SkyboxPass
    private let scene:        OceanScene
    private let uniforms:     UniformState

    private static let timeWrapWindow: Float = 86400
    private var elapsedTime:  Float = 0
    private var lastTimestamp: CFTimeInterval = -1

    // MARK: - Init

    init(
        device: MTLDevice,
        colorPixelFormat: MTLPixelFormat = .bgra8Unorm,
        depthPixelFormat: MTLPixelFormat = .depth32Float
    ) throws {
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw RendererError.initFailed("command queue")
        }
        commandQueue = queue

        guard let library = device.makeDefaultLibrary() else {
            throw RendererError.initFailed("default library")
        }

        uniforms = UniformState()

        // Load the shared cubemap used by both OceanPass and SkyboxPass.
        let loader = MTKTextureLoader(device: device)
        let envMap = try OceanPass.loadCubemap(device: device, loader: loader)

        // Drawable factory: captures library, pixel formats, and shared env map.
        let colorFmt = colorPixelFormat
        let depthFmt = depthPixelFormat
        let drawableFactory: OceanScene.DrawableFactory = { config, dev in
            guard let url = Bundle.main.url(
                forResource: (config.file as NSString).deletingPathExtension,
                withExtension: (config.file as NSString).pathExtension,
                subdirectory: "data/models")
            else { throw RendererError.assetNotFound(config.file) }
            return try ModelDrawable(
                device: dev, url: url, config: config,
                library: library,
                colorPixelFormat: colorFmt,
                depthPixelFormat: depthFmt,
                envMap: envMap
            )
        }
        scene = try OceanScene(device: device, drawableFactory: drawableFactory)

        // OceanPass: owns ocean pipelines, ocean-specific textures, mesh, reflection pass.
        oceanPass = try OceanPass(
            device: device,
            library: library,
            colorPixelFormat: colorPixelFormat,
            depthPixelFormat: depthPixelFormat,
            meshConfig: scene.mesh,
            reflectionSize: scene.reflectionSize,
            envMap: envMap
        )

        // SkyboxPass: takes the shared env map and clamp sampler from OceanPass.
        skyboxPass = try SkyboxPass(
            device: device,
            library: library,
            colorPixelFormat: colorPixelFormat,
            depthPixelFormat: depthPixelFormat,
            envMap: envMap,
            clampSampler: oceanPass.clampSampler
        )

        uniforms.lightPos = scene.lightPosition

        audio = SurfAudio()
        audio?.start()
    }

    // MARK: - Per-frame draw

    func draw(view: MTKView, camera: CameraTransform) {
        // Time
        let now = CACurrentMediaTime()
        if lastTimestamp >= 0 {
            let dt = Float(now - lastTimestamp)
            if isRunning && dt > 0 && dt < 1 { elapsedTime += dt }
        }
        lastTimestamp = now
        let time = elapsedTime.truncatingRemainder(dividingBy: OceanRenderer.timeWrapWindow)
        uniforms.tick(time: time)

        let yaw = atan2(camera.forward.x, camera.forward.z)
        audio?.generateSurf(waves: uniforms.waves, position: camera.position, yaw: yaw, time: time)

        // Camera wave floating
        let cameraXZ  = SIMD2<Float>(camera.position.x, camera.position.z)
        var floatingCamera = camera
        floatingCamera.position.y = camera.position.y + gerstnerOffset(xz: cameraXZ, time: time).y

        let size   = view.drawableSize
        let aspect = Float(size.width / size.height)
        let fovY   = scene.camera.zoom * Float.pi / 180
        let proj   = perspectiveMatrix(fovY: fovY, aspect: aspect, nearZ: 0.1, farZ: 500)

        let reflectY:        simd_float4x4 = simd_float4x4(diagonal: SIMD4<Float>(1, -1, 1, 1))
        let reflectedView    = floatingCamera.viewMatrix * reflectY
        let reflectionMatrix = proj * reflectedView

        let lightPos         = uniforms.lightPos
        let halfSize         = Float(scene.mesh.size) * 0.5
        let lightProj        = orthographicMatrix(left: -halfSize, right: halfSize,
                                                  bottom: -halfSize, top: halfSize,
                                                  near: 0.1, far: 50)
        let lightView        = lookAtMatrix(eye: lightPos, target: .zero, up: SIMD3<Float>(0,1,0))
        let lightSpaceMatrix = lightProj * lightView

        let surfaceU = uniforms.buildSurfaceUniforms()
        let waveOff: (SIMD2<Float>) -> SIMD3<Float> = { [unowned self] xz in
            self.gerstnerOffset(xz: xz, time: time)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { return }
        cmdBuf.label = "OceanFrame"

        // ---- Reflection pre-pass ----
        if let reflRPD = oceanPass.reflectionPassDescriptor {
            let reflSceneU = uniforms.buildSceneUniforms(
                time: time, model: matrix_identity_float4x4,
                view: reflectedView, projection: proj,
                reflectionMatrix: reflectionMatrix,
                lightSpaceMatrix: lightSpaceMatrix,
                cameraPos: floatingCamera.position, lightPos: lightPos,
                isReflectionPass: true
            )
            if let enc = cmdBuf.makeRenderCommandEncoder(descriptor: reflRPD) {
                enc.label = "ReflectionPass"
                skyboxPass.encode(into: enc, scene: reflSceneU)
                scene.encodeReflection(into: enc, time: time, waveOffset: waveOff,
                                       scene: reflSceneU, surface: surfaceU)
                enc.endEncoding()
            }
        }

        guard let metalLayer = view.layer as? CAMetalLayer,
              let drawable   = metalLayer.nextDrawable() else { return }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture     = drawable.texture
        rpd.colorAttachments[0].loadAction  = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor  = view.clearColor
        if let depth = view.depthStencilTexture {
            rpd.depthAttachment.texture     = depth
            rpd.depthAttachment.loadAction  = .clear
            rpd.depthAttachment.storeAction = .dontCare
            rpd.depthAttachment.clearDepth  = 1.0
        }

        let sceneU = uniforms.buildSceneUniforms(
            time: time, model: matrix_identity_float4x4,
            view: floatingCamera.viewMatrix, projection: proj,
            reflectionMatrix: reflectionMatrix,
            cameraPos: floatingCamera.position, lightPos: lightPos
        )

        if let enc = cmdBuf.makeRenderCommandEncoder(descriptor: rpd) {
            enc.label = "MainPass"

            skyboxPass.encode(into: enc, scene: sceneU)

            oceanPass.encode(into: enc, scene: sceneU, surface: surfaceU,
                             showMesh: showMesh, showNormals: showNormals, time: time)

            if showMesh {
                scene.encode(into: enc, time: time, waveOffset: waveOff,
                             scene: sceneU, surface: surfaceU)
            } else {
                scene.encodeDebugMesh(into: enc, time: time, waveOffset: waveOff,
                                      scene: sceneU, surface: surfaceU)
            }

            enc.endEncoding()
        }

        cmdBuf.present(drawable)
        cmdBuf.commit()
    }

    // MARK: - CPU Gerstner (mirrors waveOffset in OceanShaders.metal / GerstnerWave.cpp)

    private func gerstnerOffset(xz: SIMD2<Float>, time: Float) -> SIMD3<Float> {
        var offset = SIMD3<Float>.zero
        for wave in uniforms.waves where wave.amplitude > 0 {
            let k = 2 * Float.pi / max(wave.wavelength, 0.01)
            let w = sqrt(9.81 * k)
            let len = simd_length(wave.direction)
            guard len > 1e-5 else { continue }
            let D = wave.direction / len
            let phase = simd_dot(D * k, xz)
                      - (w * time).truncatingRemainder(dividingBy: 2 * Float.pi)
                      + wave.phase
            offset.y += wave.amplitude * cos(phase)
            let xzDisp = -wave.steepness * D * sin(phase) * wave.amplitude
            offset.x += xzDisp.x
            offset.z += xzDisp.y
        }
        return offset
    }

    // MARK: - Matrix helpers

    private func perspectiveMatrix(fovY: Float, aspect: Float,
                                   nearZ: Float, farZ: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY * 0.5)
        let xs = ys / aspect
        let zs = farZ / (nearZ - farZ)
        return simd_float4x4(columns: (
            SIMD4<Float>(xs,  0,  0,  0),
            SIMD4<Float>( 0, ys,  0,  0),
            SIMD4<Float>( 0,  0, zs, -1),
            SIMD4<Float>( 0,  0, zs * nearZ, 0)
        ))
    }

    private func orthographicMatrix(left: Float, right: Float,
                                    bottom: Float, top: Float,
                                    near: Float, far: Float) -> simd_float4x4 {
        let rl = right - left, tb = top - bottom, fn = far - near
        return simd_float4x4(columns: (
            SIMD4<Float>(2/rl,            0,            0, 0),
            SIMD4<Float>(   0,         2/tb,            0, 0),
            SIMD4<Float>(   0,            0,        -1/fn, 0),
            SIMD4<Float>(-(right+left)/rl, -(top+bottom)/tb, -near/fn, 1)
        ))
    }

    private func lookAtMatrix(eye: SIMD3<Float>, target: SIMD3<Float>,
                              up: SIMD3<Float>) -> simd_float4x4 {
        let z = simd_normalize(eye - target)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)
        return simd_float4x4(columns: (
            SIMD4<Float>(x.x,            y.x,            z.x,            0),
            SIMD4<Float>(x.y,            y.y,            z.y,            0),
            SIMD4<Float>(x.z,            y.z,            z.z,            0),
            SIMD4<Float>(-simd_dot(x,eye), -simd_dot(y,eye), -simd_dot(z,eye), 1)
        ))
    }
}

// MARK: - Errors

enum RendererError: Error {
    case initFailed(String)
    case assetNotFound(String)
}
