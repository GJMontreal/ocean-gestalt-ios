import Metal
import MetalKit
import simd
import GestureCamera

// ---------------------------------------------------------------------------
// Renderer — draws scene.json props (ModelDrawable) with no ocean/skybox/reflections.
// Env map is deferred; nil is passed to ModelDrawable for now.
// ---------------------------------------------------------------------------

@MainActor
final class OceanRenderer {

    // MARK: - Public

    let device:   MTLDevice
    var isRunning: Bool = true
    var audio:    SurfAudio? = nil

    /// 'm' key: when false, draws all props as a full-body wireframe instead of shaded.
    var showMesh: Bool = true
    /// 'n' key: when true, draws yellow surface-normal lines on the ocean mesh.
    var showNormals: Bool = false

    var initialCameraTransform: CameraTransform {
        // Use identity (0,1,5 looking toward -Z) while debugging the Drawable path.
        // The scene.json camera is at X≈20 looking in +Z, putting the cube at
        // origin behind it. Restore scene.json once rendering is confirmed working.
        return .identity
    }

    // MARK: - Metal

    private let commandQueue: MTLCommandQueue
    private let oceanPipeline:      MTLRenderPipelineState  // solid ocean
    private let oceanWirePipeline:  MTLRenderPipelineState  // wireframe ocean
    private let normalsPipeline:    MTLRenderPipelineState  // surface normals debug lines
    private let skyboxPipeline:     MTLRenderPipelineState

    private let solidDepthState:    MTLDepthStencilState    // less, write
    private let skyboxDepthState:   MTLDepthStencilState    // lessEqual, no write

    private let repeatSampler:  MTLSamplerState
    private let clampSampler:   MTLSamplerState

    // MARK: - Scene & uniforms
    private let scene:    OceanScene
    private let uniforms: UniformState

    // MARK: - Textures
    private let normalMapTex: MTLTexture
    private let envMapTex:    MTLTexture
    private let gustTex:      MTLTexture

    // MARK: - Ocean mesh

    private let oceanMesh: OceanMeshBuffers

    // MARK: - Reflection

    private let reflectionPass: ReflectionPass

    // MARK: - Time

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

        // ---- Textures ----
        let loader = MTKTextureLoader(device: device)
        normalMapTex = try OceanRenderer.loadTexture(loader: loader,
                                                     name: "NormalMap",
                                                     subdirectory: "data/textures",
                                                     sRGB: false)
        gustTex      = try OceanRenderer.loadTexture(loader: loader,
                                                     name: "gust_noise_512",
                                                     subdirectory: "data/textures",
                                                     sRGB: false)
        envMapTex    = try OceanRenderer.loadCubemap(device: device, loader: loader)

        let colorFmt = colorPixelFormat
        let depthFmt = depthPixelFormat
        let env = envMapTex

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
                envMap: env
            )
        }
        scene = try OceanScene(device: device, drawableFactory: drawableFactory)

        // ---- Ocean mesh ----
        guard let mesh = MeshBuilder.buildGrid(device: device,
                                               size: scene.mesh.size,
                                               subdivisions: scene.mesh.subdivisions)
        else { throw RendererError.initFailed("ocean mesh") }
        oceanMesh = mesh

        // ---- Ocean pipelines ----
        let oceanDesc = MTLRenderPipelineDescriptor()
        oceanDesc.vertexDescriptor                  = MeshBuilder.oceanVertexDescriptor
        oceanDesc.colorAttachments[0].pixelFormat   = colorPixelFormat
        oceanDesc.depthAttachmentPixelFormat        = depthPixelFormat
        oceanDesc.vertexFunction   = library.makeFunction(name: "oceanVertex")
        oceanDesc.fragmentFunction = library.makeFunction(name: "oceanFragment")
        oceanPipeline = try device.makeRenderPipelineState(descriptor: oceanDesc)

        let wireDesc = MTLRenderPipelineDescriptor()
        wireDesc.vertexDescriptor                  = MeshBuilder.oceanVertexDescriptor
        wireDesc.colorAttachments[0].pixelFormat   = colorPixelFormat
        wireDesc.depthAttachmentPixelFormat        = depthPixelFormat
        wireDesc.vertexFunction   = library.makeFunction(name: "oceanVertex")
        wireDesc.fragmentFunction = library.makeFunction(name: "wireframeFragment")
        oceanWirePipeline = try device.makeRenderPipelineState(descriptor: wireDesc)

        // No vertex descriptor: normalsVertex reads the vertex buffer directly via buffer(0)
        let normDesc = MTLRenderPipelineDescriptor()
        normDesc.colorAttachments[0].pixelFormat = colorPixelFormat
        normDesc.depthAttachmentPixelFormat      = depthPixelFormat
        normDesc.vertexFunction   = library.makeFunction(name: "normalsVertex")
        normDesc.fragmentFunction = library.makeFunction(name: "normalsFragment")
        normalsPipeline = try device.makeRenderPipelineState(descriptor: normDesc)

        // ---- Skybox pipeline ----
        let skyDesc = MTLRenderPipelineDescriptor()
        skyDesc.colorAttachments[0].pixelFormat = colorPixelFormat
        skyDesc.depthAttachmentPixelFormat      = depthPixelFormat
        skyDesc.vertexFunction   = library.makeFunction(name: "skyboxVertex")
        skyDesc.fragmentFunction = library.makeFunction(name: "skyboxFragment")
        skyboxPipeline = try device.makeRenderPipelineState(descriptor: skyDesc)

        // ---- Depth states ----
        let solidDepthDesc = MTLDepthStencilDescriptor()
        solidDepthDesc.depthCompareFunction = .less
        solidDepthDesc.isDepthWriteEnabled  = true
        guard let sds = device.makeDepthStencilState(descriptor: solidDepthDesc) else {
            throw RendererError.initFailed("solid depth state")
        }

        let skyDepthDesc = MTLDepthStencilDescriptor()
        skyDepthDesc.depthCompareFunction = .lessEqual
        skyDepthDesc.isDepthWriteEnabled  = false
        guard let skds = device.makeDepthStencilState(descriptor: skyDepthDesc) else {
            throw RendererError.initFailed("skybox depth state")
        }

        skyboxDepthState = skds
        solidDepthState = sds

        // ---- Samplers ----
        let repDesc = MTLSamplerDescriptor()
        repDesc.sAddressMode = .repeat
        repDesc.tAddressMode = .repeat
        repDesc.minFilter    = .linear
        repDesc.magFilter    = .linear
        repDesc.mipFilter    = .linear
        guard let rs = device.makeSamplerState(descriptor: repDesc) else {
            throw RendererError.initFailed("repeat sampler")
        }
        repeatSampler = rs

        let clpDesc = MTLSamplerDescriptor()
        clpDesc.sAddressMode = .clampToEdge
        clpDesc.tAddressMode = .clampToEdge
        clpDesc.minFilter    = .linear
        clpDesc.magFilter    = .linear
        guard let cs = device.makeSamplerState(descriptor: clpDesc) else {
            throw RendererError.initFailed("clamp sampler")
        }
        clampSampler = cs

        // ---- Reflection pass ----
        reflectionPass = ReflectionPass(device: device, size: scene.reflectionSize,
                                        pixelFormat: colorPixelFormat)


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

        // ---- Camera wave floating ----
        let cameraXZ = SIMD2<Float>(camera.position.x, camera.position.z)
        let cameraWave = gerstnerOffset(xz: cameraXZ, time: time)
        var floatingCamera = camera
        floatingCamera.position.y = camera.position.y + cameraWave.y


        let size   = view.drawableSize
        let aspect = Float(size.width / size.height)
        let fovY   = scene.camera.zoom * Float.pi / 180
        let proj   = perspectiveMatrix(fovY: fovY, aspect: aspect, nearZ: 0.1, farZ: 500)

        let reflectY: simd_float4x4 = simd_float4x4(diagonal: SIMD4<Float>(1, -1, 1, 1))
        let reflectedView = floatingCamera.viewMatrix * reflectY
        let reflectionMatrix = proj * reflectedView

        // Light space matrix (for future shadow pass)
        let lightPos  = uniforms.lightPos
        let halfSize  = Float(scene.mesh.size) * 0.5
        let lightProj = orthographicMatrix(left: -halfSize, right: halfSize,
                                           bottom: -halfSize, top: halfSize,
                                           near: 0.1, far: 50)
        let lightView = lookAtMatrix(eye: lightPos, target: .zero, up: SIMD3<Float>(0,1,0))
        let lightSpaceMatrix = lightProj * lightView

        let surfaceU = uniforms.buildSurfaceUniforms()

        let waveOff: (SIMD2<Float>) -> SIMD3<Float> = { [unowned self] xz in
            self.gerstnerOffset(xz: xz, time: time)
        }

        // ---- Command buffer ----
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { return }
        cmdBuf.label = "OceanFrame"

        // ---- Reflection pre-pass ----
        if let reflRPD = reflectionPass.passDescriptor {
            let reflSceneU = uniforms.buildSceneUniforms(
                time:             time,
                model:            matrix_identity_float4x4,
                view:             reflectedView,
                projection:       proj,
                reflectionMatrix: reflectionMatrix,
                lightSpaceMatrix: lightSpaceMatrix,
                cameraPos:        floatingCamera.position,
                lightPos:         lightPos,
                isReflectionPass: true
            )
            if let enc = cmdBuf.makeRenderCommandEncoder(descriptor: reflRPD) {
                enc.label = "ReflectionPass"
                encodeSkybox(enc, scene: reflSceneU)
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
            time:             time,
            model:            matrix_identity_float4x4,
            view:             floatingCamera.viewMatrix,
            projection:       proj,
            reflectionMatrix: reflectionMatrix,
            cameraPos:        floatingCamera.position,
            lightPos:         lightPos
        )

        if let enc = cmdBuf.makeRenderCommandEncoder(descriptor: rpd) {
            enc.label = "MainPass"

            // Ocean solid — hidden when showMesh is false so the wireframe + normals
            // can be inspected without the opaque surface covering them
            if showMesh {
                encodeOcean(enc, scene: sceneU, surface: surfaceU,
                            wireframe: false, time: time)
            }

            // Ocean wireframe (model matrix shifted -0.2 Y to sit just below the surface)
            var wireSceneU = sceneU
            var wireModelM = matrix_identity_float4x4
            wireModelM.columns.3.y = -0.2
            wireSceneU.modelMatrix = wireModelM
            encodeOcean(enc, scene: wireSceneU, surface: surfaceU,
                        wireframe: true, time: time)

            // Scene models: shaded (normal) or full wireframe (debug)
            if showMesh {
                scene.encode(into: enc, time: time, waveOffset: waveOff,
                             scene: sceneU, surface: surfaceU)
            } else {
                scene.encodeDebugMesh(into: enc, time: time, waveOffset: waveOff,
                                      scene: sceneU, surface: surfaceU)
            }

            // Surface normals debug lines
            if showNormals {
                encodeNormals(enc, scene: sceneU)
            }

            enc.endEncoding()
        }

        cmdBuf.present(drawable)
        cmdBuf.commit()
    }

    // MARK: - Pass encoders

    private func encodeSkybox(_ enc: MTLRenderCommandEncoder,
                               scene: SceneUniforms) {
        enc.setRenderPipelineState(skyboxPipeline)
        enc.setDepthStencilState(skyboxDepthState)
        var scn = scene
        enc.setVertexBytes(&scn, length: MemoryLayout<SceneUniforms>.size, index: 1)
        enc.setFragmentTexture(envMapTex, index: 0)
        enc.setFragmentSamplerState(clampSampler, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36)
    }

    private func encodeOcean(_ enc: MTLRenderCommandEncoder,
                              scene: SceneUniforms,
                              surface: SurfaceUniforms,
                              wireframe: Bool,
                             time: Float) {
        enc.setRenderPipelineState(wireframe ? oceanWirePipeline : oceanPipeline)
        enc.setDepthStencilState(solidDepthState)
        enc.setVertexBuffer(oceanMesh.vertexBuffer, offset: 0, index: 0)
        var scn = scene
        var srf = surface
        enc.setVertexBytes(&scn, length: MemoryLayout<SceneUniforms>.size,  index: 1)
        enc.setVertexBytes(&srf, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
        enc.setVertexTexture(gustTex, index: 0)
        enc.setVertexSamplerState(repeatSampler, index: 0)
        enc.setFragmentBytes(&scn, length: MemoryLayout<SceneUniforms>.size,  index: 1)
        enc.setFragmentBytes(&srf, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
        if !wireframe {
            enc.setFragmentTexture(normalMapTex,                    index: 0)
            enc.setFragmentTexture(envMapTex,                       index: 1)
            enc.setFragmentTexture(reflectionPass.colorTexture,     index: 2)
            enc.setFragmentSamplerState(repeatSampler, index: 0)
            enc.setFragmentSamplerState(clampSampler,  index: 1)
        }
        let buf = wireframe ? oceanMesh.wirelineBuffer : oceanMesh.indexBuffer
        let cnt = wireframe ? oceanMesh.wirelineCount  : oceanMesh.indexCount
        let prim: MTLPrimitiveType = wireframe ? .line : .triangle
        enc.drawIndexedPrimitives(type: prim, indexCount: cnt,
                                  indexType: .uint32,
                                  indexBuffer: buf, indexBufferOffset: 0)
    }

    // Mirrors the NormalsUniforms struct in OceanShaders.metal (must stay in sync).
    private struct NormalsUniforms {
        var gridWidth: UInt32
        var stride:    UInt32
        var _pad:      SIMD2<Float> = .zero
    }

    private func encodeNormals(_ enc: MTLRenderCommandEncoder,
                                scene: SceneUniforms) {
        // Sample every 8th vertex in both row and column directions.
        // For a 121-vertex-wide grid: (121+7)/8 = 16 samples per side → 256 lines.
        let stride: UInt32    = 8
        let gw: UInt32        = UInt32(oceanMesh.gridWidth)
        let samplesPerRow     = (gw + stride - 1) / stride
        var normUni           = NormalsUniforms(gridWidth: gw, stride: stride)
        let sampleCount       = Int(samplesPerRow) * Int(samplesPerRow)

        enc.setRenderPipelineState(normalsPipeline)
        enc.setDepthStencilState(solidDepthState)
        enc.setVertexBuffer(oceanMesh.vertexBuffer, offset: 0, index: 0)
        var scn = scene
        enc.setVertexBytes(&scn,     length: MemoryLayout<SceneUniforms>.size,   index: 1)
        enc.setVertexBytes(&normUni, length: MemoryLayout<NormalsUniforms>.size, index: 2)
        enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: sampleCount * 2)
    }

    // MARK: - Texture loading

    private static func loadTexture(loader: MTKTextureLoader,
                                    name: String,
                                    subdirectory: String,
                                    sRGB: Bool) throws -> MTLTexture {
        guard let url = Bundle.main.url(forResource: name, withExtension: nil,
                                        subdirectory: subdirectory)
                     ?? Bundle.main.url(forResource: name, withExtension: "png",
                                        subdirectory: subdirectory)
                     ?? Bundle.main.url(forResource: name, withExtension: "jpg",
                                        subdirectory: subdirectory)
        else { throw RendererError.assetNotFound(name) }
        let opts: [MTKTextureLoader.Option: Any] = [
            .generateMipmaps: true,
            .SRGB: sRGB
        ]
        return try loader.newTexture(URL: url, options: opts)
    }

    /// Assembles 6 face images into an MTLTexture of type .typeCube.
    /// Faces are loaded in Metal order: +X, -X, +Y, -Y, +Z, -Z (slices 0-5).
    private static func loadCubemap(device: MTLDevice,
                                    loader: MTKTextureLoader) throws -> MTLTexture {
        let faceNames = ["skybox_+X", "skybox_-X", "skybox_+Y",
                         "skybox_-Y", "skybox_+Z", "skybox_-Z"]
        let opts: [MTKTextureLoader.Option: Any] = [.SRGB: false]

        // Load all faces to determine size
        var faceTex = [MTLTexture]()
        for name in faceNames {
            guard let url = Bundle.main.url(forResource: name, withExtension: "jpg",
                                            subdirectory: "data/textures")
            else { throw RendererError.assetNotFound(name) }
            faceTex.append(try loader.newTexture(URL: url, options: opts))
        }

        let w = faceTex[0].width
        let cubeDesc = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: faceTex[0].pixelFormat,
            size: w,
            mipmapped: false
        )
        cubeDesc.usage = [.shaderRead]
        guard let cubeTex = device.makeTexture(descriptor: cubeDesc) else {
            throw RendererError.initFailed("cubemap texture")
        }
        cubeTex.label = "SkyboxCubemap"

        guard let cmdQ    = device.makeCommandQueue(),
              let cmdBuf  = cmdQ.makeCommandBuffer(),
              let blit    = cmdBuf.makeBlitCommandEncoder()
        else { throw RendererError.initFailed("cubemap blit") }

        for (slice, face) in faceTex.enumerated() {
            blit.copy(from: face, sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize:   MTLSize(width: w, height: w, depth: 1),
                      to: cubeTex,  destinationSlice: slice, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        }
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        return cubeTex
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


    // MARK: - CPU Gerstner (mirrors waveOffset in gerstner.vert / GerstnerWave.cpp)

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

}

// MARK: - Errors

enum RendererError: Error {
    case initFailed(String)
    case assetNotFound(String)
}
