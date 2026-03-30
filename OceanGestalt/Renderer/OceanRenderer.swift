import Metal
import MetalKit
import simd

@MainActor
final class OceanRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Pipelines
    private let oceanPipeline: MTLRenderPipelineState
    private let skyboxPipeline: MTLRenderPipelineState
    private let buoyPipeline: MTLRenderPipelineState

    // Depth states
    private let depthState: MTLDepthStencilState
    private let skyboxDepthState: MTLDepthStencilState   // lessEqual, no write

    // Meshes
    private let oceanMesh: MeshBuffers
    private let buoyMesh: MeshBuffers

    // Textures
    private var normalMapTex: MTLTexture?
    private var envMapTex: MTLTexture?
    private var gustNoiseTex: MTLTexture?
    private var buoyColorTex: MTLTexture?
    private var buoyNormalTex: MTLTexture?
    private var buoyBumpTex: MTLTexture?
    private var buoyRoughnessTex: MTLTexture?

    // Reflection render target (updated each frame when view size is known)
    private var reflectionColorTex: MTLTexture?
    private var reflectionDepthTex: MTLTexture?
    private var reflectionSize: CGSize = .zero

    // Samplers
    private let repeatSampler: MTLSamplerState
    private let clampSampler: MTLSamplerState

    let uniformState: UniformState

    // Camera
    var cameraYaw: Float   = 0.0
    var cameraPitch: Float = -0.3
    var moveForward: Float = 0
    var moveRight: Float   = 0
    var moveUp: Float      = 0
    private var cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 15, 30)

    private var startTime: Date = Date()

    init?(mtkView: MTKView) {
        guard let device = mtkView.device,
              let queue = device.makeCommandQueue()
        else { return nil }

        self.device       = device
        self.commandQueue = queue
        self.uniformState = UniformState()

        guard let oceanMesh = MeshBuilder.buildGrid(device: device),
              let buoyMesh  = MeshBuilder.buildSphere(device: device)
        else { return nil }
        self.oceanMesh = oceanMesh
        self.buoyMesh  = buoyMesh

        // Samplers
        let repDesc = MTLSamplerDescriptor()
        repDesc.sAddressMode   = .repeat
        repDesc.tAddressMode   = .repeat
        repDesc.minFilter      = .linear
        repDesc.magFilter      = .linear
        repDesc.mipFilter      = .linear
        guard let rep = device.makeSamplerState(descriptor: repDesc) else { return nil }
        repeatSampler = rep

        let clampDesc = MTLSamplerDescriptor()
        clampDesc.sAddressMode = .clampToEdge
        clampDesc.tAddressMode = .clampToEdge
        clampDesc.minFilter    = .linear
        clampDesc.magFilter    = .linear
        guard let clamp = device.makeSamplerState(descriptor: clampDesc) else { return nil }
        clampSampler = clamp

        guard let library = device.makeDefaultLibrary() else { return nil }

        // ---- Ocean pipeline ----
        let oceanVertDesc = MTLVertexDescriptor()
        oceanVertDesc.attributes[0].format = .float3
        oceanVertDesc.attributes[0].offset = 0
        oceanVertDesc.attributes[0].bufferIndex = 0
        oceanVertDesc.attributes[1].format = .float2
        oceanVertDesc.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        oceanVertDesc.attributes[1].bufferIndex = 0
        oceanVertDesc.layouts[0].stride = MemoryLayout<OceanVertex>.stride

        let oceanDesc = MTLRenderPipelineDescriptor()
        oceanDesc.vertexFunction   = library.makeFunction(name: "oceanVertex")
        oceanDesc.fragmentFunction = library.makeFunction(name: "oceanFragment")
        oceanDesc.vertexDescriptor = oceanVertDesc
        oceanDesc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        oceanDesc.depthAttachmentPixelFormat      = mtkView.depthStencilPixelFormat
        guard let op = try? device.makeRenderPipelineState(descriptor: oceanDesc) else { return nil }
        oceanPipeline = op

        // ---- Skybox pipeline ----
        let skyboxDesc = MTLRenderPipelineDescriptor()
        skyboxDesc.vertexFunction   = library.makeFunction(name: "skyboxVertex")
        skyboxDesc.fragmentFunction = library.makeFunction(name: "skyboxFragment")
        skyboxDesc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        skyboxDesc.depthAttachmentPixelFormat      = mtkView.depthStencilPixelFormat
        guard let sp = try? device.makeRenderPipelineState(descriptor: skyboxDesc) else { return nil }
        skyboxPipeline = sp

        // ---- Buoy pipeline ----
        let buoyVertDesc = MTLVertexDescriptor()
        buoyVertDesc.attributes[0].format = .float3
        buoyVertDesc.attributes[0].offset = 0
        buoyVertDesc.attributes[0].bufferIndex = 0
        buoyVertDesc.attributes[1].format = .float3
        buoyVertDesc.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        buoyVertDesc.attributes[1].bufferIndex = 0
        buoyVertDesc.layouts[0].stride = MemoryLayout<BuoyVertex>.stride

        let buoyDesc = MTLRenderPipelineDescriptor()
        buoyDesc.vertexFunction   = library.makeFunction(name: "buoyVertex")
        buoyDesc.fragmentFunction = library.makeFunction(name: "buoyFragment")
        buoyDesc.vertexDescriptor = buoyVertDesc
        buoyDesc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        buoyDesc.depthAttachmentPixelFormat      = mtkView.depthStencilPixelFormat
        guard let bp = try? device.makeRenderPipelineState(descriptor: buoyDesc) else { return nil }
        buoyPipeline = bp

        // ---- Depth states ----
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .less
        depthDesc.isDepthWriteEnabled  = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { return nil }
        depthState = ds

        let skyboxDepthDesc = MTLDepthStencilDescriptor()
        skyboxDepthDesc.depthCompareFunction = .lessEqual
        skyboxDepthDesc.isDepthWriteEnabled  = false
        guard let sds = device.makeDepthStencilState(descriptor: skyboxDepthDesc) else { return nil }
        skyboxDepthState = sds

        super.init()

        loadTextures()
    }

    // MARK: - Texture loading

    private func loadTextures() {
        let loader = MTKTextureLoader(device: device)
        let opts2D: [MTKTextureLoader.Option: Any] = [
            .generateMipmaps: true,
            .SRGB: false,
            .textureUsage: MTLTextureUsage.shaderRead.rawValue
        ]

        normalMapTex   = try? loader.newTexture(name: "fbm_normalmap",   scaleFactor: 1.0, bundle: nil, options: opts2D)
        gustNoiseTex   = try? loader.newTexture(name: "gust_noise_512",  scaleFactor: 1.0, bundle: nil, options: opts2D)
        buoyColorTex   = try? loader.newTexture(name: "metal_color",     scaleFactor: 1.0, bundle: nil, options: opts2D)
        buoyNormalTex  = try? loader.newTexture(name: "metal_normal",    scaleFactor: 1.0, bundle: nil, options: opts2D)
        buoyBumpTex    = try? loader.newTexture(name: "metal_bump",      scaleFactor: 1.0, bundle: nil, options: opts2D)
        buoyRoughnessTex = try? loader.newTexture(name: "metal_bump",    scaleFactor: 1.0, bundle: nil, options: opts2D)

        envMapTex = loadCubemap(loader: loader)
    }

    private func loadCubemap(loader: MTKTextureLoader) -> MTLTexture? {
        // Load each face image
        let faceNames = ["skybox_+X", "skybox_-X", "skybox_+Y", "skybox_-Y", "skybox_+Z", "skybox_-Z"]
        var faceImages: [CGImage] = []
        for name in faceNames {
            guard let url = Bundle.main.url(forResource: name, withExtension: "jpg"),
                  let uiImage = UIImage(contentsOfFile: url.path),
                  let cg = uiImage.cgImage
            else { return nil }
            faceImages.append(cg)
        }

        let faceSize = faceImages[0].width
        let desc = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: .rgba8Unorm,
            size: faceSize,
            mipmapped: true
        )
        desc.usage = [.shaderRead]
        guard let cubeTex = device.makeTexture(descriptor: desc) else { return nil }

        let bytesPerPixel = 4
        let bytesPerRow = faceSize * bytesPerPixel
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue

        var pixels = [UInt8](repeating: 0, count: faceSize * faceSize * bytesPerPixel)

        for (slice, cgImage) in faceImages.enumerated() {
            guard let ctx = CGContext(data: &pixels,
                                      width: faceSize, height: faceSize,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo)
            else { continue }
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: faceSize, height: faceSize))
            cubeTex.replace(region: MTLRegionMake2D(0, 0, faceSize, faceSize),
                            mipmapLevel: 0,
                            slice: slice,
                            withBytes: pixels,
                            bytesPerRow: bytesPerRow,
                            bytesPerImage: faceSize * bytesPerRow)
        }
        return cubeTex
    }

    // MARK: - Reflection texture setup

    private func ensureReflectionTextures(size: CGSize) {
        let halfW = max(1, Int(size.width / 2))
        let halfH = max(1, Int(size.height / 2))
        guard reflectionSize.width != CGFloat(halfW) || reflectionSize.height != CGFloat(halfH) else { return }
        reflectionSize = CGSize(width: halfW, height: halfH)

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: halfW, height: halfH, mipmapped: false)
        colorDesc.usage = [.renderTarget, .shaderRead]
        reflectionColorTex = device.makeTexture(descriptor: colorDesc)

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: halfW, height: halfH, mipmapped: false)
        depthDesc.usage   = [.renderTarget]
        depthDesc.storageMode = .private
        reflectionDepthTex = device.makeTexture(descriptor: depthDesc)
    }

    // MARK: - MTKViewDelegate

    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    nonisolated func draw(in view: MTKView) {
        Task { @MainActor in self.drawFrame(in: view) }
    }

    private func drawFrame(in view: MTKView) {
        let time = Float(Date().timeIntervalSince(startTime))
        uniformState.tick(time: time)

        let drawableSize = view.drawableSize
        ensureReflectionTextures(size: drawableSize)

        // Move camera
        let speed: Float = 5.0 * (1.0 / 60.0)
        let forward = SIMD3<Float>(-sin(cameraYaw), 0, -cos(cameraYaw))
        let right   = SIMD3<Float>( cos(cameraYaw), 0, -sin(cameraYaw))
        cameraPosition += forward * moveForward * speed
        cameraPosition += right   * moveRight   * speed
        cameraPosition.y += moveUp * speed

        let lookTarget = cameraPosition + SIMD3<Float>(
            sin(cameraYaw) * cos(cameraPitch),
            sin(cameraPitch),
            -cos(cameraYaw) * cos(cameraPitch)
        ) * 10.0

        let viewMatrix = lookAt(eye: cameraPosition, center: lookTarget, up: SIMD3<Float>(0, 1, 0))
        let aspect = Float(drawableSize.width / drawableSize.height)
        let projMatrix = perspective(fovY: Float.pi / 4.0, aspect: aspect, near: 0.1, far: 500.0)
        let modelMatrix = matrix_identity_float4x4

        // Reflected camera (flip Y)
        let reflEye    = SIMD3<Float>(cameraPosition.x, -cameraPosition.y, cameraPosition.z)
        let reflTarget = SIMD3<Float>(lookTarget.x, -lookTarget.y, lookTarget.z)
        let reflView   = lookAt(eye: reflEye, center: reflTarget, up: SIMD3<Float>(0, 1, 0))
        let reflMatrix = projMatrix * reflView   // used to project world pos into reflection UV

        var surfaceUniforms = uniformState.buildSurfaceUniforms()
        var buoyUniforms    = uniformState.buildBuoyUniforms()

        var sceneMain = uniformState.buildSceneUniforms(
            time: time, model: modelMatrix, view: viewMatrix, projection: projMatrix,
            reflectionMatrix: reflMatrix, cameraPos: cameraPosition)

        var sceneRefl = uniformState.buildSceneUniforms(
            time: time, model: modelMatrix, view: reflView, projection: projMatrix,
            reflectionMatrix: reflMatrix, cameraPos: reflEye, isReflectionPass: true)

        var buoyModel = buoyModelMatrix(time: time)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // ---- Reflection pass ----
        if let reflColor = reflectionColorTex, let reflDepth = reflectionDepthTex {
            let reflPassDesc = MTLRenderPassDescriptor()
            reflPassDesc.colorAttachments[0].texture     = reflColor
            reflPassDesc.colorAttachments[0].loadAction  = .clear
            reflPassDesc.colorAttachments[0].storeAction = .store
            reflPassDesc.colorAttachments[0].clearColor  = MTLClearColor(red: 0.4, green: 0.6, blue: 0.7, alpha: 1)
            reflPassDesc.depthAttachment.texture          = reflDepth
            reflPassDesc.depthAttachment.loadAction       = .clear
            reflPassDesc.depthAttachment.storeAction      = .dontCare
            reflPassDesc.depthAttachment.clearDepth       = 1.0

            if let enc = commandBuffer.makeRenderCommandEncoder(descriptor: reflPassDesc) {
                // Skybox
                enc.setRenderPipelineState(skyboxPipeline)
                enc.setDepthStencilState(skyboxDepthState)
                enc.setVertexBytes(&sceneRefl, length: MemoryLayout<SceneUniforms>.size, index: 1)
                if let envMap = envMapTex { enc.setFragmentTexture(envMap, index: 0) }
                enc.setFragmentSamplerState(repeatSampler, index: 0)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36)

                // Ocean (clipped at y=0 by shader)
                enc.setRenderPipelineState(oceanPipeline)
                enc.setDepthStencilState(depthState)
                enc.setVertexBuffer(oceanMesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBytes(&sceneRefl,      length: MemoryLayout<SceneUniforms>.size,  index: 1)
                enc.setVertexBytes(&surfaceUniforms, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
                if let t = gustNoiseTex { enc.setVertexTexture(t, index: 0) }
                enc.setVertexSamplerState(repeatSampler, index: 0)
                if let t = normalMapTex   { enc.setFragmentTexture(t, index: 0) }
                if let t = envMapTex      { enc.setFragmentTexture(t, index: 1) }
                enc.setFragmentTexture(reflColor, index: 2)   // self-reference is fine (stale frame)
                enc.setFragmentBytes(&sceneRefl,      length: MemoryLayout<SceneUniforms>.size,  index: 1)
                enc.setFragmentBytes(&surfaceUniforms, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
                enc.setFragmentSamplerState(repeatSampler, index: 0)
                enc.setFragmentSamplerState(clampSampler,  index: 1)
                enc.drawIndexedPrimitives(type: .triangle, indexCount: oceanMesh.indexCount,
                                          indexType: .uint32, indexBuffer: oceanMesh.indexBuffer,
                                          indexBufferOffset: 0)
                enc.endEncoding()
            }
        }

        // ---- Main pass ----
        guard let passDesc = view.currentRenderPassDescriptor,
              let enc = commandBuffer.makeRenderCommandEncoder(descriptor: passDesc)
        else {
            commandBuffer.commit()
            return
        }

        // Skybox
        enc.setRenderPipelineState(skyboxPipeline)
        enc.setDepthStencilState(skyboxDepthState)
        enc.setVertexBytes(&sceneMain, length: MemoryLayout<SceneUniforms>.size, index: 1)
        if let envMap = envMapTex { enc.setFragmentTexture(envMap, index: 0) }
        enc.setFragmentSamplerState(repeatSampler, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36)

        // Ocean
        enc.setRenderPipelineState(oceanPipeline)
        enc.setDepthStencilState(depthState)
        enc.setVertexBuffer(oceanMesh.vertexBuffer, offset: 0, index: 0)
        enc.setVertexBytes(&sceneMain,      length: MemoryLayout<SceneUniforms>.size,  index: 1)
        enc.setVertexBytes(&surfaceUniforms, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
        if let t = gustNoiseTex { enc.setVertexTexture(t, index: 0) }
        enc.setVertexSamplerState(repeatSampler, index: 0)
        if let t = normalMapTex        { enc.setFragmentTexture(t, index: 0) }
        if let t = envMapTex           { enc.setFragmentTexture(t, index: 1) }
        if let t = reflectionColorTex  { enc.setFragmentTexture(t, index: 2) }
        enc.setFragmentBytes(&sceneMain,      length: MemoryLayout<SceneUniforms>.size,  index: 1)
        enc.setFragmentBytes(&surfaceUniforms, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
        enc.setFragmentSamplerState(repeatSampler, index: 0)
        enc.setFragmentSamplerState(clampSampler,  index: 1)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: oceanMesh.indexCount,
                                  indexType: .uint32, indexBuffer: oceanMesh.indexBuffer,
                                  indexBufferOffset: 0)

        // Buoy
        enc.setRenderPipelineState(buoyPipeline)
        enc.setVertexBuffer(buoyMesh.vertexBuffer, offset: 0, index: 0)
        enc.setVertexBytes(&sceneMain,    length: MemoryLayout<SceneUniforms>.size,  index: 1)
        enc.setVertexBytes(&buoyModel,    length: MemoryLayout<simd_float4x4>.size,  index: 2)
        enc.setVertexBytes(&buoyUniforms, length: MemoryLayout<BuoyUniforms>.size,   index: 3)
        enc.setFragmentBytes(&sceneMain,       length: MemoryLayout<SceneUniforms>.size,  index: 1)
        enc.setFragmentBytes(&surfaceUniforms,  length: MemoryLayout<SurfaceUniforms>.size, index: 2)
        enc.setFragmentBytes(&buoyUniforms,    length: MemoryLayout<BuoyUniforms>.size,   index: 3)
        if let t = buoyColorTex    { enc.setFragmentTexture(t, index: 0) }
        if let t = buoyNormalTex   { enc.setFragmentTexture(t, index: 1) }
        if let t = buoyBumpTex     { enc.setFragmentTexture(t, index: 2) }
        if let t = buoyRoughnessTex { enc.setFragmentTexture(t, index: 3) }
        if let t = envMapTex       { enc.setFragmentTexture(t, index: 4) }
        enc.setFragmentSamplerState(repeatSampler, index: 0)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: buoyMesh.indexCount,
                                  indexType: .uint32, indexBuffer: buoyMesh.indexBuffer,
                                  indexBufferOffset: 0)

        enc.endEncoding()
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }

    // MARK: - Buoy

    private func waveDisplacement(at xz: SIMD2<Float>, time: Float) -> SIMD3<Float> {
        var offset = SIMD3<Float>(0, 0, 0)
        for wave in uniformState.waves {
            let safeWavelength = max(wave.wavelength, 0.01)
            let k = 2.0 * Float.pi / safeWavelength
            let w = sqrt(9.81 * k)
            let d = simd_normalize(wave.direction)
            let wt = (w * time).truncatingRemainder(dividingBy: 2.0 * Float.pi)
            let ph = simd_dot(d * k, xz) - wt + wave.phase
            offset.y += wave.amplitude * cos(ph)
            let xzOff = -wave.steepness * d * sin(ph) * wave.amplitude
            offset.x += xzOff.x
            offset.z += xzOff.y
        }
        return SIMD3<Float>(xz.x, 0, xz.y) + offset
    }

    private func buoyModelMatrix(time: Float) -> simd_float4x4 {
        let buoyXZ = SIMD2<Float>(0, 0)
        let eps: Float = 0.1
        let p0 = waveDisplacement(at: buoyXZ, time: time)
        let pX = waveDisplacement(at: buoyXZ + SIMD2<Float>(eps, 0), time: time)
        let pZ = waveDisplacement(at: buoyXZ + SIMD2<Float>(0, eps), time: time)

        let tangent   = pX - p0
        let bitangent = pZ - p0
        let up = simd_normalize(simd_cross(bitangent, tangent))
        let right   = simd_normalize(tangent - simd_dot(tangent, up) * up)
        let forward = simd_cross(up, right)

        let scale: Float = 2.0
        return simd_float4x4(
            SIMD4<Float>(right.x * scale,   right.y * scale,   right.z * scale,   0),
            SIMD4<Float>(up.x * scale,      up.y * scale,      up.z * scale,      0),
            SIMD4<Float>(forward.x * scale, forward.y * scale, forward.z * scale, 0),
            SIMD4<Float>(p0.x, p0.y, p0.z, 1)
        )
    }

    // MARK: - Matrix helpers

    private func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let f = normalize(center - eye)
        let r = normalize(cross(f, up))
        let u = cross(r, f)
        return simd_float4x4(
            SIMD4<Float>( r.x,  u.x, -f.x, 0),
            SIMD4<Float>( r.y,  u.y, -f.y, 0),
            SIMD4<Float>( r.z,  u.z, -f.z, 0),
            SIMD4<Float>(-dot(r, eye), -dot(u, eye), dot(f, eye), 1)
        )
    }

    private func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let y = 1 / tan(fovY * 0.5)
        let x = y / aspect
        let z = far / (near - far)
        return simd_float4x4(
            SIMD4<Float>(x, 0,  0,  0),
            SIMD4<Float>(0, y,  0,  0),
            SIMD4<Float>(0, 0,  z, -1),
            SIMD4<Float>(0, 0,  z * near, 0)
        )
    }
}
