import Metal
import MetalKit
import simd

/// Owns all ocean-specific GPU resources: pipelines, samplers, ocean textures,
/// mesh buffers, and the reflection pre-pass render target. Encodes the solid
/// ocean, wireframe, and optional surface-normal debug lines.
/// Mirrors the ocean-rendering concerns in src/core/OceanGestalt.cpp.
@MainActor
final class OceanPass {

    // MARK: - Exposed to OceanRenderer

    /// Clamp sampler shared with SkyboxPass.
    let clampSampler: MTLSamplerState

    /// Pass descriptor for the reflection pre-pass; nil when unavailable.
    var reflectionPassDescriptor: MTLRenderPassDescriptor? { reflectionPass.passDescriptor }

    // MARK: - Private Metal state

    private let oceanPipeline:     MTLRenderPipelineState
    private let oceanWirePipeline: MTLRenderPipelineState
    private let normalsPipeline:   MTLRenderPipelineState
    private let solidDepthState:   MTLDepthStencilState
    private let repeatSampler:     MTLSamplerState
    private let normalMapTex:      MTLTexture
    private let gustTex:           MTLTexture
    private let envMapTex:         MTLTexture
    private let oceanMesh:         OceanMeshBuffers
    private let reflectionPass:    ReflectionPass

    // MARK: - Init

    init(
        device: MTLDevice,
        library: MTLLibrary,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat,
        meshConfig: MeshConfig,
        reflectionSize: Int,
        envMap: MTLTexture
    ) throws {
        self.envMapTex = envMap

        // ---- Ocean-specific textures ----
        let loader = MTKTextureLoader(device: device)
        normalMapTex = try OceanPass.loadTexture(loader: loader,
                                                  name: "fbm_normalmap",
                                                  subdirectory: "data/textures",
                                                  sRGB: false)
        gustTex      = try OceanPass.loadTexture(loader: loader,
                                                  name: "gust_noise_512",
                                                  subdirectory: "data/textures",
                                                  sRGB: false)

        // ---- Samplers ----
        let repDesc = MTLSamplerDescriptor()
        repDesc.sAddressMode = .repeat
        repDesc.tAddressMode = .repeat
        repDesc.minFilter    = .linear
        repDesc.magFilter    = .linear
        repDesc.mipFilter    = .linear
        guard let rs = device.makeSamplerState(descriptor: repDesc) else {
            throw OceanPassError.initFailed("repeat sampler")
        }
        repeatSampler = rs

        let clpDesc = MTLSamplerDescriptor()
        clpDesc.sAddressMode = .clampToEdge
        clpDesc.tAddressMode = .clampToEdge
        clpDesc.minFilter    = .linear
        clpDesc.magFilter    = .linear
        guard let cs = device.makeSamplerState(descriptor: clpDesc) else {
            throw OceanPassError.initFailed("clamp sampler")
        }
        clampSampler = cs

        // ---- Ocean mesh ----
        guard let mesh = MeshBuilder.buildGrid(device: device,
                                               size: meshConfig.size,
                                               subdivisions: meshConfig.subdivisions)
        else { throw OceanPassError.initFailed("ocean mesh") }
        oceanMesh = mesh

        // ---- Pipelines ----
        let oceanDesc = MTLRenderPipelineDescriptor()
        oceanDesc.vertexDescriptor                = MeshBuilder.oceanVertexDescriptor
        oceanDesc.colorAttachments[0].pixelFormat = colorPixelFormat
        oceanDesc.depthAttachmentPixelFormat      = depthPixelFormat
        oceanDesc.vertexFunction   = library.makeFunction(name: "oceanVertex")
        oceanDesc.fragmentFunction = library.makeFunction(name: "oceanFragment")
        oceanPipeline = try device.makeRenderPipelineState(descriptor: oceanDesc)

        let wireDesc = MTLRenderPipelineDescriptor()
        wireDesc.vertexDescriptor                = MeshBuilder.oceanVertexDescriptor
        wireDesc.colorAttachments[0].pixelFormat = colorPixelFormat
        wireDesc.depthAttachmentPixelFormat      = depthPixelFormat
        wireDesc.vertexFunction   = library.makeFunction(name: "oceanVertex")
        wireDesc.fragmentFunction = library.makeFunction(name: "wireframeFragment")
        oceanWirePipeline = try device.makeRenderPipelineState(descriptor: wireDesc)

        // No vertex descriptor: normalsVertex reads the vertex buffer directly.
        let normDesc = MTLRenderPipelineDescriptor()
        normDesc.colorAttachments[0].pixelFormat = colorPixelFormat
        normDesc.depthAttachmentPixelFormat      = depthPixelFormat
        normDesc.vertexFunction   = library.makeFunction(name: "normalsVertex")
        normDesc.fragmentFunction = library.makeFunction(name: "normalsFragment")
        normalsPipeline = try device.makeRenderPipelineState(descriptor: normDesc)

        // ---- Depth state ----
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .less
        depthDesc.isDepthWriteEnabled  = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else {
            throw OceanPassError.initFailed("solid depth state")
        }
        solidDepthState = ds

        // ---- Reflection pass ----
        reflectionPass = ReflectionPass(device: device, size: reflectionSize,
                                        pixelFormat: colorPixelFormat)
    }

    // MARK: - Encode

    /// Main-pass encode: solid surface (gated on showMesh), wireframe, and
    /// optionally surface-normal debug lines.
    func encode(
        into encoder: MTLRenderCommandEncoder,
        scene: SceneUniforms,
        surface: SurfaceUniforms,
        showMesh: Bool,
        showNormals: Bool,
        time: Float
    ) {
        if showMesh {
            encodeOcean(encoder, scene: scene, surface: surface, wireframe: false, time: time)
        }

        // Wireframe sits -0.2 Y below the surface so wave crests naturally occlude it.
        var wireScene = scene
        var wireModel = matrix_identity_float4x4
        wireModel.columns.3.y = -0.2
        wireScene.modelMatrix = wireModel
        encodeOcean(encoder, scene: wireScene, surface: surface, wireframe: true, time: time)

        if showNormals {
            encodeNormals(encoder, scene: scene)
        }
    }

    // MARK: - Private encode helpers

    private func encodeOcean(
        _ enc: MTLRenderCommandEncoder,
        scene: SceneUniforms,
        surface: SurfaceUniforms,
        wireframe: Bool,
        time: Float
    ) {
        enc.setRenderPipelineState(wireframe ? oceanWirePipeline : oceanPipeline)
        enc.setDepthStencilState(solidDepthState)
        // Solid surface: cull front faces so the ocean disappears when viewed from below.
        // Metal's default front-face winding is CW; ocean triangles appear CCW from above
        // (back-facing, not culled) and CW from below (front-facing, culled).
        // Wireframe (line primitives) is unaffected by cull mode.
        enc.setCullMode(wireframe ? .none : .front)
        enc.setVertexBuffer(oceanMesh.vertexBuffer, offset: 0, index: 0)
        var scn = scene
        var srf = surface
        enc.setVertexBytes(&scn, length: MemoryLayout<SceneUniforms>.size,   index: 1)
        enc.setVertexBytes(&srf, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
        enc.setVertexTexture(gustTex, index: 0)
        enc.setVertexSamplerState(repeatSampler, index: 0)
        enc.setFragmentBytes(&scn, length: MemoryLayout<SceneUniforms>.size,   index: 1)
        enc.setFragmentBytes(&srf, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
        if !wireframe {
            enc.setFragmentTexture(normalMapTex,                    index: 0)
            enc.setFragmentTexture(envMapTex,                       index: 1)
            enc.setFragmentTexture(reflectionPass.colorTexture,     index: 2)
            enc.setFragmentSamplerState(repeatSampler, index: 0)
            enc.setFragmentSamplerState(clampSampler,  index: 1)
        }
        let buf  = wireframe ? oceanMesh.wirelineBuffer : oceanMesh.indexBuffer
        let cnt  = wireframe ? oceanMesh.wirelineCount  : oceanMesh.indexCount
        let prim: MTLPrimitiveType = wireframe ? .line : .triangle
        enc.drawIndexedPrimitives(type: prim, indexCount: cnt,
                                  indexType: .uint32,
                                  indexBuffer: buf, indexBufferOffset: 0)
    }

    private func encodeNormals(_ enc: MTLRenderCommandEncoder, scene: SceneUniforms) {
        enc.setRenderPipelineState(normalsPipeline)
        enc.setDepthStencilState(solidDepthState)
        enc.setVertexBuffer(oceanMesh.vertexBuffer, offset: 0, index: 0)
        var scn = scene
        enc.setVertexBytes(&scn, length: MemoryLayout<SceneUniforms>.size, index: 1)
        enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: oceanMesh.vertexCount * 2)
    }

    // MARK: - Texture loading

    static func loadTexture(loader: MTKTextureLoader,
                             name: String,
                             subdirectory: String,
                             sRGB: Bool) throws -> MTLTexture {
        guard let url = Bundle.main.url(forResource: name, withExtension: nil,
                                        subdirectory: subdirectory)
                     ?? Bundle.main.url(forResource: name, withExtension: "png",
                                        subdirectory: subdirectory)
                     ?? Bundle.main.url(forResource: name, withExtension: "jpg",
                                        subdirectory: subdirectory)
        else { throw OceanPassError.assetNotFound(name) }
        let opts: [MTKTextureLoader.Option: Any] = [
            .generateMipmaps: true,
            .SRGB: sRGB
        ]
        return try loader.newTexture(URL: url, options: opts)
    }

    static func loadCubemap(device: MTLDevice, loader: MTKTextureLoader) throws -> MTLTexture {
        let faceNames = ["skybox_+X", "skybox_-X", "skybox_+Y",
                         "skybox_-Y", "skybox_+Z", "skybox_-Z"]
        let opts: [MTKTextureLoader.Option: Any] = [.SRGB: false]

        var faceTex = [MTLTexture]()
        for name in faceNames {
            guard let url = Bundle.main.url(forResource: name, withExtension: "jpg",
                                            subdirectory: "data/textures")
            else { throw OceanPassError.assetNotFound(name) }
            faceTex.append(try loader.newTexture(URL: url, options: opts))
        }

        let w = faceTex[0].width
        let cubeDesc = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: faceTex[0].pixelFormat, size: w, mipmapped: false)
        cubeDesc.usage = [.shaderRead]
        guard let cubeTex = device.makeTexture(descriptor: cubeDesc) else {
            throw OceanPassError.initFailed("cubemap texture")
        }
        cubeTex.label = "SkyboxCubemap"

        guard let cmdQ   = device.makeCommandQueue(),
              let cmdBuf = cmdQ.makeCommandBuffer(),
              let blit   = cmdBuf.makeBlitCommandEncoder()
        else { throw OceanPassError.initFailed("cubemap blit") }

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
}

// MARK: - Errors

enum OceanPassError: Error {
    case initFailed(String)
    case assetNotFound(String)
}
