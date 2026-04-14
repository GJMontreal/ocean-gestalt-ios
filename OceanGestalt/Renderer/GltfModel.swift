import Metal
import MetalKit
import ModelIO
import simd

/// Drawable backed by a GLTF/GLB file. Loads via ModelIO, uploads to Metal
/// buffers, and performs the two-pass waterline draw (solid above / wireframe
/// below) matching the C++ normal_bump.frag / GltfModel.cpp behaviour.
//@MainActor
final class GltfModel: @MainActor Drawable {

    // MARK: - Types

    private struct Primitive {
        let vertexBuffer: MTLBuffer
        let indexBuffer: MTLBuffer
        let indexCount: Int
        let materialIndex: Int
        // Deduplicated edge list for the wireframe pass
        let edgeBuffer: MTLBuffer
        let edgeCount: Int
    }

    private struct Material {
        var colorTex: MTLTexture?
        var normalTex: MTLTexture?
        var mrTex: MTLTexture?
    }

    // MARK: - Metal state

    private let device: MTLDevice
    private let mainPipeline: MTLRenderPipelineState
    private let wirePipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let wireDepthState: MTLDepthStencilState  // less, no write — wire below surface
    private let sampler: MTLSamplerState
    private let envMap: MTLTexture?

    // MARK: - Geometry & materials

    private let primitives: [Primitive]
    private let materials: [Material]

    // MARK: - Per-model uniforms (set from scene.json via ModelConfig)

    var waterlineBias: Float      = 0.15
    var waterlineWidth: Float     = 0.5
    var waterlineStrength: Float  = 0.45
    var waterlineNoise: Float     = 0.8
    var wetStrength: Float        = 0.5
    var specularFactor: Float     = 1.0
    var bumpFactor: Float         = 0.5
    var wireframeColor: SIMD4<Float> = SIMD4<Float>(0, 0.45, 0.65, 1)

    // MARK: - Vertex descriptor (shared between MDL and MTL)

    /// Interleaved: position(float3,12) + texCoord(float2,8) + normal(float3,12) + tangent(float4,16) = 48 bytes
    static let mtlVertexDescriptor: MTLVertexDescriptor = {
        let d = MTLVertexDescriptor()
        d.attributes[0].format = .float3; d.attributes[0].offset = 0;  d.attributes[0].bufferIndex = 0
        d.attributes[1].format = .float2; d.attributes[1].offset = 12; d.attributes[1].bufferIndex = 0
        d.attributes[2].format = .float3; d.attributes[2].offset = 20; d.attributes[2].bufferIndex = 0
        d.attributes[3].format = .float4; d.attributes[3].offset = 32; d.attributes[3].bufferIndex = 0
        d.layouts[0].stride = 48
        return d
    }()

    static func mdlVertexDescriptor() -> MDLVertexDescriptor {
        let d = MDLVertexDescriptor()
        d.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition,
                                             format: .float3, offset: 0, bufferIndex: 0)
        d.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeTextureCoordinate,
                                             format: .float2, offset: 12, bufferIndex: 0)
        d.attributes[2] = MDLVertexAttribute(name: MDLVertexAttributeNormal,
                                             format: .float3, offset: 20, bufferIndex: 0)
        d.attributes[3] = MDLVertexAttribute(name: MDLVertexAttributeTangent,
                                             format: .float4, offset: 32, bufferIndex: 0)
        d.layouts[0] = MDLVertexBufferLayout(stride: 48)
        return d
    }

    // MARK: - Init

    init(
        device: MTLDevice,
        url: URL,
        config: ModelConfig,
        library: MTLLibrary,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat,
        envMap: MTLTexture?
    ) throws {
        self.device = device
        self.envMap = envMap

        // ---- Pipelines ----
        let pipeDesc = MTLRenderPipelineDescriptor()
        pipeDesc.vertexFunction   = library.makeFunction(name: "modelVertex")
        pipeDesc.fragmentFunction = library.makeFunction(name: "modelFragment")
        pipeDesc.vertexDescriptor = GltfModel.mtlVertexDescriptor
        pipeDesc.colorAttachments[0].pixelFormat = colorPixelFormat
        pipeDesc.depthAttachmentPixelFormat      = depthPixelFormat
        mainPipeline = try device.makeRenderPipelineState(descriptor: pipeDesc)

        wirePipeline = mainPipeline  // same shader — waterlineClip drives the two passes

        // ---- Depth states ----
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .less
        depthDesc.isDepthWriteEnabled  = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else {
            throw GltfModelError.initFailed("depth state")
        }
        depthState = ds

        let wireDepthDesc = MTLDepthStencilDescriptor()
        wireDepthDesc.depthCompareFunction = .less
        wireDepthDesc.isDepthWriteEnabled  = false
        guard let wds = device.makeDepthStencilState(descriptor: wireDepthDesc) else {
            throw GltfModelError.initFailed("wire depth state")
        }
        wireDepthState = wds

        // ---- Sampler ----
        let sampDesc = MTLSamplerDescriptor()
        sampDesc.sAddressMode = .repeat
        sampDesc.tAddressMode = .repeat
        sampDesc.minFilter    = .linear
        sampDesc.magFilter    = .linear
        sampDesc.mipFilter    = .linear
        guard let samp = device.makeSamplerState(descriptor: sampDesc) else {
            throw GltfModelError.initFailed("sampler")
        }
        sampler = samp

        print("canImport glb %b", MDLAsset.canImportFileExtension("glb"))
        // ---- Load GLB ----
        let bufferAllocator = MTKMeshBufferAllocator(device: device)
        let asset = MDLAsset(url: url,
                             vertexDescriptor: GltfModel.mdlVertexDescriptor(),
                             bufferAllocator: bufferAllocator)
        asset.loadTextures()

        let (mdlMeshes, mtkMeshes) = try MTKMesh.newMeshes(asset: asset, device: device)

        // ---- Fallback 1×1 textures (used when a GLB has no embedded texture) ----
        // Flat normal (0.5,0.5,1.0) → decodes to tangent-space (0,0,1); white albedo; mid roughness.
        let fbColor  = GltfModel.make1x1(device: device, rgba: (255, 255, 255, 255))
        let fbNormal = GltfModel.make1x1(device: device, rgba: (127, 127, 255, 255))
        let fbMR     = GltfModel.make1x1(device: device, rgba: (0,   128,   0, 255))

        // ---- Materials ----
        let textureLoader = MTKTextureLoader(device: device)
        var loadedMaterials: [Material] = []

        for (mdlMesh, mtkMesh) in zip(mdlMeshes, mtkMeshes) {
            for (mdlSubmesh, _) in zip(mdlMesh.submeshes as! [MDLSubmesh], mtkMesh.submeshes) {
                var mat = Material()
                if let mdlMat = mdlSubmesh.material {
                    mat.colorTex  = GltfModel.loadTexture(from: mdlMat, semantic: .baseColor,
                                                          loader: textureLoader, sRGB: true)
                    mat.normalTex = GltfModel.loadTexture(from: mdlMat, semantic: .tangentSpaceNormal,
                                                          loader: textureLoader, sRGB: false)
                    mat.mrTex     = GltfModel.loadTexture(from: mdlMat, semantic: .roughness,
                                                          loader: textureLoader, sRGB: false)
                }
                // Fall back to 1×1 defaults so the shader never samples an unbound slot
                mat.colorTex  = mat.colorTex  ?? fbColor
                mat.normalTex = mat.normalTex ?? fbNormal
                mat.mrTex     = mat.mrTex     ?? fbMR
                print("[GltfModel] \(config.name) submesh — color:\(mat.colorTex != nil) normal:\(mat.normalTex != nil) mr:\(mat.mrTex != nil)")
                loadedMaterials.append(mat)
            }
        }

        // ---- Primitives ----
        var loadedPrimitives: [Primitive] = []
        var materialIndex = 0
        for mtkMesh in mtkMeshes {
            guard let vtxBuf = mtkMesh.vertexBuffers.first?.buffer else { continue }
            for submesh in mtkMesh.submeshes {
                let edges = GltfModel.buildEdgeBuffer(
                    indexBuffer: submesh.indexBuffer.buffer,
                    indexCount: submesh.indexCount,
                    device: device
                )
                loadedPrimitives.append(Primitive(
                    vertexBuffer: vtxBuf,
                    indexBuffer:  submesh.indexBuffer.buffer,
                    indexCount:   submesh.indexCount,
                    materialIndex: min(materialIndex, loadedMaterials.count - 1),
                    edgeBuffer:   edges.buffer,
                    edgeCount:    edges.count
                ))
                materialIndex += 1
            }
        }

        primitives = loadedPrimitives
        materials  = loadedMaterials
        print("[GltfModel] \(config.name) — \(loadedPrimitives.count) primitives, \(loadedMaterials.count) materials")

        // ---- Per-model config overrides ----
        if let w = config.waterlineWidth { waterlineWidth = w }
    }

    // MARK: - Drawable

    nonisolated func encode(
        into encoder: MTLRenderCommandEncoder,
        modelMatrix: simd_float4x4,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {
        encodePasses(into: encoder, modelMatrix: modelMatrix, scene: scene,
                     isReflection: false)
    }
    
    nonisolated func encodeReflection(
        into encoder: MTLRenderCommandEncoder,
        modelMatrix: simd_float4x4,
        scene: SceneUniforms,
        surface: SurfaceUniforms
    ) {
        encodePasses(into: encoder, modelMatrix: modelMatrix, scene: scene,
                     isReflection: true)
    }

    // MARK: - Private encode

    private func encodePasses(
        into encoder: MTLRenderCommandEncoder,
        modelMatrix: simd_float4x4,
        scene: SceneUniforms,
        isReflection: Bool
    ) {
        var scn = scene
        var mdl = modelMatrix

        for prim in primitives {
            let mat = (prim.materialIndex >= 0 && prim.materialIndex < materials.count)
                    ? materials[prim.materialIndex] : Material()

            encoder.setRenderPipelineState(mainPipeline)
            encoder.setVertexBuffer(prim.vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&scn, length: MemoryLayout<SceneUniforms>.size,  index: 1)
            encoder.setVertexBytes(&mdl, length: MemoryLayout<simd_float4x4>.size,  index: 2)
            encoder.setFragmentBytes(&scn, length: MemoryLayout<SceneUniforms>.size, index: 1)
            encoder.setFragmentTexture(mat.colorTex,  index: 0)
            encoder.setFragmentTexture(mat.normalTex, index: 1)
            encoder.setFragmentTexture(mat.mrTex,     index: 2)
            encoder.setFragmentTexture(envMap,        index: 3)
            encoder.setFragmentSamplerState(sampler, index: 0)

            if isReflection {
                // Reflection pass: single solid draw, no clip
                var mu = makeModelUniforms(clip: 0)
                encoder.setFragmentBytes(&mu, length: MemoryLayout<ModelUniforms>.size, index: 3)
                encoder.setDepthStencilState(depthState)
                encoder.drawIndexedPrimitives(type: .triangle,
                                              indexCount: prim.indexCount,
                                              indexType: .uint32,
                                              indexBuffer: prim.indexBuffer,
                                              indexBufferOffset: 0)
            } else {
                // Main pass 1: solid triangles above waterline
                var mu1 = makeModelUniforms(clip: 1)
                encoder.setFragmentBytes(&mu1, length: MemoryLayout<ModelUniforms>.size, index: 3)
                encoder.setDepthStencilState(depthState)
                encoder.drawIndexedPrimitives(type: .triangle,
                                              indexCount: prim.indexCount,
                                              indexType: .uint32,
                                              indexBuffer: prim.indexBuffer,
                                              indexBufferOffset: 0)

                // Main pass 2: edge lines below waterline
                var mu2 = makeModelUniforms(clip: -1)
                encoder.setFragmentBytes(&mu2, length: MemoryLayout<ModelUniforms>.size, index: 3)
                encoder.setDepthStencilState(wireDepthState)
                encoder.drawIndexedPrimitives(type: .line,
                                              indexCount: prim.edgeCount,
                                              indexType: .uint32,
                                              indexBuffer: prim.edgeBuffer,
                                              indexBufferOffset: 0)
            }
        }
    }

    private func makeModelUniforms(clip: Float) -> ModelUniforms {
        var mu = ModelUniforms()
        mu.wireframeColor   = wireframeColor
        mu.waterlineClip    = clip
        mu.waterlineBias    = waterlineBias
        mu.waterlineWidth   = waterlineWidth
        mu.waterlineStrength = waterlineStrength
        mu.waterlineNoise   = waterlineNoise
        mu.wetStrength      = wetStrength
        mu.specularFactor   = specularFactor
        mu.bumpFactor       = bumpFactor
        return mu
    }

    // MARK: - Helpers

    /// Loads a texture from an MDLMaterial property; returns nil on failure
    /// (the shader handles nil textures via fallback behaviour in the pipeline).
    private static func loadTexture(
        from material: MDLMaterial,
        semantic: MDLMaterialSemantic,
        loader: MTKTextureLoader,
        sRGB: Bool
    ) -> MTLTexture? {
        guard let prop = material.property(with: semantic),
              prop.type == .texture,
              let mdlTex = prop.textureSamplerValue?.texture
        else { return nil }
        let opts: [MTKTextureLoader.Option: Any] = [
            .generateMipmaps: true,
            .SRGB: sRGB
        ]
        return try? loader.newTexture(texture: mdlTex, options: opts)
    }

    /// Builds a deduplicated edge (line) index buffer from a triangle index buffer.
    /// Matches GltfModel::generateWireframe() in the C++ reference.
    private static func buildEdgeBuffer(
        indexBuffer: MTLBuffer,
        indexCount: Int,
        device: MTLDevice
    ) -> (buffer: MTLBuffer, count: Int) {
        let triCount = indexCount / 3
        let ptr = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: indexCount)

        var edges = Set<UInt64>()
        for t in 0..<triCount {
            let a = ptr[t * 3], b = ptr[t * 3 + 1], c = ptr[t * 3 + 2]
            for (u, v) in [(a,b),(b,c),(a,c)] {
                let lo = min(u,v), hi = max(u,v)
                edges.insert(UInt64(lo) << 32 | UInt64(hi))
            }
        }

        var lineIdxs = [UInt32]()
        lineIdxs.reserveCapacity(edges.count * 2)
        for e in edges {
            lineIdxs.append(UInt32(e >> 32))
            lineIdxs.append(UInt32(e & 0xFFFFFFFF))
        }

        let buf = device.makeBuffer(bytes: lineIdxs,
                                    length: lineIdxs.count * MemoryLayout<UInt32>.size,
                                    options: .storageModeShared)!
        return (buf, lineIdxs.count)
    }

    /// Creates a 1×1 RGBA8 texture with the given byte values.
    /// Used as a fallback when a GLB has no embedded texture for a given semantic.
    private static func make1x1(device: MTLDevice,
                                rgba: (UInt8, UInt8, UInt8, UInt8)) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        desc.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        var pixel = [rgba.0, rgba.1, rgba.2, rgba.3]
        tex.replace(region: MTLRegionMake2D(0, 0, 1, 1),
                    mipmapLevel: 0,
                    withBytes: &pixel,
                    bytesPerRow: 4)
        return tex
    }
}

// MARK: - Errors

enum GltfModelError: Error {
    case initFailed(String)
    case loadFailed(String)
}
