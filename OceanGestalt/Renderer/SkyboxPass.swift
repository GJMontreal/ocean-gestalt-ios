import Metal
import simd

/// Owns the skybox pipeline and encodes the full-screen cube draw.
/// Receives the shared env-map cubemap and clamp sampler from OceanRenderer
/// so that both the skybox and ocean fragment shader sample the same texture.
/// Mirrors src/core/Skybox.cpp.
final class SkyboxPass {

    private let pipeline:     MTLRenderPipelineState
    private let depthState:   MTLDepthStencilState
    private let envMap:       MTLTexture
    private let clampSampler: MTLSamplerState

    init(
        device: MTLDevice,
        library: MTLLibrary,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat,
        envMap: MTLTexture,
        clampSampler: MTLSamplerState
    ) throws {
        self.envMap       = envMap
        self.clampSampler = clampSampler

        let desc = MTLRenderPipelineDescriptor()
        desc.colorAttachments[0].pixelFormat = colorPixelFormat
        desc.depthAttachmentPixelFormat      = depthPixelFormat
        desc.vertexFunction   = library.makeFunction(name: "skyboxVertex")
        desc.fragmentFunction = library.makeFunction(name: "skyboxFragment")
        pipeline = try device.makeRenderPipelineState(descriptor: desc)

        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .lessEqual
        depthDesc.isDepthWriteEnabled  = false
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else {
            throw SkyboxPassError.initFailed("depth state")
        }
        depthState = ds
    }

    func encode(into encoder: MTLRenderCommandEncoder, scene: SceneUniforms) {
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        var scn = scene
        encoder.setVertexBytes(&scn, length: MemoryLayout<SceneUniforms>.size, index: 1)
        encoder.setFragmentTexture(envMap, index: 0)
        encoder.setFragmentSamplerState(clampSampler, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36)
    }
}

// MARK: - Errors

enum SkyboxPassError: Error {
    case initFailed(String)
}
