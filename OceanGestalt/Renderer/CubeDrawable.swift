import Metal
import simd

/// A unit cube at origin rendered via the Drawable protocol.
/// Uses the same SceneUniforms + modelMatrix binding convention as GltfModel,
/// so it exercises the full scene encode path.
final class CubeDrawable: @MainActor Drawable {

    private let pipeline:   MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let vertexBuffer: MTLBuffer
    private let indexBuffer:  MTLBuffer

    init(device: MTLDevice,
         library: MTLLibrary,
         colorPixelFormat: MTLPixelFormat,
         depthPixelFormat: MTLPixelFormat) throws {

        guard let vertFn = library.makeFunction(name: "cubeVertex"),
              let fragFn = library.makeFunction(name: "cubeFragment")
        else { throw CubeDrawableError.missingShaders }

        let pd = MTLRenderPipelineDescriptor()
        pd.vertexFunction                = vertFn
        pd.fragmentFunction              = fragFn
        pd.colorAttachments[0].pixelFormat = colorPixelFormat
        pd.depthAttachmentPixelFormat    = depthPixelFormat
        pipeline = try device.makeRenderPipelineState(descriptor: pd)

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled  = true
        guard let ds = device.makeDepthStencilState(descriptor: dd) else {
            throw CubeDrawableError.depthState
        }
        depthState = ds

        let positions: [SIMD4<Float>] = [
            SIMD4(-0.5, -0.5, -0.5, 0), SIMD4( 0.5, -0.5, -0.5, 0),
            SIMD4( 0.5,  0.5, -0.5, 0), SIMD4(-0.5,  0.5, -0.5, 0),
            SIMD4(-0.5, -0.5,  0.5, 0), SIMD4( 0.5, -0.5,  0.5, 0),
            SIMD4( 0.5,  0.5,  0.5, 0), SIMD4(-0.5,  0.5,  0.5, 0),
        ]
        let indices: [UInt16] = [
            0,2,1, 0,3,2,  4,5,6, 4,6,7,
            0,1,5, 0,5,4,  3,6,2, 3,7,6,
            0,7,3, 0,4,7,  1,2,6, 1,6,5,
        ]
        guard let vb = device.makeBuffer(bytes: positions,
                                         length: MemoryLayout<SIMD4<Float>>.stride * positions.count,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(bytes: indices,
                                         length: MemoryLayout<UInt16>.stride * indices.count,
                                         options: .storageModeShared)
        else { throw CubeDrawableError.buffers }
        vertexBuffer = vb
        indexBuffer  = ib
    }

    // MARK: - Drawable

    func encode(into enc: MTLRenderCommandEncoder,
                modelMatrix: simd_float4x4,
                scene: SceneUniforms,
                surface: SurfaceUniforms) {
        var scn = scene
        var mdl = modelMatrix
        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(depthState)
        enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        enc.setVertexBytes(&scn, length: MemoryLayout<SceneUniforms>.size,  index: 1)
        enc.setVertexBytes(&mdl, length: MemoryLayout<simd_float4x4>.size,  index: 2)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: 36,
                                  indexType: .uint16,
                                  indexBuffer: indexBuffer, indexBufferOffset: 0)
    }
}

enum CubeDrawableError: Error {
    case missingShaders
    case depthState
    case buffers
}
