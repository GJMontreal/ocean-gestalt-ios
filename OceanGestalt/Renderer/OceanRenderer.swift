import Metal
import MetalKit
import simd

@MainActor
final class OceanRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState
    private let depthStencilState: MTLDepthStencilState
    private let mesh: MeshBuffers

    let uniformState: UniformState

    // Camera state
    var cameraYaw: Float   = 0.0       // radians, horizontal
    var cameraPitch: Float = -0.3      // radians, looking slightly down
    var cameraDistance: Float = 30.0
    var cameraHeight: Float   = 15.0

    // Movement
    var moveForward: Float = 0
    var moveRight: Float   = 0
    var moveUp: Float      = 0
    private var cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 15, 30)

    private var startTime: Date = Date()

    init?(mtkView: MTKView) {
        guard let device = mtkView.device,
              let queue = device.makeCommandQueue()
        else { return nil }

        self.device = device
        self.commandQueue = queue
        self.uniformState = UniformState()

        guard let mesh = MeshBuilder.buildGrid(device: device) else { return nil }
        self.mesh = mesh

        // Pipeline
        guard let library = device.makeDefaultLibrary() else { return nil }
        let vertFn = library.makeFunction(name: "oceanVertex")
        let fragFn = library.makeFunction(name: "oceanFragment")

        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<OceanVertex>.stride

        let pipelineDesc = MTLRenderPipelineDescriptor()
        pipelineDesc.vertexFunction   = vertFn
        pipelineDesc.fragmentFunction = fragFn
        pipelineDesc.vertexDescriptor = vertexDescriptor
        pipelineDesc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        pipelineDesc.depthAttachmentPixelFormat      = mtkView.depthStencilPixelFormat

        guard let ps = try? device.makeRenderPipelineState(descriptor: pipelineDesc) else { return nil }
        pipelineState = ps

        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .less
        depthDesc.isDepthWriteEnabled  = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else { return nil }
        depthStencilState = ds

        super.init()
    }

    // MARK: - MTKViewDelegate

    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    nonisolated func draw(in view: MTKView) {
        Task { @MainActor in
            self.drawFrame(in: view)
        }
    }

    private func drawFrame(in view: MTKView) {
        let time = Float(Date().timeIntervalSince(startTime))
        uniformState.tick(time: time)

        // Move camera
        let speed: Float = 5.0 * (1.0/60.0)
        let forward = SIMD3<Float>(-sin(cameraYaw), 0, -cos(cameraYaw))
        let right   = SIMD3<Float>( cos(cameraYaw), 0, -sin(cameraYaw))
        cameraPosition += forward * moveForward * speed
        cameraPosition += right   * moveRight   * speed
        cameraPosition.y += moveUp * speed

        let target = cameraPosition + SIMD3<Float>(
            sin(cameraYaw) * cos(cameraPitch),
            sin(cameraPitch),
            -cos(cameraYaw) * cos(cameraPitch)
        ) * 10.0

        let viewMatrix = lookAt(eye: cameraPosition, center: target, up: SIMD3<Float>(0,1,0))
        let aspect = Float(view.drawableSize.width / view.drawableSize.height)
        let projMatrix = perspective(fovY: Float.pi / 4.0, aspect: aspect, near: 0.1, far: 500.0)
        let modelMatrix = matrix_identity_float4x4

        var sceneUniforms = uniformState.buildSceneUniforms(
            time: time,
            model: modelMatrix,
            view: viewMatrix,
            projection: projMatrix,
            cameraPos: cameraPosition
        )
        var surfaceUniforms = uniformState.buildSurfaceUniforms()

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let passDesc = view.currentRenderPassDescriptor,
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDesc)
        else { return }

        passDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0.05, green: 0.1, blue: 0.15, alpha: 1)

        encoder.setRenderPipelineState(pipelineState)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&sceneUniforms,  length: MemoryLayout<SceneUniforms>.size, index: 1)
        encoder.setFragmentBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.size, index: 1)
        encoder.setFragmentBytes(&surfaceUniforms, length: MemoryLayout<SurfaceUniforms>.size, index: 2)

        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: mesh.indexCount,
            indexType: .uint32,
            indexBuffer: mesh.indexBuffer,
            indexBufferOffset: 0
        )

        encoder.endEncoding()
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
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
            SIMD4<Float>(-dot(r,eye), -dot(u,eye), dot(f,eye), 1)
        )
    }

    private func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let y = 1 / tan(fovY * 0.5)
        let x = y / aspect
        let z = far / (near - far)
        return simd_float4x4(
            SIMD4<Float>(x,  0,  0,  0),
            SIMD4<Float>(0,  y,  0,  0),
            SIMD4<Float>(0,  0,  z, -1),
            SIMD4<Float>(0,  0,  z*near, 0)
        )
    }
}
