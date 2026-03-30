import Metal
import MetalKit
import simd

@MainActor
final class OceanRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState
    private let buoyPipelineState: MTLRenderPipelineState
    private let depthStencilState: MTLDepthStencilState
    private let mesh: MeshBuffers
    private let buoyMesh: MeshBuffers

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

        guard let buoyMesh = MeshBuilder.buildSphere(device: device) else { return nil }
        self.buoyMesh = buoyMesh

        // Ocean pipeline
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

        // Buoy pipeline
        let buoyVertFn = library.makeFunction(name: "buoyVertex")
        let buoyFragFn = library.makeFunction(name: "buoyFragment")

        let buoyVertDesc = MTLVertexDescriptor()
        buoyVertDesc.attributes[0].format = .float3
        buoyVertDesc.attributes[0].offset = 0
        buoyVertDesc.attributes[0].bufferIndex = 0
        buoyVertDesc.attributes[1].format = .float3
        buoyVertDesc.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        buoyVertDesc.attributes[1].bufferIndex = 0
        buoyVertDesc.layouts[0].stride = MemoryLayout<BuoyVertex>.stride

        let buoyPipelineDesc = MTLRenderPipelineDescriptor()
        buoyPipelineDesc.vertexFunction   = buoyVertFn
        buoyPipelineDesc.fragmentFunction = buoyFragFn
        buoyPipelineDesc.vertexDescriptor = buoyVertDesc
        buoyPipelineDesc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        buoyPipelineDesc.depthAttachmentPixelFormat      = mtkView.depthStencilPixelFormat

        guard let bps = try? device.makeRenderPipelineState(descriptor: buoyPipelineDesc) else { return nil }
        buoyPipelineState = bps

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

        // Draw buoy
        var buoyModel = buoyModelMatrix(time: time)
        var surfaceUniformsCopy = surfaceUniforms
        encoder.setRenderPipelineState(buoyPipelineState)
        encoder.setVertexBuffer(buoyMesh.vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.size, index: 1)
        encoder.setVertexBytes(&buoyModel, length: MemoryLayout<simd_float4x4>.size, index: 2)
        encoder.setFragmentBytes(&surfaceUniformsCopy, length: MemoryLayout<SurfaceUniforms>.size, index: 2)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: buoyMesh.indexCount,
            indexType: .uint32,
            indexBuffer: buoyMesh.indexBuffer,
            indexBufferOffset: 0
        )

        encoder.endEncoding()
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

        // Gram-Schmidt: keep tangent perpendicular to up
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
