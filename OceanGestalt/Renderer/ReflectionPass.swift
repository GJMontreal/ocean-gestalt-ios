import Metal

/// Owns the off-screen color + depth textures used for the planar reflection
/// pre-pass. Mirrors ReflectionPass.cpp — allocate once, bind, draw into,
/// then sample the color texture from the ocean fragment shader.
@MainActor
final class ReflectionPass {

    private(set) var colorTexture: MTLTexture?
    private(set) var passDescriptor: MTLRenderPassDescriptor?

    private let device: MTLDevice
    private var size: Int
    private let pixelFormat: MTLPixelFormat

    init(device: MTLDevice, size: Int, pixelFormat: MTLPixelFormat = .bgra8Unorm) {
        self.device      = device
        self.size        = size
        self.pixelFormat = pixelFormat
        rebuild()
    }

    /// Call when the reflection resolution changes.
    func resize(size newSize: Int) {
        guard newSize != size else { return }
        size = newSize
        rebuild()
    }

    // MARK: - Private

    private func rebuild() {
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width:  size,
            height: size,
            mipmapped: false
        )
        texDesc.usage      = [.renderTarget, .shaderRead]
        texDesc.storageMode = .private

        guard let color = device.makeTexture(descriptor: texDesc) else { return }
        color.label = "ReflectionColor"

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width:  size,
            height: size,
            mipmapped: false
        )
        depthDesc.usage       = .renderTarget
        depthDesc.storageMode = .private

        guard let depth = device.makeTexture(descriptor: depthDesc) else { return }
        depth.label = "ReflectionDepth"

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture     = color
        rpd.colorAttachments[0].loadAction  = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0, 0.05, 0.1, 1)
        rpd.depthAttachment.texture         = depth
        rpd.depthAttachment.loadAction      = .clear
        rpd.depthAttachment.storeAction     = .dontCare
        rpd.depthAttachment.clearDepth      = 1.0

        colorTexture    = color
        passDescriptor  = rpd
    }
}
