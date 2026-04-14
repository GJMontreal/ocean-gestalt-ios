import Metal

/// Owns the depth-only texture used for the shadow pre-pass.
/// Mirrors ShadowPass.cpp — allocate once at `size × size`, render the
/// scene from the light's point of view into the depth texture, then
/// bind that texture for PCF shadow lookup in the ocean fragment shader.
@MainActor
final class ShadowPass {

    private(set) var depthTexture:  MTLTexture?
    private(set) var passDescriptor: MTLRenderPassDescriptor?

    private let device: MTLDevice
    private let size:   Int

    init(device: MTLDevice, size: Int) {
        self.device = device
        self.size   = size
        rebuild()
    }

    // MARK: - Private

    private func rebuild() {
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width:  size,
            height: size,
            mipmapped: false
        )
        texDesc.usage       = [.renderTarget, .shaderRead]
        texDesc.storageMode = .private

        guard let depth = device.makeTexture(descriptor: texDesc) else { return }
        depth.label = "ShadowDepth"

        let rpd = MTLRenderPassDescriptor()
        // Depth-only: no colour attachment.
        rpd.depthAttachment.texture     = depth
        rpd.depthAttachment.loadAction  = .clear
        rpd.depthAttachment.storeAction = .store   // must be .store so the shader can read it
        rpd.depthAttachment.clearDepth  = 1.0

        depthTexture    = depth
        passDescriptor  = rpd
    }
}
