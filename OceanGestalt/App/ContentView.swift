import SwiftUI
import MetalKit

struct MetalView: UIViewRepresentable {
    let renderer: OceanRenderer

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = renderer
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.05, green: 0.1, blue: 0.15, alpha: 1)
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = false

        // Touch for look
        let pan = UIPanGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handlePan(_:)))
        view.addGestureRecognizer(pan)

        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(renderer: renderer) }

    @MainActor
    final class Coordinator: NSObject {
        let renderer: OceanRenderer
        private var lastTranslation: CGPoint = .zero

        init(renderer: OceanRenderer) { self.renderer = renderer }

        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            let translation = gesture.translation(in: gesture.view)
            if gesture.state == .began { lastTranslation = .zero }
            let delta = CGPoint(x: translation.x - lastTranslation.x,
                                y: translation.y - lastTranslation.y)
            lastTranslation = translation
            renderer.cameraYaw   -= Float(delta.x) * 0.005
            renderer.cameraPitch += Float(delta.y) * 0.005
            renderer.cameraPitch  = max(-Float.pi/2 + 0.05, min(Float.pi/2 - 0.05, renderer.cameraPitch))
        }
    }
}

struct ContentView: View {
    @State private var renderer: OceanRenderer?
    @State private var overlayVisible = false
    @State private var overlayTimer: Timer?

    var body: some View {
        ZStack {
            if let renderer {
                MetalView(renderer: renderer)
                    .ignoresSafeArea()
                    .onTapGesture { showOverlay() }

                if overlayVisible {
                    MovementOverlay(renderer: renderer)
                        .transition(.opacity)
                }
            } else {
                Color.black.ignoresSafeArea()
                ProgressView().tint(.white)
            }
        }
        .onAppear { setupRenderer() }
        .onTapGesture { showOverlay() }
        .animation(.easeInOut(duration: 0.3), value: overlayVisible)
    }

    private func setupRenderer() {
        let mtkView = MTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .depth32Float
        renderer = OceanRenderer(mtkView: mtkView)
    }

    private func showOverlay() {
        overlayVisible = true
        overlayTimer?.invalidate()
        overlayTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: false) { _ in
            withAnimation { overlayVisible = false }
        }
    }
}

struct MovementOverlay: View {
    let renderer: OceanRenderer

    var body: some View {
        VStack {
            Spacer()
            HStack(alignment: .bottom) {
                // Left: WASD
                VStack(spacing: 0) {
                    HoldButton(label: "▲\nW") { renderer.moveForward =  1 } onRelease: { renderer.moveForward = 0 }
                    HStack(spacing: 0) {
                        HoldButton(label: "◀\nA") { renderer.moveRight = -1 } onRelease: { renderer.moveRight = 0 }
                        Spacer().frame(width: 50, height: 50)
                        HoldButton(label: "▶\nD") { renderer.moveRight =  1 } onRelease: { renderer.moveRight = 0 }
                    }
                    HoldButton(label: "▼\nS") { renderer.moveForward = -1 } onRelease: { renderer.moveForward = 0 }
                }

                Spacer()

                // Right: Up/Down
                VStack(spacing: 10) {
                    HoldButton(label: "⇑\nSpc") { renderer.moveUp =  1 } onRelease: { renderer.moveUp = 0 }
                    HoldButton(label: "⇓\nShift") { renderer.moveUp = -1 } onRelease: { renderer.moveUp = 0 }
                }
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 30)
        }
    }
}

struct HoldButton: View {
    let label: String
    let onPress: () -> Void
    let onRelease: () -> Void

    var body: some View {
        Text(label)
            .font(.system(size: 11))
            .multilineTextAlignment(.center)
            .foregroundColor(.white)
            .frame(width: 50, height: 50)
            .background(Color.black.opacity(0.5))
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in onPress() }
                    .onEnded   { _ in onRelease() }
            )
    }
}
