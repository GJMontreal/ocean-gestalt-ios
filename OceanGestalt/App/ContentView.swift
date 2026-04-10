import SwiftUI
import MetalKit
import GestureCamera
import GestureCameraMetalKit

// MARK: - Root view

struct ContentView: View {
    @StateObject private var engine = OceanEngine()

    var body: some View {
        ZStack {
            if let camera = engine.camera {
                MetalView(engine: engine)
                    .ignoresSafeArea()

                WASDOverlayView(controller: camera)
                    .ignoresSafeArea()

                // Pause / mute buttons in the top-left corner
                VStack {
                    HStack {
                        Button(engine.isRunning ? "Pause" : "Resume") {
                            engine.togglePause()
                        }
                        .buttonStyle(.borderedProminent)
                        Button(engine.isMuted ? "Unmute" : "Mute") {
                            engine.toggleMute()
                        }
                        .buttonStyle(.bordered)
                        Spacer()
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 60)
                    Spacer()
                }
            }
        }
        .background(Color.black)
    }
}

// MARK: - Engine (owns renderer, camera, audio)

@MainActor
final class OceanEngine: ObservableObject {
    @Published var isRunning = true
    @Published var isMuted   = false

    private(set) var renderer: OceanRenderer?
    private(set) var camera:   GestureCameraController?
    private(set) var adapter:  MetalKitCameraAdapter?

    let device: MTLDevice

    init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        device = dev

        do {
            let r = try OceanRenderer(device: dev)
            r.audio = SurfAudio()
            renderer = r

            camera = GestureCameraController(
                initialTransform: r.initialCameraTransform)
        } catch {
            print("OceanEngine: init failed — \(error)")
        }
    }

    // Called from MetalView.makeUIView once the MTKView exists
    func setup(view: MTKView) {
        guard let renderer, let camera else { return }

        view.device                  = device
        view.colorPixelFormat        = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount             = 1
        view.clearColor              = MTLClearColorMake(0, 0.05, 0.1, 1)

        let adp = MetalKitCameraAdapter(controller: camera, view: view)
        adp.onDraw = { [weak self] v, transform in
            guard let self, let r = self.renderer else { return }
            r.isRunning = self.isRunning
            r.draw(view: v, camera: transform)
        }
        adp.start()
        adapter = adp

        if !isMuted { renderer.audio?.start() }
    }

    func togglePause() {
        isRunning.toggle()
        if isRunning { renderer?.audio?.start() } else { renderer?.audio?.stop() }
    }

    func toggleMute() {
        isMuted.toggle()
        if isMuted { renderer?.audio?.stop() } else { renderer?.audio?.start() }
    }
}

// MARK: - Metal view

struct MetalView: UIViewRepresentable {
    let engine: OceanEngine

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: engine.device)
        engine.setup(view: view)

        // Pan → camera rotation (single finger)
        let pan = UIPanGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handlePan(_:)))
        pan.maximumNumberOfTouches = 1
        view.addGestureRecognizer(pan)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(engine: engine) }

    @MainActor
    final class Coordinator: NSObject {
        private let engine: OceanEngine
        private var lastTranslation: CGPoint = .zero

        init(engine: OceanEngine) { self.engine = engine }

        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let camera = engine.camera else { return }
            switch gesture.state {
            case .began:
                lastTranslation = .zero
            case .changed:
                let t  = gesture.translation(in: gesture.view)
                let dx = Float(t.x - lastTranslation.x)
                let dy = Float(t.y - lastTranslation.y)
                lastTranslation = t
                camera.applyRotationGesture(dx: dx, dy: dy)
            default:
                break
            }
        }
    }
}
