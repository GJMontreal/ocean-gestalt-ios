import SwiftUI
import MetalKit
import GestureCamera
import GestureCameraMetalKit

// MARK: - Root view

struct ContentView: View {
    @StateObject private var engine = OceanEngine()
    @State private var lastDragLocation: CGPoint = .zero

    var body: some View {
        if let camera = engine.camera {
            ZStack {
                // 1 — Metal render surface
                MetalView(engine: engine)
                    .ignoresSafeArea()

                // 2 — Single-finger drag → camera rotation
                //     Sits below WASDOverlayView so WASD button touches
                //     are handled by SwiftUI before reaching this layer.
                Rectangle()
                    .fill(.clear)
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 1)
                            .onChanged { value in
                                let dx = Float(value.location.x - lastDragLocation.x)
                                let dy = Float(value.location.y - lastDragLocation.y)
                                lastDragLocation = value.location
                                camera.applyRotationGesture(dx: dx, dy: dy)
                            }
                            .onEnded { _ in lastDragLocation = .zero }
                    )
                    .ignoresSafeArea()

                // 3 — WASD d-pad + vertical + motion toggle (from GestureCamera)
                WASDOverlayView(controller: camera)
                    .ignoresSafeArea()

                // 4 — Pause / mute
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
            .background(Color.black)
        }
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

// MARK: - Metal view (no gesture recognizers — handled in SwiftUI layer above)

struct MetalView: UIViewRepresentable {
    let engine: OceanEngine

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: engine.device)
        engine.setup(view: view)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}
