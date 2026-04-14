import SwiftUI
import MetalKit
import GestureCamera
import GestureCameraMetalKit

// MARK: - Root view

struct ContentView: View {
    @StateObject private var engine = OceanEngine()
    @State private var lastDragLocation: CGPoint = .zero
    @State private var showControls = true
    @State private var hideTask: Task<Void, Never>? = nil

    var body: some View {
        if let camera = engine.camera {
            ZStack {
                // 1 — Metal render surface
                MetalView(engine: engine)
                    .ignoresSafeArea()

                // 2 — Lens drop overlay (non-interactive)
                LensDropsView()

                // 3 — Single-finger drag → camera rotation.
                //     minimumDistance: 0 so a bare tap also calls touchDetected().
                //     Rotation is only applied when the delta is meaningful.
                Rectangle()
                    .fill(.clear)
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                touchDetected()
                                let prev = lastDragLocation == .zero
                                         ? value.startLocation : lastDragLocation
                                let dx = Float(value.location.x - prev.x)
                                let dy = Float(value.location.y - prev.y)
                                lastDragLocation = value.location
                                if abs(dx) > 0.5 || abs(dy) > 0.5 {
                                    camera.applyRotationGesture(dx: dx, dy: dy)
                                }
                            }
                            .onEnded { _ in lastDragLocation = .zero }
                    )
                    .ignoresSafeArea()

                // 4 — WASD d-pad + vertical + motion toggle (from GestureCamera).
                //     simultaneousGesture resets the hide timer while the WASD
                //     buttons are being used without disrupting their own gestures.
                WASDOverlayView(controller: camera)
                    .ignoresSafeArea()
                    .opacity(showControls ? 1 : 0)
                    .allowsHitTesting(showControls)
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { _ in touchDetected() }
                    )

                // 5 — Pause / mute: left side, same row as the motion toggle (20pt top)
                HStack(spacing: 8) {
                    controlButton(
                        systemImage: engine.isRunning ? "pause.fill" : "play.fill",
                        action: { touchDetected(); engine.togglePause() }
                    )
                    controlButton(
                        systemImage: engine.isMuted ? "speaker.slash.fill" : "speaker.fill",
                        action: { touchDetected(); engine.toggleMute() }
                    )
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .padding(.top, 20)
                .padding(.leading, 20)
                .opacity(showControls ? 1 : 0)
                .allowsHitTesting(showControls)
            }
            .background(Color.black)
            .onAppear { scheduleHide() }
        }
    }

    // MARK: - Controls visibility

    private func touchDetected() {
        if !showControls {
            withAnimation(.easeInOut(duration: 0.2)) { showControls = true }
        }
        scheduleHide()
    }

    private func scheduleHide() {
        hideTask?.cancel()
        hideTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(3))
                withAnimation(.easeInOut(duration: 0.5)) { showControls = false }
            } catch {}
        }
    }

    // MARK: - Control button style

    @ViewBuilder
    private func controlButton(systemImage: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 22, weight: .medium))
                .frame(width: 52, height: 52)
                .background(.black.opacity(0.55))
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 10))
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

            let cam = GestureCameraController(
                initialTransform: r.initialCameraTransform)
            cam.onCustomKey = { [weak r] key in
                guard let r else { return false }
                switch key.keyCode {
                case .keyboardM:
                    r.showMesh.toggle()
                    return true
                case .keyboardN:
                    r.showNormals.toggle()
                    return true
                default:
                    return false
                }
            }
            camera = cam
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
