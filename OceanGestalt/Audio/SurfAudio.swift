import AVFoundation
import simd

/// Procedural stereo surf audio. Matches SurfAudio.cpp — bandpass-filtered
/// white noise driven by Gerstner wave steepness sampled at two points
/// flanking the camera (L and R, ~5 units apart along the camera's right axis).
///
/// Uses AVAudioEngine with a custom source node so all DSP runs on the
/// audio thread. State is shared via atomics.
@MainActor
final class SurfAudio {

    private static let sampleRate: Double    = 44100
    private static let horizontalOffset: Float = 10
    private static let gain: Float           = 1.0

    // Steepness values written on main thread, read on audio thread
    private var steepnessLeft:  Float = 0  // protected by _steepness atomic via sendable wrapper
    private var steepnessRight: Float = 0
    private let steepnessLock = NSLock()

    private let engine     = AVAudioEngine()
    private var sourceNode: AVAudioSourceNode!

    private var leftFilter  = BandpassFilter(sampleRate: sampleRate, frequency: 600,  q: 0.7)
    private var rightFilter = BandpassFilter(sampleRate: sampleRate, frequency: 1200, q: 1.0)

    private var isStarted = false

    init() {
        let format = AVAudioFormat(standardFormatWithSampleRate: SurfAudio.sampleRate,
                                   channels: 2)!
        sourceNode = AVAudioSourceNode(format: format) { [weak self] _, _, frameCount, bufferList in
            guard let self else { return noErr }
            let left  = self.readSteepnessLeft()
            let right = self.readSteepnessRight()

            let blf = UnsafeMutableAudioBufferListPointer(bufferList)
            guard blf.count >= 2 else { return noErr }

            let l = blf[0].mData!.assumingMemoryBound(to: Float.self)
            let r = blf[1].mData!.assumingMemoryBound(to: Float.self)

            for i in 0..<Int(frameCount) {
                let noiseL = Float.random(in: -1...1)
                let noiseR = Float.random(in: -1...1)
                l[i] = self.leftFilter.process(noiseL)  * left
                r[i] = self.rightFilter.process(noiseR) * right
            }
            return noErr
        }

        engine.attach(sourceNode)
        engine.connect(sourceNode, to: engine.mainMixerNode, format: format)
    }

    // MARK: - Control

    func start() {
        guard !isStarted else { return }
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback,
                                                            mode: .default,
                                                            options: [.mixWithOthers])
            try AVAudioSession.sharedInstance().setActive(true)
            try engine.start()
            isStarted = true
        } catch {
            print("SurfAudio: start failed — \(error)")
        }
    }

    func stop() {
        guard isStarted else { return }
        engine.stop()
        isStarted = false
    }

    // MARK: - Per-frame steepness update

    /// Called each frame. `yaw` in radians (camera yaw).
    func generateSurf(waves: [WaveState], position: SIMD3<Float>, yaw: Float, time: Float) {
        let (leftXZ, rightXZ) = offsetPoints(position: SIMD2<Float>(position.x, position.z),
                                              yaw: yaw)
        let ls = steepness(waves: waves, xz: leftXZ,  time: time)
        let rs = steepness(waves: waves, xz: rightXZ, time: time)
        setSteepness(left: ls * SurfAudio.gain, right: rs * SurfAudio.gain)
    }

    // MARK: - Private

    private func setSteepness(left: Float, right: Float) {
        steepnessLock.lock()
        steepnessLeft  = min(max(left,  0), 1)
        steepnessRight = min(max(right, 0), 1)
        steepnessLock.unlock()
    }

    private func readSteepnessLeft() -> Float {
        steepnessLock.lock(); defer { steepnessLock.unlock() }
        return steepnessLeft
    }

    private func readSteepnessRight() -> Float {
        steepnessLock.lock(); defer { steepnessLock.unlock() }
        return steepnessRight
    }

    private func offsetPoints(position: SIMD2<Float>, yaw: Float) -> (SIMD2<Float>, SIMD2<Float>) {
        let forward = SIMD2<Float>(sin(yaw), cos(yaw))
        let right   = SIMD2<Float>(forward.y, -forward.x)
        let half    = SurfAudio.horizontalOffset * 0.5
        return (position - right * half, position + right * half)
    }

    /// CPU Gerstner steepness proxy — matches getSteepness() in GerstnerWave.cpp.
    private func steepness(waves: [WaveState], xz: SIMD2<Float>, time: Float) -> Float {
        let eps: Float = 0.1
        let c  = gerstnerOffset(waves: waves, xz: xz,                          time: time)
        let dx = gerstnerOffset(waves: waves, xz: xz + SIMD2<Float>(eps, 0),   time: time)
        let dz = gerstnerOffset(waves: waves, xz: xz + SIMD2<Float>(0,   eps), time: time)
        return simd_length(dx - c) + simd_length(dz - c)
    }

    private func gerstnerOffset(waves: [WaveState], xz: SIMD2<Float>, time: Float) -> SIMD3<Float> {
        var offset = SIMD3<Float>.zero
        for wave in waves where wave.amplitude > 0 {
            let k = 2 * Float.pi / max(wave.wavelength, 0.01)
            let w = sqrt(9.81 * k)
            let len = simd_length(wave.direction)
            guard len > 1e-5 else { continue }
            let D = wave.direction / len
            let phase = simd_dot(D * k, xz)
                      - (w * time).truncatingRemainder(dividingBy: 2 * Float.pi)
                      + wave.phase
            offset.y += wave.amplitude * cos(phase)
            let xzD = -wave.steepness * D * sin(phase) * wave.amplitude
            offset.x += xzD.x
            offset.z += xzD.y
        }
        return offset
    }
}

// MARK: - Simple biquad bandpass filter

/// Direct Form II biquad bandpass (constant peak-gain).
/// Matches BandpassFilter in the C++ reference.
private struct BandpassFilter {
    private var b0, b1, b2: Float
    private var a1, a2:     Float
    private var z1: Float = 0
    private var z2: Float = 0

    init(sampleRate: Double, frequency: Float, q: Float) {
        let w0 = 2 * Float.pi * frequency / Float(sampleRate)
        let alpha = sin(w0) / (2 * q)
        let cosw0 = cos(w0)
        let a0inv = 1 / (1 + alpha)
        b0 =  alpha * a0inv
        b1 =  0
        b2 = -alpha * a0inv
        a1 = -2 * cosw0 * a0inv
        a2 =  (1 - alpha) * a0inv
    }

    mutating func process(_ x: Float) -> Float {
        let y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        return y
    }
}
