import Foundation

// Port of C++ UniformAnimator. Linear interpolation, key-based deduplication.
// All methods are called from the main/render thread except animateTo which is thread-safe.

enum AnimatableValue {
    case scalar(Float)
    case vector([Float])

    static func lerp(_ a: AnimatableValue, _ b: AnimatableValue, _ t: Float) -> AnimatableValue {
        switch (a, b) {
        case (.scalar(let fa), .scalar(let fb)):
            return .scalar(fa + (fb - fa) * t)
        case (.vector(let va), .vector(let vb)):
            let result = zip(va, vb).map { $0 + ($1 - $0) * t }
            return .vector(result)
        default:
            return t >= 1.0 ? b : a
        }
    }
}

private struct PendingAnimation {
    var key: String
    var from: AnimatableValue
    var to: AnimatableValue
    var duration: Float
    var apply: (AnimatableValue) -> Void
}

private struct ActiveAnimation {
    var key: String
    var from: AnimatableValue
    var to: AnimatableValue
    var elapsed: Float = 0
    var duration: Float
    var apply: (AnimatableValue) -> Void
}

final class UniformAnimator {
    private var pending: [PendingAnimation] = []
    private var active: [ActiveAnimation] = []
    private var lastTime: Float = -1
    private let lock = NSLock()

    // Thread-safe enqueue
    func animateTo(_ from: AnimatableValue, _ to: AnimatableValue, duration: Float,
                   apply: @escaping (AnimatableValue) -> Void, key: String = "") {
        lock.lock()
        pending.append(PendingAnimation(key: key, from: from, to: to, duration: duration, apply: apply))
        lock.unlock()
    }

    func cancel(key: String) {
        lock.lock()
        pending.removeAll { $0.key == key }
        lock.unlock()
        active.removeAll { $0.key == key }
    }

    // Call once per frame from render thread
    func tick(time: Float) {
        var dt: Float = 0
        if lastTime >= 0 && time > lastTime {
            dt = time - lastTime
        }
        lastTime = time

        // Merge pending
        lock.lock()
        let arrivals = pending
        pending.removeAll()
        lock.unlock()

        for p in arrivals {
            if !p.key.isEmpty, let idx = active.firstIndex(where: { $0.key == p.key }) {
                let t = min(active[idx].elapsed / active[idx].duration, 1.0)
                active[idx].from     = AnimatableValue.lerp(active[idx].from, active[idx].to, t)
                active[idx].to       = p.to
                active[idx].elapsed  = 0
                active[idx].duration = p.duration
                active[idx].apply    = p.apply
            } else {
                active.append(ActiveAnimation(key: p.key, from: p.from, to: p.to,
                                              duration: p.duration, apply: p.apply))
            }
        }

        for i in active.indices {
            active[i].elapsed += dt
            let t = active[i].duration > 0 ? min(active[i].elapsed / active[i].duration, 1.0) : 1.0
            active[i].apply(AnimatableValue.lerp(active[i].from, active[i].to, t))
        }

        active.removeAll { $0.elapsed >= $0.duration }
    }
}
