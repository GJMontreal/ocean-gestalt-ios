import Foundation

/// Drives smooth parameter transitions. Mirrors src/api/UniformAnimator.cpp.
/// Single-threaded (@MainActor) — no mutex needed.
@MainActor
final class UniformAnimator {

    private struct Animation {
        let key: String
        var from: Float
        let to: Float
        var elapsed: Float
        let duration: Float
        let apply: (Float) -> Void
    }

    private var active: [Animation] = []

    /// Queue a float transition. If an animation with the same non-empty key
    /// is already running, it is replaced starting from its current value.
    func animateTo(_ from: Float, _ to: Float, duration: Float,
                   key: String = "", apply: @escaping (Float) -> Void) {
        if !key.isEmpty, let i = active.firstIndex(where: { $0.key == key }) {
            // Capture current interpolated value as new start
            let t = min(active[i].elapsed / active[i].duration, 1)
            let currentVal = active[i].from + (active[i].to - active[i].from) * t
            active[i] = Animation(key: key, from: currentVal, to: to,
                                  elapsed: 0, duration: duration, apply: apply)
        } else {
            active.append(Animation(key: key, from: from, to: to,
                                    elapsed: 0, duration: duration, apply: apply))
        }
    }

    func cancel(key: String) {
        active.removeAll { $0.key == key }
    }

    /// Advance all animations and fire apply callbacks. Call once per frame.
    func tick(deltaTime dt: Float) {
        for i in active.indices {
            active[i].elapsed += dt
            let t = active[i].duration > 0
                  ? min(active[i].elapsed / active[i].duration, 1)
                  : 1
            let val = active[i].from + (active[i].to - active[i].from) * t
            active[i].apply(val)
        }
        active.removeAll { $0.elapsed >= $0.duration }
    }
}
