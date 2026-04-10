import simd

/// Position and physics flags for a scene object. No rendering knowledge.
/// Mirrors src/core/include/Moveable.hpp.
struct Moveable {
    var name: String
    var position: SIMD3<Float>  // base world position (XZ from scene.json; Y = offset above surface)
    var isFloating: Bool
    var isTethered: Bool
}
