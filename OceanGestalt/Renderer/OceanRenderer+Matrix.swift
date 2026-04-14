// MARK: - Matrix helpers

extension OceanRenderer {
    func perspectiveMatrix(fovY: Float, aspect: Float,
                                   nearZ: Float, farZ: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY * 0.5)
        let xs = ys / aspect
        let zs = farZ / (nearZ - farZ)
        return simd_float4x4(columns: (
            SIMD4<Float>(xs,  0,  0,  0),
            SIMD4<Float>( 0, ys,  0,  0),
            SIMD4<Float>( 0,  0, zs, -1),
            SIMD4<Float>( 0,  0, zs * nearZ, 0)
        ))
    }

    func orthographicMatrix(left: Float, right: Float,
                                    bottom: Float, top: Float,
                                    near: Float, far: Float) -> simd_float4x4 {
        let rl = right - left, tb = top - bottom, fn = far - near
        return simd_float4x4(columns: (
            SIMD4<Float>(2/rl,            0,            0, 0),
            SIMD4<Float>(   0,         2/tb,            0, 0),
            SIMD4<Float>(   0,            0,        -1/fn, 0),
            SIMD4<Float>(-(right+left)/rl, -(top+bottom)/tb, -near/fn, 1)
        ))
    }

    func lookAtMatrix(eye: SIMD3<Float>, target: SIMD3<Float>,
                              up: SIMD3<Float>) -> simd_float4x4 {
        let z = simd_normalize(eye - target)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)
        return simd_float4x4(columns: (
            SIMD4<Float>(x.x,            y.x,            z.x,            0),
            SIMD4<Float>(x.y,            y.y,            z.y,            0),
            SIMD4<Float>(x.z,            y.z,            z.z,            0),
            SIMD4<Float>(-simd_dot(x,eye), -simd_dot(y,eye), -simd_dot(z,eye), 1)
        ))
    }

}
