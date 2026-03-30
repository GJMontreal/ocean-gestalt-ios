import Metal
import simd

struct MeshBuffers {
    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let indexCount: Int
}

struct OceanVertex {
    var position: SIMD3<Float>
    var texCoord: SIMD2<Float>
}

enum MeshBuilder {
    static func buildGrid(device: MTLDevice, size: Float = 200.0, resolution: Int = 200) -> MeshBuffers? {
        let vertexCount = (resolution + 1) * (resolution + 1)
        var vertices = [OceanVertex]()
        vertices.reserveCapacity(vertexCount)

        let step = size / Float(resolution)
        let origin = -size / 2.0

        for z in 0...resolution {
            for x in 0...resolution {
                let px = origin + Float(x) * step
                let pz = origin + Float(z) * step
                let u = Float(x) / Float(resolution)
                let v = Float(z) / Float(resolution)
                vertices.append(OceanVertex(
                    position: SIMD3<Float>(px, 0.0, pz),
                    texCoord: SIMD2<Float>(u, v)
                ))
            }
        }

        var indices = [UInt32]()
        indices.reserveCapacity(resolution * resolution * 6)

        for z in 0..<resolution {
            for x in 0..<resolution {
                let topLeft     = UInt32(z * (resolution + 1) + x)
                let topRight    = topLeft + 1
                let bottomLeft  = topLeft + UInt32(resolution + 1)
                let bottomRight = bottomLeft + 1

                indices.append(topLeft)
                indices.append(bottomLeft)
                indices.append(topRight)
                indices.append(topRight)
                indices.append(bottomLeft)
                indices.append(bottomRight)
            }
        }

        guard let vb = device.makeBuffer(bytes: vertices,
                                         length: vertices.count * MemoryLayout<OceanVertex>.stride,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(bytes: indices,
                                         length: indices.count * MemoryLayout<UInt32>.stride,
                                         options: .storageModeShared)
        else { return nil }

        return MeshBuffers(vertexBuffer: vb, indexBuffer: ib, indexCount: indices.count)
    }
}
