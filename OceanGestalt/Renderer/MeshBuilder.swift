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

struct BuoyVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
}

enum MeshBuilder {
    /// Returns (indexBuffer, lineCount) reusing the same vertex buffer as buildGrid.
    static func buildGridLines(device: MTLDevice, resolution: Int = 200) -> (MTLBuffer, Int)? {
        var indices = [UInt32]()
        indices.reserveCapacity(4 * resolution * (resolution + 1))
        // Horizontal edges
        for z in 0...resolution {
            for x in 0..<resolution {
                indices.append(UInt32(z * (resolution + 1) + x))
                indices.append(UInt32(z * (resolution + 1) + x + 1))
            }
        }
        // Vertical edges
        for x in 0...resolution {
            for z in 0..<resolution {
                indices.append(UInt32(z * (resolution + 1) + x))
                indices.append(UInt32((z + 1) * (resolution + 1) + x))
            }
        }
        guard let ib = device.makeBuffer(bytes: indices,
                                          length: indices.count * MemoryLayout<UInt32>.stride,
                                          options: .storageModeShared)
        else { return nil }
        return (ib, indices.count)
    }

    static func buildSphere(device: MTLDevice, rings: Int = 24, sectors: Int = 24) -> MeshBuffers? {
        var vertices: [BuoyVertex] = []
        var indices: [UInt32] = []

        for r in 0...rings {
            let phi = Float.pi * Float(r) / Float(rings)
            for s in 0...sectors {
                let theta = 2.0 * Float.pi * Float(s) / Float(sectors)
                let x = sin(phi) * cos(theta)
                let y = cos(phi)
                let z = sin(phi) * sin(theta)
                let n = SIMD3<Float>(x, y, z)
                vertices.append(BuoyVertex(position: n, normal: n))
            }
        }

        for r in 0..<rings {
            for s in 0..<sectors {
                let r0 = UInt32(r * (sectors + 1) + s)
                let r1 = UInt32((r + 1) * (sectors + 1) + s)
                indices.append(contentsOf: [r0, r1, r0 + 1, r0 + 1, r1, r1 + 1])
            }
        }

        guard let vb = device.makeBuffer(bytes: vertices,
                                         length: vertices.count * MemoryLayout<BuoyVertex>.stride,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(bytes: indices,
                                         length: indices.count * MemoryLayout<UInt32>.stride,
                                         options: .storageModeShared)
        else { return nil }

        return MeshBuffers(vertexBuffer: vb, indexBuffer: ib, indexCount: indices.count)
    }

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
