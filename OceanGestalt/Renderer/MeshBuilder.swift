import Metal
import simd

/// Ocean vertex: base grid position (Y=0) + UV.
/// UV is stored so the shader can derive normal-map coordinates from the
/// undisplaced XZ without recomputing from position each frame.
///
/// Uses explicit Float fields rather than SIMD3/SIMD2 to guarantee a packed,
/// C-compatible layout: position 12 bytes + texCoord 8 bytes = stride 20.
/// SIMD3<Float> has a 16-byte stride in Swift which would misalign texCoord
/// against the Metal .float3 vertex format (which reads exactly 12 bytes).
struct OceanVertex {
    var px, py, pz: Float   // position
    var u, v: Float          // texCoord (= base XZ, used for UV derivation in shader)
}

struct MeshBuffers {
    let vertexBuffer: MTLBuffer
    let indexBuffer:  MTLBuffer
    let indexCount:   Int
}

struct OceanMeshBuffers {
    let vertexBuffer:  MTLBuffer
    let indexBuffer:   MTLBuffer
    let indexCount:    Int
    let vertexCount:   Int
    let gridWidth:     Int   // n+1 vertices per side of the square grid
    /// Horizontal + vertical grid lines for the wireframe pass.
    let wirelineBuffer: MTLBuffer
    let wirelineCount:  Int
}

enum MeshBuilder {

    // MARK: - Ocean grid

    /// Builds an ocean grid of (size × subdivisions)² quads, centred at origin.
    /// Mirrors Mesh::generateMesh() / generateTriangularIndices() /
    /// generateWireframeIndices() in src/core/Mesh.cpp.
    static func buildGrid(device: MTLDevice, size: Int, subdivisions: Int) -> OceanMeshBuffers? {
        let n = size * subdivisions          // vertices per side (cells)
        let step = Float(size) / Float(n)    // world-space step
        let half = Float(size) * 0.5

        // (n+1)² vertices
        var verts = [OceanVertex]()
        verts.reserveCapacity((n + 1) * (n + 1))
        for row in 0...n {
            for col in 0...n {
                let x = Float(col) * step - half
                let z = Float(row) * step - half
                verts.append(OceanVertex(px: x, py: 0, pz: z, u: x, v: z))
            }
        }

        // Triangle indices — two triangles per quad
        var tris = [UInt32]()
        tris.reserveCapacity(n * n * 6)
        for row in 0..<n {
            for col in 0..<n {
                let bl = UInt32(col + 0 + (n + 1) * (row + 0))
                let br = UInt32(col + 1 + (n + 1) * (row + 0))
                let tl = UInt32(col + 0 + (n + 1) * (row + 1))
                let tr = UInt32(col + 1 + (n + 1) * (row + 1))
                tris += [bl, tr, br,
                         tr, bl, tl]
            }
        }

        // Wireline indices — horizontal rows then vertical columns
        var lines = [UInt32]()
        lines.reserveCapacity((n + 1) * n * 4)
        for row in 0...n {
            for col in 0..<n {
                lines.append(UInt32(col + 0 + (n + 1) * row))
                lines.append(UInt32(col + 1 + (n + 1) * row))
            }
        }
        for col in 0...n {
            for row in 0..<n {
                lines.append(UInt32(col + (n + 1) * (row + 0)))
                lines.append(UInt32(col + (n + 1) * (row + 1)))
            }
        }

        guard let vb = device.makeBuffer(bytes: verts,
                                         length: verts.count * MemoryLayout<OceanVertex>.stride,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(bytes: tris,
                                         length: tris.count * MemoryLayout<UInt32>.stride,
                                         options: .storageModeShared),
              let wb = device.makeBuffer(bytes: lines,
                                         length: lines.count * MemoryLayout<UInt32>.stride,
                                         options: .storageModeShared)
        else { return nil }

        return OceanMeshBuffers(vertexBuffer: vb, indexBuffer: ib,
                                indexCount: tris.count,
                                vertexCount: verts.count,
                                gridWidth: n + 1,
                                wirelineBuffer: wb, wirelineCount: lines.count)
    }

    // MARK: - MTLVertexDescriptor for the ocean grid

    static var oceanVertexDescriptor: MTLVertexDescriptor {
        let d = MTLVertexDescriptor()
        d.attributes[0].format     = .float3
        d.attributes[0].offset     = 0
        d.attributes[0].bufferIndex = 0
        d.attributes[1].format     = .float2
        d.attributes[1].offset     = 12   // immediately after the packed float3
        d.attributes[1].bufferIndex = 0
        d.layouts[0].stride        = MemoryLayout<OceanVertex>.stride
        return d
    }
}
