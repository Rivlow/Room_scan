import ARKit
import Foundation
import simd

struct MeshStats {
    var vertices = 0
    var triangles = 0
    var floorY: Double?
    var ceilingY: Double?
    var classTriangles: [String: Int] = [:]
}

/// Fusion des ARMeshAnchor (repère monde ARKit, y vers le haut) en mesh.ply binaire (sommets, normales, faces avec la
/// classe par face), écrit en flux ancre par ancre : un seul tampon de la taille de la plus grosse ancre (~240 Ko).
enum MeshExporter {
    static let classOrder = ["wall", "floor", "ceiling", "table", "seat", "window", "door", "none"]
    static let classColors: [String: SIMD3<Float>] = [
        "wall": SIMD3(0.85, 0.82, 0.75), "floor": SIMD3(0.45, 0.35, 0.25), "ceiling": SIMD3(0.90, 0.90, 0.95),
        "table": SIMD3(0.75, 0.55, 0.30), "seat": SIMD3(0.30, 0.50, 0.80), "window": SIMD3(0.50, 0.80, 0.95),
        "door": SIMD3(0.60, 0.40, 0.60), "none": SIMD3(0.70, 0.70, 0.70),
    ]

    static func className(_ raw: UInt8) -> String {
        switch ARMeshClassification(rawValue: Int(raw)) ?? ARMeshClassification.none {
        case .wall: return "wall"
        case .floor: return "floor"
        case .ceiling: return "ceiling"
        case .table: return "table"
        case .seat: return "seat"
        case .window: return "window"
        case .door: return "door"
        case .none: return "none"
        @unknown default: return "none"
        }
    }

    private static func readFloat3(_ src: ARGeometrySource, _ i: Int) -> SIMD3<Float> {
        let p = src.buffer.contents().advanced(by: src.offset + i * src.stride).assumingMemoryBound(to: Float.self)
        return SIMD3(p[0], p[1], p[2])
    }

    /// Histogramme de hauteurs : 4 000 bacs de 1 cm sur [-20, 20] m.
    private struct HeightHistogram {
        static let bins = 4000
        var counts = [Int](repeating: 0, count: bins)
        var n = 0

        mutating func add(_ y: Float) {
            guard y.isFinite else { return }
            counts[Int(max(0, min(Float(Self.bins - 1), ((y + 20) * 100).rounded(.down))))] += 1
            n += 1
        }

        /// Centre du premier bac où le cumul atteint (n + 1) / 2.
        var median: Double? {
            guard n > 0 else { return nil }
            var cum = 0
            for (i, c) in counts.enumerated() {
                cum += c
                if 2 * cum >= n + 1 { return -20 + (Double(i) + 0.5) * 0.01 }
            }
            return nil
        }
    }

    /// Sommets d'une ancre dans le repère monde, 24 octets chacun (x y z nx ny nz en float32), écrits dans `raw`
    /// (au moins vertices.count × 24 octets)
    static func writeWorldVertices(_ a: ARMeshAnchor, into raw: UnsafeMutableRawBufferPointer) {
        let g = a.geometry
        let T = a.transform
        let R = simd_float3x3(xyz(T.columns.0), xyz(T.columns.1), xyz(T.columns.2))
        for i in 0..<g.vertices.count {
            let p = readFloat3(g.vertices, i)
            let pw = T * SIMD4(p.x, p.y, p.z, 1)
            let n = i < g.normals.count ? simd_normalize(R * readFloat3(g.normals, i)) : SIMD3<Float>(0, 1, 0)
            let o = i * 24
            raw.storeBytes(of: pw.x, toByteOffset: o, as: Float.self)
            raw.storeBytes(of: pw.y, toByteOffset: o + 4, as: Float.self)
            raw.storeBytes(of: pw.z, toByteOffset: o + 8, as: Float.self)
            raw.storeBytes(of: n.x, toByteOffset: o + 12, as: Float.self)
            raw.storeBytes(of: n.y, toByteOffset: o + 16, as: Float.self)
            raw.storeBytes(of: n.z, toByteOffset: o + 20, as: Float.self)
        }
    }

    /// Chaque face d'une ancre : indices de ses sommets (dans l'ancre) et classe ARKit (0 sans classification)
    static func forEachFace(_ a: ARMeshAnchor, _ body: (SIMD3<Int>, UInt8) -> Void) {
        let g = a.geometry
        let f = g.faces
        guard f.count > 0 else { return }
        let ip = f.buffer.contents()
        let per = f.indexCountPerPrimitive
        let wide = f.bytesPerIndex != 2
        var cls: UnsafeMutableRawPointer?
        var classStride = 0
        if let c = g.classification, c.count == f.count {
            cls = c.buffer.contents().advanced(by: c.offset)
            classStride = c.stride
        }
        for i in 0..<f.count {
            var tri = SIMD3<Int>(repeating: 0)
            for v in 0..<3 {
                let off = (i * per + v) * f.bytesPerIndex
                tri[v] = wide ? Int(ip.load(fromByteOffset: off, as: UInt32.self)) : Int(ip.load(fromByteOffset: off, as: UInt16.self))
            }
            body(tri, cls?.load(fromByteOffset: i * classStride, as: UInt8.self) ?? 0)
        }
    }

    /// Point d'une ancre dans le repère monde
    static func worldPoint(_ a: ARMeshAnchor, _ i: Int) -> SIMD3<Float> {
        let p = readFloat3(a.geometry.vertices, i)
        return xyz(a.transform * SIMD4(p.x, p.y, p.z, 1))
    }

    /// Le voile des zones manquantes (missing.ply), au format de mesh.ply : normales nulles, classe 0
    static func writeVeil(_ veil: CoverageMap.Veil, to url: URL) throws {
        let nv = veil.positions.count, nf = veil.indices.count / 3
        var data = Data("""
        ply
        format binary_little_endian 1.0
        comment RoomScan missing zones (faces between seen air and unknown cells), ARKit frame, metres
        element vertex \(nv)
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        element face \(nf)
        property list uchar int vertex_indices
        property uchar class
        end_header

        """.utf8)
        data.reserveCapacity(data.count + nv * 24 + nf * 14)
        func put<T>(_ v: T) { withUnsafeBytes(of: v) { data.append(contentsOf: $0) } }
        for p in veil.positions {
            put(p.x); put(p.y); put(p.z); put(Float(0)); put(Float(0)); put(Float(0))
        }
        for t in 0..<nf {
            put(UInt8(3))
            for v in 0..<3 { put(Int32(veil.indices[3 * t + v])) }
            put(UInt8(0))
        }
        try data.write(to: url, options: .atomic)
    }

    /// En-tête d'un mesh.ply de l'exporteur : nombre de sommets, de faces, et longueur de l'en-tête
    struct PreviousMesh {
        let url: URL
        let vertices: Int
        let faces: Int
        let headerBytes: Int

        /// nil si le fichier n'existe pas ou n'est pas au format de l'exporteur (la reprise repart alors sans lui)
        init?(url: URL) {
            guard let h = FileHandle(forReadingAtPath: url.path) else { return nil }
            defer { try? h.close() }
            guard let head = try? h.read(upToCount: 4096), let end = head.range(of: Data("end_header\n".utf8)) else { return nil }
            var nv = 0, nf = 0
            for line in String(decoding: head[0..<end.lowerBound], as: UTF8.self).split(separator: "\n") {
                let w = line.split(separator: " ")
                if w.count == 3 && w[0] == "element" {
                    if w[1] == "vertex" { nv = Int(w[2]) ?? 0 } else if w[1] == "face" { nf = Int(w[2]) ?? 0 }
                }
            }
            guard let size = try? h.seekToEnd(), size == UInt64(end.upperBound + nv * 24 + nf * 14) else { return nil }
            self.url = url
            vertices = nv
            faces = nf
            headerBytes = end.upperBound
        }
    }

    /// Copie `count` octets de `src` (à partir de sa position courante) dans `dst`, par morceaux ; `each` reçoit chaque
    /// morceau (longueurs multiples de `align`).
    private static func copy(_ src: FileHandle, to dst: FileHandle, count: Int, align: Int, each: ((Data) -> Void)? = nil) throws {
        var left = count
        let chunk = (1 << 20) / align * align
        while left > 0 {
            let n = min(chunk, left)
            guard let d = try src.read(upToCount: n), d.count == n else { throw CocoaError(.fileReadCorruptFile) }
            try dst.write(contentsOf: d)
            each?(d)
            left -= n
        }
    }

    /// mesh.ply des ancres, précédé des sommets et faces de `previous` (le mesh.ply d'une session antérieure du même
    /// scan, même repère grâce à la carte ARKit) : une zone rescannée figure alors deux fois, ce que les convertisseurs
    /// tolèrent. Sol et plafond médians : session courante seulement.
    static func export(anchors: [ARMeshAnchor], directory: URL, previous: PreviousMesh? = nil) throws -> MeshStats {
        // passe 0 : tailles totales et plus grosse ancre
        var totalV = previous?.vertices ?? 0, totalF = previous?.faces ?? 0, maxBytes = 0
        for a in anchors {
            autoreleasepool {
                let nv = a.geometry.vertices.count, nf = a.geometry.faces.count
                totalV += nv
                totalF += nf
                maxBytes = max(maxBytes, max(nv * 24, nf * 14))
            }
        }

        let fm = FileManager.default
        let url = directory.appendingPathComponent("mesh.ply")
        let part = directory.appendingPathComponent("mesh.ply.part")
        try? fm.removeItem(at: part)
        guard fm.createFile(atPath: part.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        var stats = MeshStats(vertices: totalV, triangles: totalF)
        do {
            let file = try FileHandle(forWritingTo: part)
            defer { try? file.close() }
            let old = previous.flatMap { FileHandle(forReadingAtPath: $0.url.path) }
            defer { try? old?.close() }
            if previous != nil && old == nil { throw CocoaError(.fileReadNoSuchFile) }     // l'en-tête annonce ses sommets
            let header = """
            ply
            format binary_little_endian 1.0
            comment RoomScan ARKit mesh, ARKit frame (x right, y up, z backward), metres
            comment class: 0 none 1 wall 2 floor 3 ceiling 4 table 5 seat 6 window 7 door
            element vertex \(totalV)
            property float x
            property float y
            property float z
            property float nx
            property float ny
            property float nz
            element face \(totalF)
            property list uchar int vertex_indices
            property uchar class
            end_header

            """
            try file.write(contentsOf: Data(header.utf8))
            var scratch = [UInt8](repeating: 0, count: max(maxBytes, 1))

            // passe A : sommets (24 octets : x y z nx ny nz en float32 petit-boutiste), ceux de la session
            // précédente d'abord (leurs indices de faces restent valables)
            if let p = previous, let old = old {
                try old.seek(toOffset: UInt64(p.headerBytes))
                try copy(old, to: file, count: p.vertices * 24, align: 24)
            }
            for a in anchors {
                try autoreleasepool {
                    let nv = a.geometry.vertices.count
                    guard nv > 0 else { return }
                    scratch.withUnsafeMutableBytes { raw in writeWorldVertices(a, into: raw) }
                    try file.write(contentsOf: scratch[0..<(nv * 24)])
                }
            }

            // passe B : faces (14 octets : uchar 3, 3 int32, uchar classe), compteurs et histogrammes sol / plafond
            let floorRaw = UInt8(ARMeshClassification.floor.rawValue), ceilRaw = UInt8(ARMeshClassification.ceiling.rawValue)
            var perClass = [Int](repeating: 0, count: 256)
            var floorH = HeightHistogram(), ceilH = HeightHistogram()
            var base = 0
            if let p = previous, old != nil {
                try copy(old!, to: file, count: p.faces * 14, align: 14) { d in
                    d.withUnsafeBytes { raw in
                        for i in Swift.stride(from: 13, to: raw.count, by: 14) { perClass[Int(raw[i])] += 1 }
                    }
                }
                base = p.vertices
            }
            for a in anchors {
                try autoreleasepool {
                    let nf = a.geometry.faces.count
                    let nv = a.geometry.vertices.count
                    defer { base += nv }
                    guard nf > 0 else { return }
                    let offset = base
                    scratch.withUnsafeMutableBytes { raw in
                        var i = 0
                        forEachFace(a) { tri, c in
                            perClass[Int(c)] += 1
                            if c == floorRaw || c == ceilRaw {
                                let y = (worldPoint(a, tri[0]).y + worldPoint(a, tri[1]).y + worldPoint(a, tri[2]).y) / 3
                                if c == floorRaw { floorH.add(y) } else { ceilH.add(y) }
                            }
                            let o = i * 14
                            raw.storeBytes(of: 3, toByteOffset: o, as: UInt8.self)
                            raw.storeBytes(of: Int32(tri[0] + offset), toByteOffset: o + 1, as: Int32.self)
                            raw.storeBytes(of: Int32(tri[1] + offset), toByteOffset: o + 5, as: Int32.self)
                            raw.storeBytes(of: Int32(tri[2] + offset), toByteOffset: o + 9, as: Int32.self)
                            raw.storeBytes(of: c, toByteOffset: o + 13, as: UInt8.self)
                            i += 1
                        }
                    }
                    try file.write(contentsOf: scratch[0..<(nf * 14)])
                }
            }
            for raw in 0..<256 where perClass[raw] > 0 {
                stats.classTriangles[className(UInt8(raw)), default: 0] += perClass[raw]
            }
            stats.floorY = floorH.median
            stats.ceilingY = ceilH.median
        } catch {
            try? fm.removeItem(at: part)
            throw error
        }
        try? fm.removeItem(at: url)
        try fm.moveItem(at: part, to: url)              // renommage atomique, sans copie
        return stats
    }
}
