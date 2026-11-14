/// Couverture du scan en direct, pour montrer les zones manquantes (voile rouge).
///
/// Grille creuse de mailles de 10 cm (blocs de 16³ créés à la demande, l'appartement entier tient en ~400 Ko). Chaque
/// image de profondeur LiDAR, sous-échantillonnée, trace ses rayons depuis la caméra : les mailles traversées sont de
/// l'air vu, celle où le rayon s'arrête une surface vue, les autres restent inconnues. Une zone manquante est une face
/// entre de l'air vu et de l'inconnu : le regard s'est arrêté là sans rien rencontrer (mur pas encore scanné, trou,
/// arrière d'un meuble). Ne sont pas montrées : les faces près d'une porte ou d'une fenêtre (une porte ouverte donne
/// sur la pièce voisine) et les zones de moins de `minHoleFaces` faces (souvent sous ou derrière un meuble).
///
/// Code Swift pur (sans ARKit ni simd d'Apple) : testé sous Linux par tests/coverage_tests.swift.
final class CoverageMap {
    static let cellSize: Float = 0.10                 // m
    static let maxDepth: Float = 5.0                  // m, comme --max-depth de scan_to_domain.py
    static let minDepth: Float = 0.1                  // m, en deçà le LiDAR n'est pas fiable
    static let minConfidence: UInt8 = 1               // confiance ARKit moyenne ou haute
    static let minHoleFaces = 10                      // 0,1 m² : les zones plus petites ne sont pas montrées
    static let maxVeilFaces = 150_000                 // borne de l'affichage (≈ 1 500 m² de voile)
    static let maxRaySteps = 128                      // > maxDepth / cellSize × √3

    private static let blockBits: Int32 = 4           // blocs de 16 × 16 × 16 mailles
    private static let blockSide = 1 << 4
    private static let unknown: UInt8 = 0, free: UInt8 = 1, surface: UInt8 = 2

    /// Une image de profondeur et sa pose. Intrinsèques à la résolution de `depth` ; caméra ARKit : x à droite, y en
    /// haut, regard vers -z, ligne v de l'image vers le bas.
    struct DepthFrame {
        var depth: [Float]                            // mètres, width × height
        var confidence: [UInt8]?                      // 0, 1, 2
        var width: Int
        var height: Int
        var fx: Float, fy: Float, cx: Float, cy: Float
        var cameraToWorld: [SIMD4<Float>]             // 4 colonnes de la matrice caméra -> monde
    }

    /// Le voile : quadrilatères des faces manquantes, chacun en deux triangles dans les deux sens (visible des deux côtés).
    struct Veil {
        var positions: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        var faces: Int { positions.count / 4 }
    }

    private var blocks: [SIMD3<Int32>: [UInt8]] = [:]

    var cellCount: Int { blocks.count * Self.blockSide * Self.blockSide * Self.blockSide }

    // MARK: - sauvegarde (coverage.bin : la couverture survit à la reprise d'un scan)

    private static let magic: [UInt8] = Array("RSCV".utf8)
    private static let fileVersion: UInt32 = 1

    /// Octets : "RSCV", version (uint32), taille de maille (float32), nombre de blocs (uint32), puis par bloc trois
    /// int32 (indices du bloc) et ses 16³ mailles. Petit-boutiste.
    func serialize() -> [UInt8] {
        var out = Self.magic
        func put<T>(_ v: T) { withUnsafeBytes(of: v) { out.append(contentsOf: $0) } }
        put(Self.fileVersion.littleEndian)
        put(Self.cellSize.bitPattern.littleEndian)
        put(UInt32(blocks.count).littleEndian)
        for (b, cells) in blocks.sorted(by: { ($0.key.x, $0.key.y, $0.key.z) < ($1.key.x, $1.key.y, $1.key.z) }) {
            put(b.x.littleEndian); put(b.y.littleEndian); put(b.z.littleEndian)
            out.append(contentsOf: cells)
        }
        return out
    }

    /// Recharge une sauvegarde (en remplaçant le contenu) ; false si les octets ne sont pas une sauvegarde valide ou si
    /// la taille de maille a changé.
    @discardableResult
    func load(_ bytes: [UInt8]) -> Bool {
        let n = Self.blockSide * Self.blockSide * Self.blockSide
        guard bytes.count >= 16, Array(bytes[0..<4]) == Self.magic else { return false }
        func u32(_ o: Int) -> UInt32 { bytes[o..<o + 4].enumerated().reduce(0) { $0 | UInt32($1.element) << (8 * UInt32($1.offset)) } }
        guard u32(4) == Self.fileVersion, Float(bitPattern: u32(8)) == Self.cellSize else { return false }
        let count = Int(u32(12))
        guard bytes.count == 16 + count * (12 + n) else { return false }
        var loaded: [SIMD3<Int32>: [UInt8]] = [:]
        var o = 16
        for _ in 0..<count {
            let b = SIMD3(Int32(bitPattern: u32(o)), Int32(bitPattern: u32(o + 4)), Int32(bitPattern: u32(o + 8)))
            loaded[b] = Array(bytes[(o + 12)..<(o + 12 + n)])
            o += 12 + n
        }
        blocks = loaded
        return true
    }

    // MARK: - mailles

    static func cellOf(_ p: SIMD3<Float>) -> SIMD3<Int32> {
        SIMD3(Int32((p.x / cellSize).rounded(.down)), Int32((p.y / cellSize).rounded(.down)), Int32((p.z / cellSize).rounded(.down)))
    }

    private static func split(_ c: SIMD3<Int32>) -> (SIMD3<Int32>, Int) {
        let b = SIMD3(c.x >> blockBits, c.y >> blockBits, c.z >> blockBits)
        let m = Int32(blockSide - 1)
        let i = (Int(c.x & m) * blockSide + Int(c.y & m)) * blockSide + Int(c.z & m)
        return (b, i)
    }

    func state(_ c: SIMD3<Int32>) -> UInt8 {
        let (b, i) = Self.split(c)
        return blocks[b]?[i] ?? Self.unknown
    }

    /// L'air n'efface pas une surface (un rayon bruité ne troue pas un mur déjà vu) ; une surface remplace l'air.
    private func mark(_ c: SIMD3<Int32>, _ s: UInt8) {
        let (b, i) = Self.split(c)
        if blocks[b] == nil { blocks[b] = [UInt8](repeating: Self.unknown, count: Self.blockSide * Self.blockSide * Self.blockSide) }
        let old = blocks[b]![i]
        if s == Self.surface ? old != Self.surface : old == Self.unknown { blocks[b]![i] = s }
    }

    // MARK: - intégration

    /// Trace les rayons d'une image, un pixel sur `stride` dans chaque direction.
    func integrate(_ f: DepthFrame, stride: Int) {
        let T = f.cameraToWorld
        let origin = SIMD3(T[3].x, T[3].y, T[3].z)
        for v in Swift.stride(from: stride / 2, to: f.height, by: stride) {
            for u in Swift.stride(from: stride / 2, to: f.width, by: stride) {
                let k = v * f.width + u
                let d = f.depth[k]
                guard d.isFinite, d >= Self.minDepth, d <= Self.maxDepth else { continue }
                if let c = f.confidence, c[k] < Self.minConfidence { continue }
                let pc = SIMD3((Float(u) - f.cx) / f.fx * d, -(Float(v) - f.cy) / f.fy * d, -d)
                let r = T[0] * pc.x + T[1] * pc.y + T[2] * pc.z
                trace(from: origin, to: origin + SIMD3(r.x, r.y, r.z))
            }
        }
    }

    /// Parcours des mailles du segment (Amanatides et Woo) : air jusqu'à la maille du point touché, qui est une surface.
    private func trace(from a: SIMD3<Float>, to b: SIMD3<Float>) {
        var c = Self.cellOf(a)
        let end = Self.cellOf(b)
        let d = b - a
        var step = SIMD3<Int32>(repeating: 0)
        var tMax = SIMD3<Float>(repeating: .infinity)
        var tDelta = SIMD3<Float>(repeating: .infinity)
        for ax in 0..<3 where d[ax] != 0 {
            step[ax] = d[ax] > 0 ? 1 : -1
            let edge = Float(c[ax] + (d[ax] > 0 ? 1 : 0)) * Self.cellSize
            tMax[ax] = (edge - a[ax]) / d[ax]
            tDelta[ax] = Self.cellSize / abs(d[ax])
        }
        for _ in 0..<Self.maxRaySteps {
            if c == end { break }
            mark(c, Self.free)
            let ax = tMax.x < tMax.y ? (tMax.x < tMax.z ? 0 : 2) : (tMax.y < tMax.z ? 1 : 2)
            if tMax[ax] > 1 { break }
            c[ax] += step[ax]
            tMax[ax] += tDelta[ax]
        }
        mark(end, Self.surface)
    }

    // MARK: - voile

    private static let directions: [SIMD3<Int32>] = [SIMD3(-1, 0, 0), SIMD3(1, 0, 0), SIMD3(0, -1, 0), SIMD3(0, 1, 0), SIMD3(0, 0, -1), SIMD3(0, 0, 1)]

    /// Faces entre air vu et inconnu, sauf près des points de porte ou de fenêtre (`openingPoints`, monde, m) ; groupées par
    /// zones d'inconnu adjacentes, les zones de moins de minHoleFaces faces sont écartées.
    func veil(openingPoints: [SIMD3<Float>]) -> Veil {
        var nearOpening = Set<SIMD3<Int32>>()
        for p in openingPoints {
            let c = Self.cellOf(p)
            for dx in -1...1 { for dy in -1...1 { for dz in -1...1 { nearOpening.insert(c &+ SIMD3(Int32(dx), Int32(dy), Int32(dz))) } } }
        }

        // faces candidates (maille d'air, direction) et mailles inconnues qu'elles touchent
        var faces: [(cell: SIMD3<Int32>, dir: Int, hole: Int)] = []
        var holeIndex: [SIMD3<Int32>: Int] = [:]
        var holes: [SIMD3<Int32>] = []
        let n = Int32(Self.blockSide)
        for (b, cells) in blocks {
            let base = SIMD3(b.x * n, b.y * n, b.z * n)
            for i in 0..<cells.count where cells[i] == Self.free {
                let c = base &+ SIMD3(Int32(i / (Self.blockSide * Self.blockSide)), Int32((i / Self.blockSide) % Self.blockSide), Int32(i % Self.blockSide))
                if nearOpening.contains(c) { continue }
                for (dir, o) in Self.directions.enumerated() {
                    let nb = c &+ o
                    guard state(nb) == Self.unknown, !nearOpening.contains(nb), !touchesSurface(nb) else { continue }
                    let h: Int
                    if let known = holeIndex[nb] { h = known } else { h = holes.count; holeIndex[nb] = h; holes.append(nb) }
                    faces.append((c, dir, h))
                }
            }
        }

        // zones : union des mailles inconnues voisines (6-connexité)
        var parent = Array(0..<holes.count)
        func root(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { r = parent[r] }
            var y = x
            while parent[y] != r { let next = parent[y]; parent[y] = r; y = next }
            return r
        }
        for (i, h) in holes.enumerated() {
            for o in Self.directions {
                if let j = holeIndex[h &+ o] {
                    let a = root(i), b = root(j)
                    if a != b { parent[a] = b }
                }
            }
        }
        var zoneFaces = [Int](repeating: 0, count: holes.count)
        for f in faces { zoneFaces[root(f.hole)] += 1 }

        var veil = Veil()
        for f in faces where zoneFaces[root(f.hole)] >= Self.minHoleFaces {
            if veil.faces >= Self.maxVeilFaces { break }
            Self.appendQuad(&veil, cell: f.cell, dir: f.dir)
        }
        return veil
    }

    /// Une maille inconnue collée à une surface vue est derrière cette surface, pas un oubli : un mur n'a qu'une maille
    /// d'épaisseur de surface et forme des marches (mur en biais, points tombés de part et d'autre d'une limite de maille).
    /// Un vrai trou de mur garde en son centre des mailles qui ne touchent aucune surface.
    private func touchesSurface(_ c: SIMD3<Int32>) -> Bool {
        Self.directions.contains { state(c &+ $0) == Self.surface }
    }

    /// Quadrilatère de la face `dir` de la maille, deux fois (un sens par côté)
    private static func appendQuad(_ veil: inout Veil, cell: SIMD3<Int32>, dir: Int) {
        let ax = dir / 2, plus = dir % 2 == 1
        let u = (ax + 1) % 3, w = (ax + 2) % 3
        var p0 = SIMD3<Float>(Float(cell.x), Float(cell.y), Float(cell.z)) * cellSize
        if plus { p0[ax] += cellSize }
        var du = SIMD3<Float>(repeating: 0), dw = SIMD3<Float>(repeating: 0)
        du[u] = cellSize
        dw[w] = cellSize
        let base = UInt32(veil.positions.count)
        veil.positions += [p0, p0 + du, p0 + du + dw, p0 + dw]
        veil.indices += [base, base + 1, base + 2, base, base + 2, base + 3,
                         base, base + 2, base + 1, base, base + 3, base + 2]
    }
}
