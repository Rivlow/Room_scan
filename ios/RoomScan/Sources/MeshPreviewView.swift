import SwiftUI
import SceneKit
import ARKit

/// Maillage prêt pour SceneKit : sommets de 24 octets (x y z nx ny nz en float32) et indices de triangles par classe ARKit.
struct RawMesh {
    var vertexData: Data
    var offset: Int                     // début des sommets dans vertexData (multiple de 4)
    var vertexCount: Int
    var indices: [[Int32]]              // 8 classes (0 none … 7 door)

    var triangleCount: Int { indices.reduce(0) { $0 + $1.count } / 3 }
}

/// Lecture des .ply de l'exporteur (mesh.ply, missing.ply).
enum PLYMesh {
    struct Unreadable: LocalizedError {
        var errorDescription: String? { "fichier .ply illisible" }
    }

    static func read(_ url: URL) throws -> RawMesh {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard let end = data.range(of: Data("end_header\n".utf8), in: 0..<min(4096, data.count)) else { throw Unreadable() }
        var nv = 0, nf = 0
        for line in String(decoding: data[0..<end.lowerBound], as: UTF8.self).split(separator: "\n") {
            let w = line.split(separator: " ")
            if w.count == 3 && w[0] == "element" {
                if w[1] == "vertex" { nv = Int(w[2]) ?? 0 } else if w[1] == "face" { nf = Int(w[2]) ?? 0 }
            }
        }
        let hdr = end.upperBound, facesAt = hdr + nv * 24
        guard data.count == facesAt + nf * 14 else { throw Unreadable() }

        // sommets : directement sur le fichier projeté. La longueur de l'en-tête est quelconque et les attributs de
        // sommets doivent être alignés sur 4 octets pour le GPU : si elle n'est pas multiple de 4, copie du seul bloc.
        let (vertexData, at) = hdr % 4 == 0 ? (data, hdr) : (data.subdata(in: hdr..<facesAt), 0)

        // faces réparties par classe (octet 0..7 de l'en-tête PLY, autre valeur = none)
        var indices = [[Int32]](repeating: [], count: 8)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for i in 0..<nf {
                let o = facesAt + i * 14
                let c = Int(raw[o + 13])
                for v in 0..<3 {
                    indices[c < 8 ? c : 0].append(raw.loadUnaligned(fromByteOffset: o + 1 + 4 * v, as: Int32.self))
                }
            }
        }
        return RawMesh(vertexData: vertexData, offset: at, vertexCount: nv, indices: indices)
    }
}

/// Le maillage vivant d'une session (aperçu pendant la pause) : copie des ancres dans le repère monde
enum LiveMesh {
    static func read(_ anchors: [ARMeshAnchor]) -> RawMesh {
        let nv = anchors.reduce(0) { $0 + $1.geometry.vertices.count }
        var data = Data(count: nv * 24)
        var indices = [[Int32]](repeating: [], count: 8)
        var base = 0
        data.withUnsafeMutableBytes { (all: UnsafeMutableRawBufferPointer) in
            for a in anchors {
                autoreleasepool {
                    let n = a.geometry.vertices.count
                    MeshExporter.writeWorldVertices(a, into: UnsafeMutableRawBufferPointer(rebasing: all[(base * 24)..<((base + n) * 24)]))
                    let offset = base
                    MeshExporter.forEachFace(a) { tri, c in
                        let k = c < 8 ? Int(c) : 0
                        indices[k].append(Int32(tri[0] + offset))
                        indices[k].append(Int32(tri[1] + offset))
                        indices[k].append(Int32(tri[2] + offset))
                    }
                    base += n
                }
            }
        }
        return RawMesh(vertexData: data, offset: 0, vertexCount: nv, indices: indices)
    }
}

/// Géométries SceneKit : le maillage coloré par classe, le voile rouge translucide des zones manquantes
enum MeshGeometry {
    static let veilColor = UIColor(red: 1.0, green: 0.1, blue: 0.1, alpha: 0.45)

    private static func vertexSources(_ m: RawMesh) -> [SCNGeometrySource] {
        func source(_ semantic: SCNGeometrySource.Semantic, _ offset: Int) -> SCNGeometrySource {
            SCNGeometrySource(data: m.vertexData, semantic: semantic, vectorCount: m.vertexCount, usesFloatComponents: true,
                              componentsPerVector: 3, bytesPerComponent: 4, dataOffset: m.offset + offset, dataStride: 24)
        }
        return [source(.vertex, 0), source(.normal, 12)]
    }

    private static func element(_ indices: [Int32]) -> SCNGeometryElement {
        SCNGeometryElement(data: indices.withUnsafeBytes { Data($0) }, primitiveType: .triangles,
                           primitiveCount: indices.count / 3, bytesPerIndex: 4)
    }

    static func classes(_ m: RawMesh) -> SCNGeometry {
        var elements: [SCNGeometryElement] = []
        var materials: [SCNMaterial] = []
        for c in 0..<8 where !m.indices[c].isEmpty {
            elements.append(element(m.indices[c]))
            let col = MeshExporter.classColors[MeshExporter.className(UInt8(c))]!
            let mat = SCNMaterial()
            mat.diffuse.contents = UIColor(red: CGFloat(col.x), green: CGFloat(col.y), blue: CGFloat(col.z), alpha: 1)
            materials.append(mat)
        }
        let g = SCNGeometry(sources: vertexSources(m), elements: elements)
        g.materials = materials
        return g
    }

    /// Le voile : sans éclairage, des deux côtés, sans écrire la profondeur (on voit le maillage au travers)
    static func veil(_ m: RawMesh) -> SCNGeometry {
        let g = SCNGeometry(sources: vertexSources(m), elements: [element(m.indices.flatMap { $0 })])
        let mat = SCNMaterial()
        mat.diffuse.contents = veilColor
        mat.lightingModel = .constant
        mat.isDoubleSided = true
        mat.writesToDepthBuffer = false
        mat.blendMode = .alpha
        g.materials = [mat]
        return g
    }

    /// Le voile d'une session en cours, en RawMesh (normales nulles)
    static func veilMesh(_ v: CoverageMap.Veil) -> RawMesh {
        var data = Data(count: v.positions.count * 24)
        data.withUnsafeMutableBytes { raw in
            for (i, p) in v.positions.enumerated() {
                raw.storeBytes(of: p.x, toByteOffset: i * 24, as: Float.self)
                raw.storeBytes(of: p.y, toByteOffset: i * 24 + 4, as: Float.self)
                raw.storeBytes(of: p.z, toByteOffset: i * 24 + 8, as: Float.self)
            }
        }
        var indices = [[Int32]](repeating: [], count: 8)
        indices[0] = v.indices.map { Int32($0) }
        return RawMesh(vertexData: data, offset: 0, vertexCount: v.positions.count, indices: indices)
    }
}

/// Visualisation d'un maillage (couleurs par classe) et du voile des zones manquantes, orbite au doigt. La source : le
/// dossier d'un scan (mesh.ply, missing.ply), ou un instantané de la session en pause.
struct MeshPreviewView: View {
    let title: String
    let load: () throws -> (mesh: RawMesh, veil: RawMesh?)

    @State private var scene: SCNScene?
    @State private var veilNode: SCNNode?
    @State private var showVeil = true
    @State private var error: String?

    init(scan: ScanRecord) {
        title = scan.name
        let meshURL = scan.meshURL, veilURL = scan.missingURL
        load = {
            guard FileManager.default.fileExists(atPath: meshURL.path) else { throw MissingMesh() }
            let veil = FileManager.default.fileExists(atPath: veilURL.path) ? try? PLYMesh.read(veilURL) : nil
            return (try PLYMesh.read(meshURL), veil)
        }
    }

    init(title: String, load: @escaping () throws -> (mesh: RawMesh, veil: RawMesh?)) {
        self.title = title
        self.load = load
    }

    struct MissingMesh: LocalizedError {
        var errorDescription: String? { "Pas de mesh.ply dans ce scan" }
    }

    var body: some View {
        Group {
            if let scene = scene {
                SceneView(scene: scene, options: [.allowsCameraControl, .autoenablesDefaultLighting])
                    .ignoresSafeArea(edges: .bottom)
            } else if let error = error {
                Text(error).foregroundColor(.orange).padding()
            } else {
                ProgressView("Chargement du maillage…")
            }
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                if veilNode != nil {
                    Button {
                        showVeil.toggle()
                        veilNode?.isHidden = !showVeil
                    } label: {
                        Label("Zones manquantes", systemImage: showVeil ? "exclamationmark.triangle.fill" : "exclamationmark.triangle")
                    }
                    .tint(.red)
                }
            }
        }
        .onAppear(perform: build)
    }

    private func build() {
        guard scene == nil else { return }
        let load = self.load
        DispatchQueue.global(qos: .userInitiated).async {
            let meshes: (mesh: RawMesh, veil: RawMesh?)
            do {
                meshes = try load()
            } catch let e {
                DispatchQueue.main.async { error = e.localizedDescription }
                return
            }
            let s = SCNScene()
            let node = SCNNode(geometry: MeshGeometry.classes(meshes.mesh))
            s.rootNode.addChildNode(node)
            var vn: SCNNode?
            if let v = meshes.veil, v.triangleCount > 0 {
                let n = SCNNode(geometry: MeshGeometry.veil(v))
                n.renderingOrder = 1                    // après le maillage opaque
                s.rootNode.addChildNode(n)
                vn = n
            }
            s.background.contents = UIColor(white: 0.08, alpha: 1)
            // caméra de départ : au-dessus et devant la boîte englobante
            let (lo, hi) = node.boundingBox
            let center = SCNVector3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
            let size = max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z))
            let camNode = SCNNode()
            camNode.camera = SCNCamera()
            camNode.camera?.zFar = 200
            camNode.position = SCNVector3(center.x + size * 0.9, center.y + size * 0.8, center.z + size * 0.9)
            camNode.look(at: center)
            s.rootNode.addChildNode(camNode)
            DispatchQueue.main.async {
                veilNode = vn
                scene = s
            }
        }
    }
}
