import ARKit
import Metal
import RealityKit
import UIKit
import os

/// Modes de visualisation pendant le scan. Le maillage est le rendu de débogage d'ARKit ; les autres dessinent le
/// nuage de points LiDAR accumulé (PointCloud) par un shader Metal en post-traitement de l'ARView.
enum ViewMode: Int, CaseIterable, Identifiable {
    case camera = 0          // image caméra seule
    case mesh                // image caméra + maillage ARKit coloré par classe (défaut)
    case points              // fond noir, points colorés par hauteur, les points frais brillent
    case confidence          // fond noir, points colorés par confiance LiDAR (rouge faible, jaune, vert haute)
    case pointsCamera        // image caméra + points colorés par hauteur

    var id: Int { rawValue }
    var title: String {
        switch self {
        case .camera: return "Caméra"
        case .mesh: return "Caméra + maillage"
        case .points: return "Nuage de points"
        case .confidence: return "Confiance LiDAR"
        case .pointsCamera: return "Caméra + nuage"
        }
    }
    var icon: String {
        switch self {
        case .camera: return "camera"
        case .mesh: return "cube.transparent.fill"
        case .points: return "circle.grid.3x3.fill"
        case .confidence: return "waveform.path.ecg"
        case .pointsCamera: return "circle.grid.3x3"
        }
    }
    var showsMesh: Bool { self == .mesh }
    var drawsPoints: Bool { self == .points || self == .confidence || self == .pointsCamera }
    var blackBackground: Bool { self == .points || self == .confidence }
    /// valeur passée au shader
    var shaderMode: UInt32 { self == .confidence ? 1 : 0 }
}

/// Nuage de points LiDAR accumulé pendant le scan, directement dans un tampon Metal partagé (16 octets par point :
/// position et attributs). Un point par maille de `cell` (dédoublonnage par un ensemble de clés), ajouté sur la file
/// de couverture ; le rendu lit [0, count) sur le fil principal. Capacité fixe : ~24 Mo, un appartement entier.
final class PointCloud {
    struct Vertex {
        var x: Float, y: Float, z: Float
        var attr: UInt32            // bits 0-1 confiance (0, 1, 2), bits 2-17 temps en dixièmes de seconde
    }

    static let capacity = 1_500_000
    static let cell: Float = 0.025
    static let minDepth: Float = 0.1, maxDepth: Float = 5.0

    let buffer: MTLBuffer
    private let lock = OSAllocatedUnfairLock(initialState: 0)      // count publié
    private var seen = Set<Int64>()                                 // file de couverture seulement
    private var next = 0                                            // file de couverture seulement
    private var heights = [Int](repeating: 0, count: 4000)          // bacs de 1 cm sur [-20, 20] m, file de couverture
    private var heightCount = 0
    private let rangeLock = OSAllocatedUnfairLock(initialState: (lo: Float(0), hi: Float(2.5)))
    private(set) var full = false

    init?(device: MTLDevice) {
        guard let b = device.makeBuffer(length: Self.capacity * MemoryLayout<Vertex>.stride, options: .storageModeShared) else { return nil }
        buffer = b
        buffer.label = "RoomScan points"
        seen.reserveCapacity(200_000)
    }

    var count: Int { lock.withLock { $0 } }

    /// Hauteurs (y ARKit) entre lesquelles étaler la palette : 2e et 98e centiles des points
    var heightRange: (lo: Float, hi: Float) { rangeLock.withLock { $0 } }

    private static func key(_ p: SIMD3<Float>) -> Int64 {
        let c = SIMD3<Int64>(Int64((p.x / cell).rounded(.down)), Int64((p.y / cell).rounded(.down)), Int64((p.z / cell).rounded(.down)))
        return ((c.x & 0x1FFFFF) << 42) | ((c.y & 0x1FFFFF) << 21) | (c.z & 0x1FFFFF)
    }

    /// Ajoute les points d'une image de profondeur (un pixel sur `stride`), temps `t` en secondes depuis le début du
    /// scan. Sur la file de couverture.
    func integrate(_ f: CoverageMap.DepthFrame, stride: Int, time t: Float) {
        guard !full else { return }
        let T = f.cameraToWorld
        let origin = SIMD3(T[3].x, T[3].y, T[3].z)
        let stamp = UInt32(max(0, min(65535, (t * 10).rounded()))) << 2
        let base = buffer.contents().bindMemory(to: Vertex.self, capacity: Self.capacity)
        var added = 0
        for v in Swift.stride(from: stride / 2, to: f.height, by: stride) {
            for u in Swift.stride(from: stride / 2, to: f.width, by: stride) {
                let k = v * f.width + u
                let d = f.depth[k]
                guard d.isFinite, d >= Self.minDepth, d <= Self.maxDepth else { continue }
                let conf = f.confidence?[k] ?? 2
                if conf < 1 { continue }
                let pc = SIMD3((Float(u) - f.cx) / f.fx * d, -(Float(v) - f.cy) / f.fy * d, -d)
                let r = T[0] * pc.x + T[1] * pc.y + T[2] * pc.z
                let p = origin + SIMD3(r.x, r.y, r.z)
                guard seen.insert(Self.key(p)).inserted else { continue }
                if next >= Self.capacity { full = true; break }
                base[next] = Vertex(x: p.x, y: p.y, z: p.z, attr: stamp | UInt32(min(conf, 2)))
                next += 1
                added += 1
                if p.y.isFinite {
                    heights[Int(max(0, min(Float(heights.count - 1), ((p.y + 20) * 100).rounded(.down))))] += 1
                    heightCount += 1
                }
            }
            if full { break }
        }
        if added > 0 {
            let n = next
            lock.withLock { $0 = n }
            updateRange()
        }
    }

    private func updateRange() {
        guard heightCount > 100 else { return }
        var lo: Float = 0, hi: Float = 2.5
        var cum = 0
        var gotLo = false
        for (i, c) in heights.enumerated() where c > 0 {
            cum += c
            if !gotLo && 50 * cum >= heightCount { lo = -20 + Float(i) * 0.01; gotLo = true }
            if 50 * cum >= 49 * heightCount { hi = -20 + Float(i + 1) * 0.01; break }
        }
        if hi - lo < 0.5 { hi = lo + 0.5 }
        let range = (lo: lo, hi: hi)
        rangeLock.withLock { $0 = range }
    }
}

/// Dessin du nuage en post-traitement de l'ARView : copie du rendu RealityKit, puis les points par-dessus (taille en
/// pixels décroissante avec la distance, test de profondeur entre points). Shaders compilés à l'exécution depuis la
/// source : rien à ajouter au projet. Si la compilation échoue, le rendu RealityKit passe tel quel.
final class PointCloudRenderer {
    struct Uniforms {
        var viewProj: simd_float4x4
        var now: Float
        var heightLo: Float
        var heightHi: Float
        var pointScale: Float
        var mode: UInt32
        var pad: SIMD3<UInt32> = .zero
    }

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex { packed_float3 p; uint attr; };
    struct Uniforms { float4x4 viewProj; float now; float lo; float hi; float pointScale; uint mode; uint3 pad; };
    struct VOut { float4 pos [[position]]; float size [[point_size]]; float3 color; };

    static float3 heightColor(float t) {
        // bleu profond -> bleu -> cyan -> vert -> jaune (t dans [0, 1])
        float3 c0 = float3(0.05, 0.10, 0.45), c1 = float3(0.10, 0.45, 1.00), c2 = float3(0.10, 0.95, 0.95);
        float3 c3 = float3(0.30, 0.95, 0.30), c4 = float3(1.00, 0.95, 0.30);
        float s = clamp(t, 0.0, 1.0) * 4.0;
        if (s < 1.0) return mix(c0, c1, s);
        if (s < 2.0) return mix(c1, c2, s - 1.0);
        if (s < 3.0) return mix(c2, c3, s - 2.0);
        return mix(c3, c4, s - 3.0);
    }

    vertex VOut pointVertex(uint vid [[vertex_id]], const device Vertex* pts [[buffer(0)]], constant Uniforms& u [[buffer(1)]]) {
        VOut o;
        float3 p = float3(pts[vid].p);
        uint attr = pts[vid].attr;
        float4 clip = u.viewProj * float4(p, 1.0);
        o.pos = clip;
        float w = max(clip.w, 0.05);
        o.size = clamp(u.pointScale / w, 1.5, 10.0);
        float age = u.now - float((attr >> 2) & 0xFFFFu) * 0.1;
        float fresh = exp(-max(age, 0.0) / 1.5);            // les points de la dernière seconde brillent
        float3 c;
        if (u.mode == 1u) {
            uint conf = attr & 3u;
            c = conf >= 2u ? float3(0.20, 0.85, 0.30) : (conf == 1u ? float3(0.95, 0.80, 0.20) : float3(0.90, 0.20, 0.20));
        } else {
            c = heightColor((p.y - u.lo) / max(u.hi - u.lo, 0.1));
        }
        o.color = mix(c, float3(1.0), fresh * 0.8);
        return o;
    }

    fragment float4 pointFragment(VOut in [[stage_in]], float2 pc [[point_coord]]) {
        float2 d = pc - 0.5;
        if (dot(d, d) > 0.25) discard_fragment();
        return float4(in.color, 1.0);
    }
    """

    private var device: MTLDevice?
    private var pipelines: [MTLPixelFormat: MTLRenderPipelineState] = [:]
    private var library: MTLLibrary?
    private var depthLess: MTLDepthStencilState?
    private var depthGreater: MTLDepthStencilState?
    private var depthTexture: MTLTexture?
    private(set) var failure: String?

    func prepare(device: MTLDevice) {
        self.device = device
        do {
            library = try device.makeLibrary(source: Self.source, options: nil)
        } catch {
            failure = "shaders du nuage : \(error.localizedDescription)"
            return
        }
        let less = MTLDepthStencilDescriptor()
        less.depthCompareFunction = .less
        less.isDepthWriteEnabled = true
        depthLess = device.makeDepthStencilState(descriptor: less)
        let greater = MTLDepthStencilDescriptor()
        greater.depthCompareFunction = .greater
        greater.isDepthWriteEnabled = true
        depthGreater = device.makeDepthStencilState(descriptor: greater)
    }

    private func pipeline(for format: MTLPixelFormat) -> MTLRenderPipelineState? {
        if let p = pipelines[format] { return p }
        guard let device = device, let library = library else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "RoomScan points"
        d.vertexFunction = library.makeFunction(name: "pointVertex")
        d.fragmentFunction = library.makeFunction(name: "pointFragment")
        d.colorAttachments[0].pixelFormat = format
        d.depthAttachmentPixelFormat = .depth32Float
        do {
            let p = try device.makeRenderPipelineState(descriptor: d)
            pipelines[format] = p
            return p
        } catch {
            failure = "pipeline du nuage : \(error.localizedDescription)"
            return nil
        }
    }

    private func depth(for target: MTLTexture) -> MTLTexture? {
        if let t = depthTexture, t.width == target.width, t.height == target.height { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: target.width, height: target.height, mipmapped: false)
        d.usage = .renderTarget
        d.storageMode = .private
        depthTexture = device?.makeTexture(descriptor: d)
        return depthTexture
    }

    /// Copie le rendu RealityKit dans la cible puis, si `mode` dessine des points, les ajoute avec les matrices de la
    /// caméra ARKit pour l'orientation de l'interface.
    func render(_ context: ARView.PostProcessContext, cloud: PointCloud?, camera: ARCamera?, orientation: UIInterfaceOrientation,
                mode: ViewMode, now: Float) {
        guard let blit = context.commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: context.sourceColorTexture, to: context.targetColorTexture)
        blit.endEncoding()
        guard mode.drawsPoints, failure == nil, let cloud = cloud, let camera = camera else { return }
        let count = cloud.count
        guard count > 0, let pipeline = pipeline(for: context.targetColorTexture.pixelFormat),
              let depthTex = depth(for: context.targetColorTexture) else { return }
        let target = context.targetColorTexture
        let size = CGSize(width: target.width, height: target.height)
        let proj = camera.projectionMatrix(for: orientation, viewportSize: size, zNear: 0.05, zFar: 60)
        let view = camera.viewMatrix(for: orientation)
        // convention de profondeur de la projection : proche -> 0 (test « less ») ou proche -> 1 (« greater »)
        let near = proj * SIMD4<Float>(0, 0, -1, 1), far = proj * SIMD4<Float>(0, 0, -10, 1)
        let standard = near.z / near.w < far.z / far.w
        let range = cloud.heightRange
        var u = Uniforms(viewProj: proj * view, now: now, heightLo: range.lo, heightHi: range.hi,
                         pointScale: Float(target.height) * 0.0045, mode: mode.shaderMode)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depthTex
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = standard ? 1.0 : 0.0
        pass.depthAttachment.storeAction = .dontCare
        guard let enc = context.commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "RoomScan points"
        enc.setRenderPipelineState(pipeline)
        if let ds = standard ? depthLess : depthGreater { enc.setDepthStencilState(ds) }
        enc.setVertexBuffer(cloud.buffer, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: count)
        enc.endEncoding()
    }
}
