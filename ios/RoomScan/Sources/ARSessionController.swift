import ARKit
import RealityKit
import Combine
import UIKit
import os

@inline(__always) func xyz(_ v: SIMD4<Float>) -> SIMD3<Float> { SIMD3(v.x, v.y, v.z) }

/// Session ARKit : suivi, reconstruction de scène (maillage), capture des images clés, export final.
/// Les rappels du délégué arrivent sur la file principale (delegateQueue nil) : les @Published sont modifiés là.
final class ARSessionController: NSObject, ObservableObject, ARSessionDelegate {
    enum Phase: Equatable { case idle, scanning, paused, finishing, done, failed }

    @Published var phase: Phase = .idle
    @Published var trackingMessage = "Démarrage…"
    @Published var trackingGood = false
    @Published var keyframeCount = 0
    @Published var triangleCount = 0
    @Published var ceilingShare = 0.0          // part des triangles classés plafond
    @Published var elapsed: TimeInterval = 0
    @Published var freeGigabytes = 0.0
    @Published var errorMessage: String?
    @Published var captureNotice: String?      // arrêt de la capture ou cadence réduite
    @Published var result: ScanMeta?
    @Published private(set) var scanURL: URL?
    @Published private(set) var veilVisible = true
    @Published private(set) var missingArea = 0.0         // m² de voile (zones manquantes) au dernier calcul
    @Published private(set) var viewMode: ViewMode = .mesh  // visualisation : maillage ARKit ou nuage de points (shaders)
    @Published private(set) var pointCount = 0            // points du nuage accumulé
    @Published private(set) var cloudNotice: String?      // nuage indisponible (shaders refusés) ou plein
    @Published private(set) var resuming = false          // session de reprise d'un scan existant (carte ARKit rechargée)
    @Published private(set) var relocalized = false       // reprise : ARKit a retrouvé le repère du scan précédent
    @Published private(set) var mappingStatus = ""        // état de la carte ARKit (vide tant qu'inconnu)

    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)

    private weak var settings: ScanSettings?
    private weak var store: ScanStore?
    private var writer: KeyframeWriter?
    private var previousMeta: ScanMeta?                   // reprise : scan.json des sessions précédentes
    private var lastMappingStatus: ARFrame.WorldMappingStatus?
    private var stopped = false                           // écran quitté pendant la lecture de la carte
    private var startDate: Date?
    private var pausedTotal: TimeInterval = 0
    private var pauseDate: Date?
    private var ticker: Timer?
    private var lastKeyTimestamp: TimeInterval = -1
    private var lastKeyTransform: simd_float4x4?
    private var meshAnchors: [UUID: ARMeshAnchor] = [:]
    private var anchorStats: [UUID: (tri: Int, ceil: Int)] = [:]
    private var totalTri = 0
    private var totalCeil = 0
    private var lastMeshStat: Date = .distantPast
    private var smoothedDepth = false
    private var depthSize = (0, 0)
    private var lastTrackingKey = ""

    private enum CaptureStop { case disk, memory }
    private var captureStop: CaptureStop?
    private var thermalNotice: String?
    private var thermalFactor = 1.0
    private var meshHiddenByThermal = false
    private var skipKeyframesUntil: Date = .distantPast
    private var memoryWarnings: [Date] = []
    private var peakFootprint = 0
    private var observers: [NSObjectProtocol] = []

    // nuage de points : rempli sur la file de couverture, dessiné sur le fil de rendu de RealityKit (post-traitement)
    private struct RenderState { var mode: ViewMode = .mesh; var orientation: UIInterfaceOrientation = .portrait; var firstTimestamp: TimeInterval = -1 }
    private let renderState = OSAllocatedUnfairLock(initialState: RenderState())
    private var cloud: PointCloud?
    private let cloudRenderer = PointCloudRenderer()
    private static let cloudStride = 4                            // un pixel de profondeur sur 4 (3 000 points par image)
    private var cloudFailureLogged = false

    // zones manquantes : grille de couverture nourrie par la profondeur (file série), voile recalculé chaque seconde
    private static let coverageInterval: TimeInterval = 0.2       // s entre deux images intégrées (× facteur thermique)
    private static let coverageStride = 6                         // un pixel de profondeur sur 6 dans chaque direction
    private static let veilInterval: TimeInterval = 1.0           // s entre deux calculs du voile
    private let coverage = CoverageMap()
    private let coverageQueue = DispatchQueue(label: "roomscan.coverage", qos: .utility)
    private var coverageBusy = false
    private var lastCoverage: TimeInterval = -1
    private var veilBusy = false
    private var lastVeil: Date = .distantPast
    private var latestVeil = CoverageMap.Veil()
    private var openingPoints: [UUID: [SIMD3<Float>]] = [:]       // centres des triangles porte / fenêtre, par ancre
    private let veilAnchor = AnchorEntity(world: matrix_identity_float4x4)
    private let veilEntity = ModelEntity()

    static var isSupported: Bool {
        ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification)
            && ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    // MARK: - commandes

    /// Nouveau scan, ou reprise de `resuming` : sa carte ARKit (worldmap.arworldmap) est rechargée, le suivi démarre en
    /// « relocalisation » jusqu'à ce que la caméra reconnaisse un endroit déjà scanné, puis les nouvelles images clés
    /// tombent dans le même repère que les anciennes. Le maillage ARKit repart de zéro (la carte ne le contient pas) :
    /// l'ancien mesh.ply est fusionné au nouveau à l'export ; la grille des zones manquantes est rechargée.
    func start(settings: ScanSettings, store: ScanStore, resuming: ScanRecord? = nil) {
        guard phase == .idle else { return }
        self.settings = settings
        self.store = store
        guard Self.isSupported else {
            errorMessage = "Cet appareil n'a pas de LiDAR : le scan de pièce n'est pas disponible."
            phase = .failed
            return
        }
        do {
            if let r = resuming {
                try ScanStore.ensureKeyframeDirectories(in: r.url)
                scanURL = r.url
                previousMeta = r.meta
                self.resuming = true
            } else {
                scanURL = try store.newScanDirectory()
            }
        } catch {
            errorMessage = "Impossible de créer le dossier du scan : \(error.localizedDescription)"
            phase = .failed
            return
        }
        writer = KeyframeWriter(root: scanURL!.appendingPathComponent("keyframes", isDirectory: true),
                                jpegQuality: settings.jpegQuality)
        smoothedDepth = settings.smoothedDepth
        keyframeCount = resuming?.meta?.keyframes ?? 0

        arView.session.delegate = self
        arView.environment.sceneUnderstanding.options = []
        viewMode = settings.showMesh ? .mesh : .camera
        renderState.withLock { $0.mode = viewMode }
        arView.debugOptions = viewMode.showsMesh ? [.showSceneUnderstanding] : []
        if let device = MTLCreateSystemDefaultDevice() {
            cloud = PointCloud(device: device)
            if cloud == nil { cloudNotice = "Nuage de points indisponible : mémoire insuffisante" }
        }
        arView.renderCallbacks.prepareWithDevice = { [cloudRenderer] device in cloudRenderer.prepare(device: device) }
        arView.renderCallbacks.postProcess = { [weak self] context in self?.postProcess(context) }
        veilEntity.isEnabled = false
        veilAnchor.addChild(veilEntity)
        arView.scene.addAnchor(veilAnchor)

        guard let r = resuming else {
            run(worldMap: nil)
            return
        }
        // la carte (dizaines de Mo) et la couverture se lisent hors du fil principal
        trackingMessage = "Lecture de la carte du scan…"
        let mapURL = r.worldMapURL, coverageURL = r.coverageURL, coverage = self.coverage, coverageQueue = self.coverageQueue
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let loaded: Result<ARWorldMap, Error> = Result {
                let data = try Data(contentsOf: mapURL)
                guard let map = try NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: data) else {
                    throw NSError(domain: "RoomScan", code: 2, userInfo: [NSLocalizedDescriptionKey: "worldmap.arworldmap illisible"])
                }
                return map
            }
            var coverageLoaded = false
            if let bytes = try? Data(contentsOf: coverageURL) {
                coverageLoaded = coverageQueue.sync { coverage.load([UInt8](bytes)) }
            }
            DispatchQueue.main.async {
                guard let self = self, self.phase == .idle, !self.stopped else { return }
                switch loaded {
                case .success(let map):
                    self.writer?.log("reprise : carte ARKit \(map.anchors.count) ancres, couverture \(coverageLoaded ? "rechargee" : "absente")")
                    self.run(worldMap: map)
                case .failure(let e):
                    self.errorMessage = "Impossible de reprendre ce scan : \(e.localizedDescription)"
                    self.phase = .failed
                }
            }
        }
    }

    private func run(worldMap: ARWorldMap?) {
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = .meshWithClassification
        config.planeDetection = []
        config.environmentTexturing = .none
        config.frameSemantics.insert(.sceneDepth)
        if smoothedDepth && ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        config.initialWorldMap = worldMap
        if worldMap != nil {
            trackingMessage = "Relocalisation : visez un endroit déjà scanné"
            lastTrackingKey = ""
        }
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors, .resetSceneReconstruction])

        startDate = Date()
        pausedTotal = 0
        phase = .scanning
        ticker = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in self?.tick() }
        updateFreeSpace()
        let center = NotificationCenter.default
        observers = [
            center.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
                self?.applyThermalState()
            },
            center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
                self?.handleMemoryWarning()
            },
        ]
        applyThermalState()
    }

    /// Change la visualisation : maillage ARKit (rendu de débogage), fond noir ou image caméra, points dessinés en
    /// post-traitement. Le nuage s'accumule quel que soit le mode : on peut changer à tout moment.
    func setViewMode(_ mode: ViewMode) {
        viewMode = mode
        renderState.withLock { $0.mode = mode }
        arView.environment.background = mode.blackBackground ? .color(.black) : .cameraFeed()
        applyMeshVisibility()
    }

    private func applyMeshVisibility() {
        let show = viewMode.showsMesh && !meshHiddenByThermal && phase == .scanning
        arView.debugOptions = show ? [.showSceneUnderstanding] : []
    }

    /// Fil de rendu de RealityKit : copie du rendu puis nuage de points selon le mode courant
    private func postProcess(_ context: ARView.PostProcessContext) {
        let st = renderState.withLock { $0 }
        let frame = arView.session.currentFrame
        let now = Float((frame?.timestamp ?? 0) - max(st.firstTimestamp, 0))
        cloudRenderer.render(context, cloud: cloud, camera: frame?.camera, orientation: st.orientation, mode: st.mode, now: now)
    }

    func setVeilVisible(_ visible: Bool) {
        veilVisible = visible
        veilEntity.isEnabled = visible && latestVeil.faces > 0
        if visible { lastVeil = .distantPast }        // recalcul au prochain tic
    }

    /// Pause : la session ARKit continue (le repère est conservé, on peut marcher et reprendre exactement là) ; seuls
    /// l'enregistrement et la couverture s'arrêtent, et le maillage en surimpression est masqué pour moins chauffer.
    /// L'écran reste allumé (ScanView) : sans cela la mise en veille couperait la session.
    func pause() {
        guard phase == .scanning else { return }
        pauseDate = Date()
        phase = .paused
        applyMeshVisibility()
    }

    func resume() {
        guard phase == .paused else { return }
        if let p = pauseDate { pausedTotal += Date().timeIntervalSince(p) }
        pauseDate = nil
        phase = .scanning
        applyMeshVisibility()
    }

    /// Le scan tel qu'il est maintenant (aperçu pendant la pause), pris sur le fil principal : les ancres (leurs tampons
    /// restent valides, la conversion peut se faire ailleurs) et le dernier voile calculé
    func snapshot() -> (anchors: [ARMeshAnchor], veil: CoverageMap.Veil) {
        (Array(meshAnchors.values), latestVeil)
    }

    /// Terminer : la carte ARKit est demandée (reprise ultérieure du scan) tant que la session tourne, puis export.
    func finish() {
        guard phase == .scanning || phase == .paused else { return }
        if let p = pauseDate { pausedTotal += Date().timeIntervalSince(p); pauseDate = nil }
        phase = .finishing
        ticker?.invalidate()
        ticker = nil
        guard let url = scanURL, let writer = writer else { phase = .failed; return }
        arView.debugOptions = []
        // le rappel peut arriver sur n'importe quelle file ; sans suivi correct la carte est refusée (le scan reste
        // exploitable, mais ne pourra pas être repris)
        arView.session.getCurrentWorldMap { [weak self] map, error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                if let e = error { writer.log("carte ARKit non obtenue : \(e.localizedDescription)") }
                self.export(to: url, writer: writer, worldMap: map)
            }
        }
    }

    /// Arrêt immédiat de la session (les ARMeshAnchor retenus gardent leurs MTLBuffer valides : la mémoire et le GPU
    /// d'ARKit se libèrent avant l'export), puis carte ARKit, couverture, maillage (fusionné à celui des sessions
    /// précédentes en cas de reprise), attente de la fin d'écriture des images clés, scan.json.
    private func export(to url: URL, writer: KeyframeWriter, worldMap: ARWorldMap?) {
        arView.session.pause()
        let footprintBefore = Self.footprintMB()
        let peakBefore = max(peakFootprint, footprintBefore)
        writer.log("finish: footprint=\(footprintBefore)MB avant export")
        let anchors = Array(meshAnchors.values)
        let duration = elapsedNow()
        let prevMeta = previousMeta
        let previous = prevMeta != nil ? MeshExporter.PreviousMesh(url: url.appendingPathComponent("mesh.ply")) : nil
        if prevMeta != nil && previous == nil { writer.log("reprise : mesh.ply precedent absent ou illisible, non fusionne") }
        let meta0 = ScanMeta(name: url.lastPathComponent,
                            device: UIDevice.current.model + " " + Self.modelIdentifier(),
                            systemVersion: UIDevice.current.systemVersion,
                            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?",
                            started: startDate ?? Date(), ended: nil, durationSeconds: duration,
                            keyframes: 0, vertices: 0, triangles: 0, floorHeight: nil, ceilingHeight: nil,
                            classTriangles: [:], depthWidth: depthSize.0, depthHeight: depthSize.1,
                            imageWidth: KeyframeWriter.outputWidth, imageHeight: KeyframeWriter.outputHeight,
                            smoothedDepth: smoothedDepth,
                            coordinateFrame: "arkit: x right, y up, z backward; camera looks along -z; units m")
        let points = Array(openingPoints.values.joined())
        let coverage = self.coverage, coverageQueue = self.coverageQueue
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            // carte ARKit : worldmap.arworldmap (NSKeyedArchiver, dizaines de Mo pour un appartement)
            var mapBytes = prevMeta?.worldMapBytes
            if let map = worldMap {
                do {
                    let data = try NSKeyedArchiver.archivedData(withRootObject: map, requiringSecureCoding: true)
                    try data.write(to: url.appendingPathComponent("worldmap.arworldmap"), options: .atomic)
                    mapBytes = data.count
                    writer.log("carte ARKit : \(data.count >> 20) Mo, \(map.anchors.count) ancres")
                } catch {
                    writer.log("carte ARKit non ecrite : \(error.localizedDescription)")
                }
            }
            // voile final (après les intégrations en cours, sur leur file) : missing.ply, montré dans l'aperçu ;
            // la grille elle-même : coverage.bin, rechargée à la reprise
            let (veil, coverageBytes) = coverageQueue.sync { (coverage.veil(openingPoints: points), coverage.serialize()) }
            do {
                try MeshExporter.writeVeil(veil, to: url.appendingPathComponent("missing.ply"))
                try Data(coverageBytes).write(to: url.appendingPathComponent("coverage.bin"), options: .atomic)
            } catch {
                writer.log("missing.ply / coverage.bin : \(error.localizedDescription)")
            }
            var withMesh = meta0
            do {
                let stats = try MeshExporter.export(anchors: anchors, directory: url, previous: previous)
                withMesh.vertices = stats.vertices
                withMesh.triangles = stats.triangles
                withMesh.floorHeight = stats.floorY ?? prevMeta?.floorHeight
                withMesh.ceilingHeight = stats.ceilingY ?? prevMeta?.ceilingHeight
                withMesh.classTriangles = stats.classTriangles
            } catch {
                let text = "Export du maillage : \(error.localizedDescription)"
                DispatchQueue.main.async { self?.errorMessage = text }
            }
            let afterMesh = withMesh
            let footprintAfter = Self.footprintMB()
            writer.log("finish: footprint=\(footprintAfter)MB apres export (delta \(footprintAfter - footprintBefore)MB)")
            writer.finish { snapshot in
                var final = afterMesh
                final.keyframes = (prevMeta?.keyframes ?? 0) + snapshot.written
                final.keyframesSkipped = (prevMeta?.keyframesSkipped ?? 0) + snapshot.skippedFull
                final.peakFootprintMB = max(peakBefore, footprintAfter, prevMeta?.peakFootprintMB ?? 0)
                final.started = prevMeta?.started ?? meta0.started
                final.durationSeconds = (prevMeta?.durationSeconds ?? 0) + duration
                final.sessions = (prevMeta?.sessions ?? 1) + (prevMeta != nil ? 1 : 0)
                final.worldMapBytes = mapBytes
                final.ended = Date()
                try? ScanStore.writeMeta(final, to: url)
                let done = final
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.result = done
                    self.phase = .done
                    self.store?.refresh()
                }
            }
        }
    }

    /// Quitter l'écran : arrêt de la session (un scan non terminé est supprimé par cancel).
    func stop() {
        stopped = true
        ticker?.invalidate()
        ticker = nil
        for o in observers { NotificationCenter.default.removeObserver(o) }
        observers = []
        arView.session.pause()
    }

    /// Abandon : un nouveau scan est supprimé ; une reprise ne garde que les sessions précédentes (les images clés de
    /// cette session sont effacées, mesh.ply, scan.json et la carte restent tels quels).
    func cancel() {
        stop()
        if resuming {
            writer?.discardSession { [weak self] in DispatchQueue.main.async { self?.store?.refresh() } }
        } else if let u = scanURL {
            try? FileManager.default.removeItem(at: u)
            scanURL = nil
            store?.refresh()
        }
        phase = .idle
    }

    // MARK: - état

    private func elapsedNow() -> TimeInterval {
        guard let s = startDate else { return 0 }
        var e = Date().timeIntervalSince(s) - pausedTotal
        if let p = pauseDate { e -= Date().timeIntervalSince(p) }
        return max(e, 0)
    }

    private func tick() {
        elapsed = elapsedNow()
        let orientation = arView.window?.windowScene?.interfaceOrientation ?? .portrait
        renderState.withLock { $0.orientation = orientation }
        if let c = cloud {
            let n = c.count
            if n != pointCount { pointCount = n }
            if c.full && cloudNotice == nil { cloudNotice = "Nuage de points plein (\(PointCloud.capacity) points) : il n'est plus complété" }
        }
        if let f = cloudRenderer.failure, !cloudFailureLogged {
            cloudFailureLogged = true
            cloudNotice = "Nuage de points indisponible : \(f)"
            writer?.log("nuage : \(f)")
        }
        if Date().timeIntervalSince(lastVeil) >= Self.veilInterval { updateVeil() }
        if Date().timeIntervalSince(lastMeshStat) > 2 {
            lastMeshStat = Date()
            publishMeshStats()
            updateFreeSpace()
            logStatus()
        }
    }

    // MARK: - zones manquantes

    /// Copie de la profondeur LiDAR d'une image et de sa pose (intrinsèques ramenées à la résolution de la profondeur)
    private static func depthFrame(_ frame: ARFrame) -> CoverageMap.DepthFrame? {
        guard let d = frame.sceneDepth else { return nil }
        let dm = d.depthMap
        guard CVPixelBufferGetPixelFormatType(dm) == kCVPixelFormatType_DepthFloat32 else { return nil }
        let w = CVPixelBufferGetWidth(dm), h = CVPixelBufferGetHeight(dm)
        var depth = [Float](repeating: 0, count: w * h)
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(dm) {
            let bpr = CVPixelBufferGetBytesPerRow(dm)
            depth.withUnsafeMutableBytes { dst in
                for y in 0..<h { memcpy(dst.baseAddress! + y * w * 4, base + y * bpr, w * 4) }
            }
        }
        CVPixelBufferUnlockBaseAddress(dm, .readOnly)
        var conf: [UInt8]? = nil
        if let cm = d.confidenceMap, CVPixelBufferGetWidth(cm) == w, CVPixelBufferGetHeight(cm) == h {
            var c = [UInt8](repeating: 0, count: w * h)
            CVPixelBufferLockBaseAddress(cm, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(cm) {
                let bpr = CVPixelBufferGetBytesPerRow(cm)
                c.withUnsafeMutableBytes { dst in
                    for y in 0..<h { memcpy(dst.baseAddress! + y * w, base + y * bpr, w) }
                }
            }
            CVPixelBufferUnlockBaseAddress(cm, .readOnly)
            conf = c
        }
        let K = frame.camera.intrinsics
        let sx = Float(w) / Float(frame.camera.imageResolution.width), sy = Float(h) / Float(frame.camera.imageResolution.height)
        let T = frame.camera.transform
        return CoverageMap.DepthFrame(depth: depth, confidence: conf, width: w, height: h,
                                      fx: K.columns.0.x * sx, fy: K.columns.1.y * sy, cx: K.columns.2.x * sx, cy: K.columns.2.y * sy,
                                      cameraToWorld: [T.columns.0, T.columns.1, T.columns.2, T.columns.3])
    }

    /// Une image de profondeur dans la grille de couverture, sur sa file ; une seule à la fois (les suivantes attendent
    /// la fin de celle-ci plutôt que de s'accumuler)
    private func integrateCoverage(_ frame: ARFrame) {
        let t = frame.timestamp
        guard !coverageBusy, lastCoverage < 0 || t - lastCoverage >= Self.coverageInterval * thermalFactor,
              let f = Self.depthFrame(frame) else { return }
        lastCoverage = t
        coverageBusy = true
        let coverage = self.coverage, cloud = self.cloud
        let first = renderState.withLock { $0.firstTimestamp }
        coverageQueue.async { [weak self] in
            coverage.integrate(f, stride: Self.coverageStride)
            cloud?.integrate(f, stride: Self.cloudStride, time: Float(t - first))
            DispatchQueue.main.async { self?.coverageBusy = false }
        }
    }

    /// Recalcule le voile hors du fil principal, puis remplace le maillage RealityKit (fil principal)
    private func updateVeil() {
        guard veilVisible, phase == .scanning || phase == .paused, !veilBusy else { return }
        lastVeil = Date()
        veilBusy = true
        let points = Array(openingPoints.values.joined())
        let coverage = self.coverage
        coverageQueue.async { [weak self] in
            let veil = coverage.veil(openingPoints: points)
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.veilBusy = false
                self.showVeil(veil)
            }
        }
    }

    private func showVeil(_ veil: CoverageMap.Veil) {
        latestVeil = veil
        missingArea = Double(veil.faces) * Double(CoverageMap.cellSize * CoverageMap.cellSize)
        guard veilVisible, veil.faces > 0 else {
            veilEntity.isEnabled = false
            return
        }
        var desc = MeshDescriptor(name: "zones manquantes")
        desc.positions = MeshBuffers.Positions(veil.positions)
        desc.primitives = .triangles(veil.indices)
        guard let mesh = try? MeshResource.generate(from: [desc]) else { return }
        var material = UnlitMaterial(color: MeshGeometry.veilColor.withAlphaComponent(1))
        material.blending = .transparent(opacity: .init(floatLiteral: Float(MeshGeometry.veilColor.cgColor.alpha)))
        veilEntity.model = ModelComponent(mesh: mesh, materials: [material])
        veilEntity.isEnabled = true
    }

    private func updateFreeSpace() {
        if let v = try? URL(fileURLWithPath: NSHomeDirectory()).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let cap = v.volumeAvailableCapacityForImportantUsage {
            freeGigabytes = Double(cap) / 1e9
            if freeGigabytes < 0.5 && captureStop == nil && phase == .scanning {
                captureStop = .disk
                writer?.log("capture arretee : disque (\(freeGigabytes) Go libres)")
                updateCaptureNotice()
            }
        }
    }

    private func publishMeshStats() {
        if triangleCount != totalTri { triangleCount = totalTri }
        let share = totalTri > 0 ? Double(totalCeil) / Double(totalTri) : 0
        if ceilingShare != share { ceilingShare = share }
    }

    /// Une ligne toutes les 2 s dans capture_log.txt (et la console) pour relire une marche test après coup.
    private func logStatus() {
        guard let writer = writer else { return }
        let s = writer.snapshot()
        let footprint = Self.footprintMB()
        peakFootprint = max(peakFootprint, footprint)
        let line = String(format: "t=%.1f kf=%d depth=%d max=%d skipFull=%d fail=%d thermal=%@ footprint=%dMB free=%.1fGB tri=%d map=%@",
                          elapsedNow(), keyframeCount, s.inFlight, s.maxInFlight, s.skippedFull, s.failures,
                          Self.thermalName(ProcessInfo.processInfo.thermalState), footprint, freeGigabytes, totalTri,
                          lastMappingStatus.map(Self.mappingName) ?? "?")
        writer.log(line)
        print(line)
    }

    /// État de la carte ARKit : la reprise d'un scan n'est fiable qu'à partir de « extending »
    static func mappingName(_ s: ARFrame.WorldMappingStatus) -> String {
        switch s {
        case .notAvailable: return "indisponible"
        case .limited: return "limitée"
        case .extending: return "en extension"
        case .mapped: return "complète"
        @unknown default: return "?"
        }
    }

    private static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "?"
        }
    }

    /// Empreinte mémoire physique du processus (la grandeur que surveille Jetsam), en Mo.
    static func footprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint >> 20) : 0
    }

    private func updateCaptureNotice() {
        let notice: String?
        switch captureStop {
        case .disk: notice = "Espace disque insuffisant : capture des images arrêtée (le maillage continue)"
        case .memory: notice = "Mémoire insuffisante : capture des images arrêtée (le maillage continue)"
        case nil: notice = thermalNotice
        }
        if captureNotice != notice { captureNotice = notice }
    }

    private func applyThermalState() {
        let state = ProcessInfo.processInfo.thermalState
        switch state {
        case .serious:
            thermalFactor = 2
            thermalNotice = "Téléphone chaud : cadence des images clés réduite"
        case .critical:
            thermalFactor = 4
            thermalNotice = "Téléphone chaud : cadence des images clés réduite, maillage masqué"
            meshHiddenByThermal = true        // l'aperçu de millions de triangles est le premier consommateur GPU
            applyMeshVisibility()
        default:
            thermalFactor = 1
            thermalNotice = nil
            if meshHiddenByThermal {
                meshHiddenByThermal = false
                applyMeshVisibility()
            }
        }
        writer?.log("thermal=\(Self.thermalName(state)) facteur=\(thermalFactor)")
        updateCaptureNotice()
    }

    /// Les tampons d'ARKit (maillage, classification) dominent l'empreinte et ne sont pas à nous : on ne peut que
    /// réduire ce qui nous appartient (caches Core Image, images clés en cours) et, si l'alerte se répète, s'arrêter.
    private func handleMemoryWarning() {
        let now = Date()
        skipKeyframesUntil = now.addingTimeInterval(5)
        writer?.releaseCaches()
        memoryWarnings = memoryWarnings.filter { now.timeIntervalSince($0) < 30 } + [now]
        writer?.log("alerte memoire : footprint=\(Self.footprintMB())MB, \(memoryWarnings.count) en 30 s")
        if memoryWarnings.count >= 2 && captureStop == nil && (phase == .scanning || phase == .paused) {
            captureStop = .memory
            writer?.log("capture arretee : memoire")
            updateCaptureNotice()
        }
    }

    private func updateTracking(_ state: ARCamera.TrackingState) {
        switch state {
        case .normal:
            trackingMessage = "Suivi correct"
            trackingGood = true
        case .notAvailable:
            trackingMessage = "Suivi indisponible"
            trackingGood = false
        case .limited(let reason):
            trackingGood = false
            switch reason {
            case .initializing: trackingMessage = "Initialisation : bougez lentement le téléphone"
            case .excessiveMotion: trackingMessage = "Trop rapide : ralentissez"
            case .insufficientFeatures: trackingMessage = "Trop peu de détails ou trop sombre"
            case .relocalizing: trackingMessage = "Relocalisation…"
            @unknown default: trackingMessage = "Suivi limité"
            }
        }
    }

    static func describe(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "Normal"
        case .notAvailable: return "NotAvailable"
        case .limited(let r):
            switch r {
            case .initializing: return "Limited.Initializing"
            case .excessiveMotion: return "Limited.ExcessiveMotion"
            case .insufficientFeatures: return "Limited.InsufficientFeatures"
            case .relocalizing: return "Limited.Relocalizing"
            @unknown default: return "Limited"
            }
        }
    }

    private static func modelIdentifier() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { ptr in
            String(decoding: ptr.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

    // MARK: - ARSessionDelegate

    /// Statistiques par deltas : seule l'ancre modifiée est relue, jamais tout le maillage. Les centres de ses triangles
    /// porte et fenêtre sont gardés pour le voile (pas de zone manquante signalée près d'une ouverture).
    private func record(_ m: ARMeshAnchor) {
        meshAnchors[m.identifier] = m
        let tri = m.geometry.faces.count
        var ceil = 0
        var openings: [SIMD3<Float>] = []
        let ceilRaw = UInt8(ARMeshClassification.ceiling.rawValue)
        let doorRaw = UInt8(ARMeshClassification.door.rawValue), windowRaw = UInt8(ARMeshClassification.window.rawValue)
        MeshExporter.forEachFace(m) { t, c in
            if c == ceilRaw { ceil += 1 }
            if c == doorRaw || c == windowRaw {
                openings.append((MeshExporter.worldPoint(m, t[0]) + MeshExporter.worldPoint(m, t[1]) + MeshExporter.worldPoint(m, t[2])) / 3)
            }
        }
        openingPoints[m.identifier] = openings.isEmpty ? nil : openings
        let old = anchorStats[m.identifier] ?? (0, 0)
        totalTri += tri - old.tri
        totalCeil += ceil - old.ceil
        anchorStats[m.identifier] = (tri, ceil)
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for a in anchors { if let m = a as? ARMeshAnchor { record(m) } }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for a in anchors { if let m = a as? ARMeshAnchor { record(m) } }
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for a in anchors where a is ARMeshAnchor {
            meshAnchors.removeValue(forKey: a.identifier)
            openingPoints.removeValue(forKey: a.identifier)
            if let old = anchorStats.removeValue(forKey: a.identifier) {
                totalTri -= old.tri
                totalCeil -= old.ceil
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        renderState.withLock { if $0.firstTimestamp < 0 { $0.firstTimestamp = frame.timestamp } }
        let key = Self.describe(frame.camera.trackingState)
        if key != lastTrackingKey {
            lastTrackingKey = key
            updateTracking(frame.camera.trackingState)
            if resuming && !relocalized, case .normal = frame.camera.trackingState {
                relocalized = true                      // le repère du scan précédent est retrouvé
                writer?.log(String(format: "relocalise t=%.1f", elapsedNow()))
            }
        }
        if frame.worldMappingStatus != lastMappingStatus {
            lastMappingStatus = frame.worldMappingStatus
            mappingStatus = Self.mappingName(frame.worldMappingStatus)
        }
        // la couverture suit ce qui est enregistré : en scan seulement, suivi normal (comme les images clés)
        if phase == .scanning, case .normal = frame.camera.trackingState { integrateCoverage(frame) }
        guard phase == .scanning, let settings = settings, settings.captureKeyframes, let writer = writer else { return }
        guard captureStop == nil, Date() >= skipKeyframesUntil else { return }
        guard case .normal = frame.camera.trackingState else { return }
        let t = frame.timestamp
        if lastKeyTimestamp >= 0 && t - lastKeyTimestamp < settings.keyframeInterval * thermalFactor { return }
        let T = frame.camera.transform
        if let L = lastKeyTransform {
            let move = simd_distance(xyz(T.columns.3), xyz(L.columns.3))
            let fa = -xyz(T.columns.2), fb = -xyz(L.columns.2)          // la caméra regarde vers -z
            let ang = acos(max(-1, min(1, simd_dot(fa, fb)))) * 180 / .pi
            if move < Float(settings.minMove) && ang < Float(settings.minRotate) { return }
        }
        let depth: ARDepthData?
        if smoothedDepth { depth = frame.smoothedSceneDepth ?? frame.sceneDepth } else { depth = frame.sceneDepth }
        guard let d = depth else { return }
        switch writer.capture(frame: frame, depth: d) {
        case .queued:
            depthSize = (CVPixelBufferGetWidth(d.depthMap), CVPixelBufferGetHeight(d.depthMap))
            lastKeyTimestamp = t
            lastKeyTransform = T
            keyframeCount += 1
        case .queueFull, .renderFailed:
            break
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        errorMessage = "Session AR : \(error.localizedDescription)"
        phase = .failed
    }

    func sessionWasInterrupted(_ session: ARSession) {
        lastTrackingKey = ""
        trackingMessage = "Session interrompue"
        trackingGood = false
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        lastTrackingKey = ""
        trackingMessage = "Reprise du suivi…"
    }
}
