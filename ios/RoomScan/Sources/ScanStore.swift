import Foundation
import Combine

/// Métadonnées d'un scan (scan.json à la racine du dossier du scan).
struct ScanMeta: Codable {
    var name: String
    var device: String
    var systemVersion: String
    var appVersion: String
    var started: Date
    var ended: Date?
    var durationSeconds: Double
    var keyframes: Int
    var vertices: Int
    var triangles: Int
    var floorHeight: Double?          // y ARKit (m) médian des faces classées sol
    var ceilingHeight: Double?        // y ARKit (m) médian des faces classées plafond
    var classTriangles: [String: Int]
    var depthWidth: Int
    var depthHeight: Int
    var imageWidth: Int
    var imageHeight: Int
    var smoothedDepth: Bool
    var coordinateFrame: String       // "arkit: x right, y up, z backward; camera looks along -z"
    var keyframesSkipped: Int? = nil  // tentatives ignorées car la file d'écriture était pleine
    var peakFootprintMB: Int? = nil   // empreinte mémoire maximale relevée pendant le scan
    var sessions: Int? = nil          // sessions de capture (1, puis +1 à chaque reprise grâce à la carte ARKit)
    var worldMapBytes: Int? = nil     // taille de worldmap.arworldmap, nil si la carte n'a pas pu être sauvegardée
}

/// Un dossier de scan sur disque.
struct ScanRecord: Identifiable, Hashable {
    let url: URL
    let meta: ScanMeta?
    let bytes: Int64?                 // nil tant que la taille n'est pas calculée

    var id: String { url.lastPathComponent }
    var name: String { url.lastPathComponent }
    var meshURL: URL { url.appendingPathComponent("mesh.ply") }
    var missingURL: URL { url.appendingPathComponent("missing.ply") }     // voile des zones manquantes en fin de scan
    var keyframesURL: URL { url.appendingPathComponent("keyframes", isDirectory: true) }
    var worldMapURL: URL { url.appendingPathComponent("worldmap.arworldmap") }         // carte ARKit (reprise du scan)
    var coverageURL: URL { url.appendingPathComponent("coverage.bin") }                // grille des zones manquantes
    var hasMesh: Bool { FileManager.default.fileExists(atPath: meshURL.path) }
    /// Reprise possible : scan terminé (scan.json) et carte ARKit sauvegardée
    var canResume: Bool { meta != nil && FileManager.default.fileExists(atPath: worldMapURL.path) }

    static func == (a: ScanRecord, b: ScanRecord) -> Bool { a.id == b.id && a.bytes == b.bytes }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

/// Dossier Documents/Scans (visible dans l'app Fichiers grâce à UIFileSharingEnabled).
final class ScanStore: ObservableObject {
    @Published private(set) var scans: [ScanRecord] = []
    let root: URL
    private let sizeQueue = DispatchQueue(label: "roomscan.sizes", qos: .utility)
    private var sizeCache: [String: Int64] = [:]        // scans terminés seulement ; fil principal
    private var sizesInFlight: Set<String> = []         // fil principal

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        root = docs.appendingPathComponent("Scans", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        Self.purgeStaleZips()
        refresh()
    }

    /// Archives laissées par un partage interrompu (l'app tuée avant la fin de la feuille de partage).
    private static func purgeStaleZips() {
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(at: fm.temporaryDirectory, includingPropertiesForKeys: nil)) ?? []
        for f in items where f.pathExtension.lowercased() == "zip" { try? fm.removeItem(at: f) }
    }

    /// Nouveau dossier <AAAA-MM-JJ_HHMMSS>/ avec keyframes/{images,depth,confidence,cameras}
    func newScanDirectory() throws -> URL {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd_HHmmss"
        let stem = f.string(from: Date())
        var url = root.appendingPathComponent(stem, isDirectory: true)
        var n = 1
        while FileManager.default.fileExists(atPath: url.path) {
            n += 1
            url = root.appendingPathComponent("\(stem)_\(n)", isDirectory: true)
        }
        try Self.ensureKeyframeDirectories(in: url)
        return url
    }

    /// keyframes/{images,depth,confidence,cameras} (déjà là pour un scan repris : sans effet)
    static func ensureKeyframeDirectories(in url: URL) throws {
        for sub in ["keyframes/images", "keyframes/depth", "keyframes/confidence", "keyframes/cameras"] {
            try FileManager.default.createDirectory(at: url.appendingPathComponent(sub, isDirectory: true),
                                                    withIntermediateDirectories: true)
        }
    }

    func refresh() {
        let fm = FileManager.default
        let dirs = (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])) ?? []
        var list: [ScanRecord] = []
        for d in dirs {
            guard (try? d.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
            var meta: ScanMeta? = nil
            if let data = try? Data(contentsOf: d.appendingPathComponent("scan.json")) {
                let dec = JSONDecoder()
                dec.dateDecodingStrategy = .iso8601
                meta = try? dec.decode(ScanMeta.self, from: data)
            }
            list.append(ScanRecord(url: d, meta: meta, bytes: meta != nil ? sizeCache[d.lastPathComponent] : nil))
        }
        scans = list.sorted { $0.name > $1.name }
        computeMissingSizes()
    }

    /// Parcourir des milliers de fichiers prend du temps : jamais sur le fil principal.
    private func computeMissingSizes() {
        let missing = scans.filter { $0.bytes == nil && !sizesInFlight.contains($0.id) }
        guard !missing.isEmpty else { return }
        for s in missing { sizesInFlight.insert(s.id) }
        sizeQueue.async { [weak self] in
            for s in missing {
                let n = ScanStore.directorySize(s.url)
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.sizesInFlight.remove(s.id)
                    if s.meta != nil { self.sizeCache[s.id] = n }
                    if let i = self.scans.firstIndex(where: { $0.id == s.id }) {
                        self.scans[i] = ScanRecord(url: s.url, meta: self.scans[i].meta, bytes: n)
                    }
                }
            }
        }
    }

    func delete(_ scan: ScanRecord) {
        try? FileManager.default.removeItem(at: scan.url)
        sizeCache[scan.id] = nil
        refresh()
    }

    static func directorySize(_ url: URL) -> Int64 {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey]
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys, options: [], errorHandler: nil) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in e {
            let v = try? f.resourceValues(forKeys: Set(keys))
            total += Int64(v?.totalFileAllocatedSize ?? v?.fileAllocatedSize ?? 0)
        }
        return total
    }

    static func writeMeta(_ meta: ScanMeta, to dir: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        enc.dateEncodingStrategy = .iso8601
        try enc.encode(meta).write(to: dir.appendingPathComponent("scan.json"), options: .atomic)
    }

    /// Archive zip du dossier (NSFileCoordinator .forUploading), dans le dossier temporaire ; bloquant.
    func makeZip(of scan: ScanRecord) throws -> URL {
        let fm = FileManager.default
        let need = scan.bytes ?? Self.directorySize(scan.url)
        if let v = try? fm.temporaryDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let free = v.volumeAvailableCapacityForImportantUsage, free < need {
            let text = String(format: "Espace disque insuffisant pour compresser le scan (%.1f Go nécessaires)", Double(need) / 1e9)
            throw NSError(domain: "RoomScan", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
        }
        let dest = fm.temporaryDirectory.appendingPathComponent(scan.name + ".zip")
        try? fm.removeItem(at: dest)
        var coordError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: scan.url, options: .forUploading, error: &coordError) { zipURL in
            do {
                try fm.moveItem(at: zipURL, to: dest)           // renommage : pas de seconde copie du zip
            } catch {
                do { try fm.copyItem(at: zipURL, to: dest) } catch { copyError = error }
            }
        }
        if let e = coordError { throw e }
        if let e = copyError { throw e }
        return dest
    }

    /// Supprime l'archive temporaire une fois la feuille de partage fermée.
    func discardZip(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }
}
