import ARKit
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import os
import simd

/// Copie autonome de ce qu'il faut d'une ARFrame (jamais l'ARFrame elle-même : ARKit a peu de tampons).
struct KeyframeData {
    let timestamp: Int64                    // microsecondes de frame.timestamp (comme l'export Polycam)
    let image: CVPixelBuffer                // 1024x768 BGRA tiré du pool du writer, rempli par `render` sur le GPU
    let render: CIRenderTask?               // à attendre avant de lire `image`
    let depth: [Float]                      // mètres, depthWidth x depthHeight
    let depthWidth: Int
    let depthHeight: Int
    let confidence: [UInt8]?                // 0, 1, 2
    let transform: simd_float4x4            // caméra -> monde
    let intrinsics: simd_float3x3           // pour imageWidth x imageHeight
    let imageWidth: Int
    let imageHeight: Int
    let trackingState: String
    let exposureDuration: Double

    static func make(frame: ARFrame, depth: ARDepthData, image: CVPixelBuffer, render: CIRenderTask?) -> KeyframeData? {
        let dm = depth.depthMap
        guard CVPixelBufferGetPixelFormatType(dm) == kCVPixelFormatType_DepthFloat32 else { return nil }
        let w = CVPixelBufferGetWidth(dm), h = CVPixelBufferGetHeight(dm)
        var d = [Float](repeating: 0, count: w * h)
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(dm) {
            let bpr = CVPixelBufferGetBytesPerRow(dm)
            d.withUnsafeMutableBytes { dst in
                for y in 0..<h { memcpy(dst.baseAddress! + y * w * 4, base + y * bpr, w * 4) }
            }
        }
        CVPixelBufferUnlockBaseAddress(dm, .readOnly)
        var conf: [UInt8]? = nil
        if let cm = depth.confidenceMap, CVPixelBufferGetWidth(cm) == w, CVPixelBufferGetHeight(cm) == h {
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
        return KeyframeData(timestamp: Int64(frame.timestamp * 1e6), image: image, render: render, depth: d, depthWidth: w, depthHeight: h,
                            confidence: conf, transform: frame.camera.transform, intrinsics: frame.camera.intrinsics,
                            imageWidth: Int(frame.camera.imageResolution.width), imageHeight: Int(frame.camera.imageResolution.height),
                            trackingState: ARSessionController.describe(frame.camera.trackingState),
                            exposureDuration: frame.camera.exposureDuration)
    }
}

/// Écriture des images clés sur une file série, au format de l'export « keyframes » :
///   images/<ts>.jpg       1024x768, orientation du capteur (paysage), JPEG
///   depth/<ts>.png        16 bits gris, millimètres
///   confidence/<ts>.png   8 bits (0, 1, 2)
///   cameras/<ts>.json     t_00..t_23 (lignes de la matrice caméra -> monde), fx fy cx cy pour 1024x768, width, height…
///
/// Mémoire bornée : au plus `slots` images clés en vol (tampon 1024x768 recyclé + profondeur, ~3,4 Mo chacune).
/// La réduction se fait sur le GPU dès la capture ; la pleine résolution n'est jamais copiée.
final class KeyframeWriter {
    static let outputWidth = 1024
    static let outputHeight = 768
    static let slots = 4

    enum CaptureResult { case queued, queueFull, renderFailed }

    struct Snapshot {
        var written = 0
        var failures = 0
        var skippedFull = 0
        var inFlight = 0
        var maxInFlight = 0
    }

    private let queue = DispatchQueue(label: "roomscan.keyframes", qos: .utility)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let root: URL
    private let jpegQuality: Double
    private let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
    private let slotsFree = DispatchSemaphore(value: KeyframeWriter.slots)
    private var pool: CVPixelBufferPool?
    private let lock: UnsafeMutablePointer<os_unfair_lock>
    private var counters = Snapshot()               // sous `lock`
    private var logHandle: FileHandle?              // file d'écriture seulement
    private var sessionNames: [String] = []         // images clés écrites par cette session (file d'écriture seulement)

    init(root: URL, jpegQuality: Double) {
        self.root = root
        self.jpegQuality = jpegQuality
        lock = UnsafeMutablePointer<os_unfair_lock>.allocate(capacity: 1)
        lock.initialize(to: os_unfair_lock())
        let poolAttrs = [kCVPixelBufferPoolMinimumBufferCountKey: Self.slots] as CFDictionary
        let bufferAttrs = [kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                           kCVPixelBufferWidthKey: Self.outputWidth,
                           kCVPixelBufferHeightKey: Self.outputHeight,
                           kCVPixelBufferMetalCompatibilityKey: true,
                           kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as [CFString: Any] as CFDictionary
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs, bufferAttrs, &pool)
    }

    deinit {
        lock.deinitialize(count: 1)
        lock.deallocate()
    }

    private func locked<T>(_ body: (inout Snapshot) -> T) -> T {
        os_unfair_lock_lock(lock)
        defer { os_unfair_lock_unlock(lock) }
        return body(&counters)
    }

    func snapshot() -> Snapshot { locked { $0 } }

    /// Sur le fil principal : ne bloque jamais le délégué ARKit (file pleine = image clé ignorée, comptée).
    func capture(frame: ARFrame, depth: ARDepthData) -> CaptureResult {
        guard slotsFree.wait(timeout: .now()) == .success else {
            locked { $0.skippedFull += 1 }
            return .queueFull
        }
        guard let (image, task) = startDownscale(frame.capturedImage),
              let k = KeyframeData.make(frame: frame, depth: depth, image: image, render: task) else {
            locked { $0.failures += 1 }
            slotsFree.signal()
            return .renderFailed
        }
        locked {
            $0.inFlight += 1
            $0.maxInFlight = max($0.maxInFlight, $0.inFlight)
        }
        queue.async {
            self.write(k)
            self.locked { $0.inFlight -= 1 }
            self.slotsFree.signal()
        }
        return .queued
    }

    /// Tire un tampon du pool et lance la réduction 1920x1440 -> 1024x768 (encodage GPU seulement, ~1 ms).
    private func startDownscale(_ src: CVPixelBuffer) -> (CVPixelBuffer, CIRenderTask)? {
        guard let pool = pool else { return nil }
        var out: CVPixelBuffer?
        let aux = [kCVPixelBufferPoolAllocationThresholdKey: Self.slots + 1] as CFDictionary
        guard CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pool, aux, &out) == kCVReturnSuccess,
              let dst = out else { return nil }
        let ci = CIImage(cvPixelBuffer: src)
        let sx = CGFloat(Self.outputWidth) / ci.extent.width, sy = CGFloat(Self.outputHeight) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        let destination = CIRenderDestination(pixelBuffer: dst)
        destination.colorSpace = srgb
        destination.isFlipped = true        // première ligne du tampon = haut de l'image, comme CIImage(cvPixelBuffer:) à la relecture
        guard let task = try? ciContext.startTask(toRender: scaled, to: destination) else { return nil }
        return (dst, task)
    }

    /// Ajoute une ligne à <scan>/capture_log.txt (non tamponné : le journal survit à un arrêt par Jetsam).
    func log(_ line: String) {
        queue.async { self.appendLog(line) }
    }

    private func appendLog(_ line: String) {
        if logHandle == nil {
            let url = root.deletingLastPathComponent().appendingPathComponent("capture_log.txt")
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            logHandle = try? FileHandle(forWritingTo: url)
            _ = try? logHandle?.seekToEnd()
        }
        try? logHandle?.write(contentsOf: Data((line + "\n").utf8))
    }

    func releaseCaches() {
        ciContext.clearCaches()
    }

    /// Appelé une fois toutes les écritures en attente terminées (sur la file d'écriture).
    func finish(_ completion: @escaping (Snapshot) -> Void) {
        queue.async {
            try? self.logHandle?.close()
            self.logHandle = nil
            completion(self.snapshot())
        }
    }

    /// Abandon d'une reprise : efface les images clés écrites par cette session (celles des sessions précédentes
    /// restent), après les écritures en attente ; `completion` sur la file d'écriture.
    func discardSession(_ completion: @escaping () -> Void) {
        queue.async {
            self.appendLog("reprise abandonnee : \(self.sessionNames.count) images cles effacees")
            for name in self.sessionNames {
                for sub in ["images/\(name).jpg", "depth/\(name).png", "confidence/\(name).png", "cameras/\(name).json"] {
                    try? FileManager.default.removeItem(at: self.root.appendingPathComponent(sub))
                }
            }
            self.sessionNames = []
            try? self.logHandle?.close()
            self.logHandle = nil
            completion()
        }
    }

    private func write(_ k: KeyframeData) {
        let name = String(k.timestamp)
        var ok = true
        do {
            _ = try k.render?.waitUntilCompleted()
            let cs = srgb
            let scaled = CIImage(cvPixelBuffer: k.image, options: [.colorSpace: cs])
            let opt = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): jpegQuality]
            if let jpeg = ciContext.jpegRepresentation(of: scaled, colorSpace: cs, options: opt) {
                try jpeg.write(to: root.appendingPathComponent("images/\(name).jpg"))
            } else { ok = false }
            var mm = [UInt16](repeating: 0, count: k.depth.count)
            for i in 0..<k.depth.count {
                let d = k.depth[i]
                mm[i] = d.isFinite && d > 0 ? UInt16(min(d * 1000, 65535)) : 0
            }
            ok = Self.writeGrayPNG16(mm, width: k.depthWidth, height: k.depthHeight, to: root.appendingPathComponent("depth/\(name).png")) && ok
            if let c = k.confidence {
                ok = Self.writeGrayPNG8(c, width: k.depthWidth, height: k.depthHeight, to: root.appendingPathComponent("confidence/\(name).png")) && ok
            }
            try Self.cameraJSON(k, blur: blurScore(scaled)).write(to: root.appendingPathComponent("cameras/\(name).json"))
        } catch {
            ok = false
        }
        let succeeded = ok
        locked { if succeeded { $0.written += 1 } else { $0.failures += 1 } }
        sessionNames.append(name)                   // même en échec partiel : les fichiers écrits seront effacés
    }

    private static func cameraJSON(_ k: KeyframeData, blur: Double) throws -> Data {
        var d: [String: Any] = [:]
        for r in 0..<3 {
            for c in 0..<4 { d["t_\(r)\(c)"] = Double(k.transform[c][r]) }       // simd : [colonne][ligne]
        }
        let s = Double(outputWidth) / Double(k.imageWidth)
        d["fx"] = Double(k.intrinsics[0][0]) * s
        d["fy"] = Double(k.intrinsics[1][1]) * s
        d["cx"] = Double(k.intrinsics[2][0]) * s
        d["cy"] = Double(k.intrinsics[2][1]) * s
        d["width"] = outputWidth
        d["height"] = outputHeight
        d["source_width"] = k.imageWidth
        d["source_height"] = k.imageHeight
        d["depth_width"] = k.depthWidth
        d["depth_height"] = k.depthHeight
        d["timestamp"] = k.timestamp
        d["tracking_state"] = k.trackingState
        d["exposure_time"] = k.exposureDuration
        d["blur_score"] = blur
        let ci = (k.depthHeight / 2) * k.depthWidth + k.depthWidth / 2
        d["center_depth"] = Double(k.depth[ci])
        d["manual_keyframe"] = false
        return try JSONSerialization.data(withJSONObject: d, options: [.prettyPrinted, .sortedKeys])
    }

    /// Variance du laplacien sur une réduction 128x96 en niveaux de gris (grand = net).
    private func blurScore(_ image: CIImage) -> Double {
        let w = 128, h = 96
        let s = image.transformed(by: CGAffineTransform(scaleX: CGFloat(w) / image.extent.width, y: CGFloat(h) / image.extent.height))
        var buf = [UInt8](repeating: 0, count: w * h)
        buf.withUnsafeMutableBytes { ptr in
            ciContext.render(s, toBitmap: ptr.baseAddress!, rowBytes: w, bounds: CGRect(x: 0, y: 0, width: w, height: h),
                             format: .L8, colorSpace: CGColorSpaceCreateDeviceGray())
        }
        var sum = 0.0, sum2 = 0.0, n = 0.0
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let c = Double(buf[y * w + x])
                let lap = Double(buf[y * w + x - 1]) + Double(buf[y * w + x + 1]) + Double(buf[(y - 1) * w + x]) + Double(buf[(y + 1) * w + x]) - 4 * c
                sum += lap
                sum2 += lap * lap
                n += 1
            }
        }
        let mean = sum / n
        return sum2 / n - mean * mean
    }

    private static func writeGrayPNG16(_ px: [UInt16], width: Int, height: Int, to url: URL) -> Bool {
        px.withUnsafeBytes { raw -> Bool in
            guard let cf = CFDataCreate(nil, raw.baseAddress!.assumingMemoryBound(to: UInt8.self), raw.count),
                  let provider = CGDataProvider(data: cf),
                  let img = CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: width * 2,
                                    space: CGColorSpaceCreateDeviceGray(),
                                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue),
                                    provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
            else { return false }
            return writePNG(img, to: url)
        }
    }

    private static func writeGrayPNG8(_ px: [UInt8], width: Int, height: Int, to url: URL) -> Bool {
        px.withUnsafeBytes { raw -> Bool in
            guard let cf = CFDataCreate(nil, raw.baseAddress!.assumingMemoryBound(to: UInt8.self), raw.count),
                  let provider = CGDataProvider(data: cf),
                  let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                                    space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                    provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
            else { return false }
            return writePNG(img, to: url)
        }
    }

    private static func writePNG(_ img: CGImage, to url: URL) -> Bool {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return false }
        CGImageDestinationAddImage(dest, img, nil)
        return CGImageDestinationFinalize(dest)
    }
}
