// Tests de CoverageMap sous Linux (WSL) : bash tests/run_tests.sh
// Pièce synthétique 4 × 2,5 × 3 m (x, y vers le haut, z) vue par une caméra virtuelle au centre qui tourne sur elle-même.

#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

var failures = 0
func check(_ ok: Bool, _ what: String, line: Int = #line) {
    if !ok {
        print("ÉCHEC ligne \(line) : \(what)")
        failures += 1
    }
}

let roomLo = SIMD3<Float>(-2.0, 0.0, -1.5), roomHi = SIMD3<Float>(2.0, 2.5, 1.5)
let eye = SIMD3<Float>(0.03, 1.4, 0.02)           // pas exactement sur une limite de maille
let width = 128, height = 96
let fx: Float = 100, fy: Float = 100, cx: Float = 64, cy: Float = 48

/// Caméra au point `eye`, tournée de `yaw` autour de la verticale puis de `pitch` (radians) ; colonnes caméra -> monde.
func pose(yaw: Float, pitch: Float) -> [SIMD4<Float>] {
    let cyw = cos(yaw), syw = sin(yaw), cp = cos(pitch), sp = sin(pitch)
    // R = Ry(yaw) · Rx(pitch)
    let x = SIMD3<Float>(cyw, 0, -syw)
    let y = SIMD3<Float>(syw * sp, cp, cyw * sp)
    let z = SIMD3<Float>(syw * cp, -sp, cyw * cp)
    return [SIMD4(x.x, x.y, x.z, 0), SIMD4(y.x, y.y, y.z, 0), SIMD4(z.x, z.y, z.z, 0), SIMD4(eye.x, eye.y, eye.z, 1)]
}

/// Image de profondeur de la pièce vue de cette pose (intersection du rayon avec l'intérieur de la boîte)
func render(_ T: [SIMD4<Float>]) -> CoverageMap.DepthFrame {
    var depth = [Float](repeating: 0, count: width * height)
    for v in 0..<height {
        for u in 0..<width {
            let dc = SIMD3((Float(u) - cx) / fx, -(Float(v) - cy) / fy, -1)       // profondeur 1
            let r = T[0] * dc.x + T[1] * dc.y + T[2] * dc.z
            let dir = SIMD3(r.x, r.y, r.z)
            var s = Float.infinity
            for ax in 0..<3 where dir[ax] != 0 {
                let bound = dir[ax] > 0 ? roomHi[ax] : roomLo[ax]
                s = min(s, (bound - eye[ax]) / dir[ax])
            }
            depth[v * width + u] = s
        }
    }
    return CoverageMap.DepthFrame(depth: depth, confidence: nil, width: width, height: height, fx: fx, fy: fy, cx: cx, cy: cy, cameraToWorld: T)
}

/// Tour complet sauf les directions de lacet dans `skip` (degrés). Champ vertical ±25,6° : six inclinaisons couvrent du
/// sol au plafond sans bande oubliée.
func scan(skip: ClosedRange<Float>? = nil) -> CoverageMap {
    let map = CoverageMap()
    for yawDeg in stride(from: Float(0), to: 360, by: 15) {
        if let s = skip, s.contains(yawDeg) { continue }
        for pitchDeg: Float in [-75, -45, -15, 15, 45, 75] {
            map.integrate(render(pose(yaw: yawDeg * .pi / 180, pitch: pitchDeg * .pi / 180)), stride: 2)
        }
    }
    return map
}

func centroid(_ v: CoverageMap.Veil) -> SIMD3<Float> {
    v.positions.reduce(SIMD3<Float>(repeating: 0), +) / Float(max(v.positions.count, 1))
}

// 1. tout vu : pas de voile
do {
    let veil = scan().veil(openingPoints: [])
    check(veil.faces == 0, "pièce entièrement vue : \(veil.faces) faces de voile, centre \(centroid(veil)), premier \(veil.positions.first.map { "\($0)" } ?? "-")")
}

// 2. le mur z = +1,5 jamais regardé (lacets autour de 0°, la caméra regarde vers -z à lacet 0 : +z est à 180°)
let partial = scan(skip: 120...240)
let veil = partial.veil(openingPoints: [])
check(veil.faces >= CoverageMap.minHoleFaces, "mur manqué : \(veil.faces) faces")
let c = centroid(veil)
check(c.z > 0.3, "le voile est du côté du mur manqué (z = \(c.z))")
check(veil.indices.count == veil.faces * 12 && veil.positions.count == veil.faces * 4, "deux triangles par côté et par face")
check(veil.positions.allSatisfy { $0.z > -0.5 }, "rien sur le mur opposé, déjà vu")

// 3. une porte sur tout ce côté : plus de voile près d'elle
var door: [SIMD3<Float>] = []
for x in stride(from: Float(-2.0), through: 2.0, by: 0.05) {
    for y in stride(from: Float(0), through: 2.5, by: 0.05) {
        for z in stride(from: Float(-1.0), through: 1.5, by: 0.05) { door.append(SIMD3(x, y, z)) }
    }
}
check(partial.veil(openingPoints: door).faces == 0, "voile retiré près d'une porte ou fenêtre")

// 4. un petit oubli (< 0,1 m²) n'est pas montré : un seul rayon manquant de plus, au milieu d'une pièce vue, ne laisse
// que des trous isolés d'une maille
do {
    let map = scan()
    var f = render(pose(yaw: 0, pitch: 0))
    f.depth = f.depth.map { _ in 0 }
    map.integrate(f, stride: 1)
    check(map.veil(openingPoints: []).faces == 0, "image vide sans effet")
}

// 5. mailles : négatifs arrondis vers le bas
check(CoverageMap.cellOf(SIMD3(-0.01, 0.0, 0.19)) == SIMD3<Int32>(-1, 0, 1), "maille d'un point négatif")

// 6. sauvegarde et rechargement (reprise d'un scan) : même voile, même nombre de mailles ; octets abîmés refusés
do {
    let bytes = partial.serialize()
    let again = CoverageMap()
    check(again.load(bytes), "rechargement de la sauvegarde")
    check(again.cellCount == partial.cellCount, "mailles après rechargement : \(again.cellCount) au lieu de \(partial.cellCount)")
    let v2 = again.veil(openingPoints: [])
    let d = centroid(v2) - c                    // l'ordre des blocs (donc de la somme) peut changer : tolérance
    check(v2.faces == veil.faces, "même nombre de faces après rechargement (\(v2.faces) au lieu de \(veil.faces))")
    check(abs(d.x) + abs(d.y) + abs(d.z) < 1e-3, "même voile après rechargement (centre décalé de \(d))")
    check(!CoverageMap().load(Array(bytes.dropLast())), "sauvegarde tronquée refusée")
    check(!CoverageMap().load([]), "sauvegarde vide refusée")
}

if failures == 0 { print("coverage_tests : tous les tests passent") } else { print("coverage_tests : \(failures) échec(s)"); exit(1) }
