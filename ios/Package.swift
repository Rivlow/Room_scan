// swift-tools-version: 6.0
// Build xtool (Linux / Windows via WSL) : voir README_xtool.md.  Le projet Xcode (project.yml) reste valable sur Mac.
import PackageDescription

let package = Package(
    name: "RoomScan",
    platforms: [.iOS(.v16)],
    products: [
        // xtool transforme l'unique produit bibliothèque en application (exécutable RoomScan).
        .library(name: "RoomScan", targets: ["RoomScan"]),
    ],
    targets: [
        .target(
            name: "RoomScan",
            path: "RoomScan/Sources"
        ),
    ],
    // Code écrit pour Swift 5.9 : pas de vérification stricte de la concurrence Swift 6.
    swiftLanguageModes: [.v5]
)
