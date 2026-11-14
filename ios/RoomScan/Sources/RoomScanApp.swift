// RoomScan : scan LiDAR d'une pièce avec le maillage temps réel d'ARKit, export du dossier brut (images, profondeur,
// confiance, poses) lu par scan_to_domain.py et du maillage (mesh.ply binaire, classe par face). Capture à mémoire
// bornée pour un appartement entier ; journal dans capture_log.txt.
import SwiftUI

@main
struct RoomScanApp: App {
    @StateObject private var store = ScanStore()
    @StateObject private var settings = ScanSettings()

    var body: some Scene {
        WindowGroup {
            ScansListView()
                .environmentObject(store)
                .environmentObject(settings)
        }
    }
}
