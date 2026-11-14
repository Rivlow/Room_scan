import SwiftUI
import UIKit

/// « … » tant que la taille est en cours de calcul.
func sizeText(_ bytes: Int64?) -> String {
    bytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "…"
}

struct ShareItem: Identifiable {
    let url: URL
    var id: String { url.path }
}

struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]
    let onComplete: () -> Void
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let vc = UIActivityViewController(activityItems: items, applicationActivities: nil)
        vc.completionWithItemsHandler = { _, _, _, _ in onComplete() }
        return vc
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// Liste des scans, accès au nouveau scan et aux réglages.
struct ScansListView: View {
    @EnvironmentObject var store: ScanStore
    @State private var showSettings = false
    @State private var share: ShareItem?
    @State private var zipping = false
    @State private var errorText: String?

    var body: some View {
        NavigationStack {
            List {
                if !ARSessionController.isSupported {
                    Section {
                        Label("Cet appareil n'a pas de LiDAR : le scan n'est pas disponible", systemImage: "exclamationmark.triangle")
                            .foregroundColor(.orange)
                    }
                }
                Section {
                    NavigationLink { ScanView() } label: {
                        Label("Nouveau scan", systemImage: "camera.viewfinder").font(.headline)
                    }
                    .disabled(!ARSessionController.isSupported)
                }
                Section("Scans enregistrés") {
                    if store.scans.isEmpty {
                        Text("Aucun scan. Les scans sont dans Fichiers > Sur mon iPhone > RoomScan > Scans.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                    ForEach(store.scans) { scan in
                        NavigationLink(value: scan) { row(scan) }
                    }
                    .onDelete { idx in
                        for i in idx { store.delete(store.scans[i]) }
                    }
                }
            }
            .navigationTitle("RoomScan")
            .navigationDestination(for: ScanRecord.self) { scan in
                ScanDetailView(scan: scan, onShare: { shareScan(scan) })
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                }
            }
            .sheet(isPresented: $showSettings) {
                NavigationStack { SettingsView() }
            }
            .sheet(item: $share) { item in
                ActivityView(items: [item.url]) {
                    store.discardZip(item.url)
                    share = nil
                }
            }
            .overlay {
                if zipping {
                    ProgressView("Compression du scan…")
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                }
            }
            .alert("Erreur", isPresented: Binding(get: { errorText != nil }, set: { if !$0 { errorText = nil } })) {
                Button("OK") { errorText = nil }
            } message: {
                Text(errorText ?? "")
            }
            .onAppear { store.refresh() }
        }
    }

    private func row(_ scan: ScanRecord) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(scan.name).font(.body.monospaced())
            HStack(spacing: 10) {
                if let m = scan.meta {
                    Text("\(ScanView.compact(m.triangles)) tri · \(m.keyframes) img · \(ScanView.clock(m.durationSeconds))")
                } else {
                    Text("incomplet").foregroundColor(.orange)
                }
                Text(sizeText(scan.bytes))
            }
            .font(.caption).foregroundColor(.secondary)
        }
    }

    private func shareScan(_ scan: ScanRecord) {
        zipping = true
        DispatchQueue.global(qos: .userInitiated).async {
            let r = Result { try store.makeZip(of: scan) }
            DispatchQueue.main.async {
                zipping = false
                switch r {
                case .success(let url): share = ShareItem(url: url)
                case .failure(let e): errorText = "Compression impossible : \(e.localizedDescription)"
                }
            }
        }
    }
}

struct ScanDetailView: View {
    @EnvironmentObject var store: ScanStore
    @Environment(\.dismiss) private var dismiss
    let scan: ScanRecord
    let onShare: () -> Void
    @State private var askDelete = false

    var body: some View {
        List {
            if let m = scan.meta {
                Section("Scan") {
                    line("Appareil", m.device)
                    line("iOS", m.systemVersion)
                    line("Début", m.started.formatted(date: .abbreviated, time: .shortened))
                    line("Durée", ScanView.clock(m.durationSeconds))
                    line("Images clés", "\(m.keyframes)")
                    if let s = m.sessions, s > 1 { line("Sessions", "\(s)") }
                    line("Carte ARKit", m.worldMapBytes.map { sizeText(Int64($0)) } ?? "absente")
                    line("Sommets", "\(m.vertices)")
                    line("Triangles", "\(m.triangles)")
                    if let f = m.floorHeight, let c = m.ceilingHeight {
                        line("Hauteur sous plafond", String(format: "%.2f m", c - f))
                    }
                    line("Taille", sizeText(store.scans.first { $0.id == scan.id }?.bytes ?? scan.bytes))
                }
                if !m.classTriangles.isEmpty {
                    Section("Triangles par classe") {
                        ForEach(MeshExporter.classOrder, id: \.self) { c in
                            if let n = m.classTriangles[c] { line(c, "\(n)") }
                        }
                    }
                }
            } else {
                Section { Text("Scan incomplet (pas de scan.json)").foregroundColor(.orange) }
            }
            Section {
                if scan.hasMesh {
                    NavigationLink { MeshPreviewView(scan: scan) } label: { Label("Voir le maillage", systemImage: "cube") }
                }
                if scan.canResume {
                    NavigationLink { ScanView(resuming: scan) } label: {
                        Label("Reprendre le scan (même repère)", systemImage: "arrow.triangle.2.circlepath")
                    }
                    .disabled(!ARSessionController.isSupported)
                }
                Button { onShare() } label: { Label("Partager (zip : AirDrop, Fichiers…)", systemImage: "square.and.arrow.up") }
                Button(role: .destructive) { askDelete = true } label: { Label("Supprimer", systemImage: "trash") }
            } footer: {
                if scan.canResume {
                    Text("Reprendre : la carte ARKit du scan est rechargée ; dès que la caméra reconnaît un endroit déjà scanné, "
                         + "les nouvelles images clés et le nouveau maillage s'ajoutent dans le même repère.")
                }
            }
        }
        .navigationTitle(scan.name)
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog("Supprimer ce scan ?", isPresented: $askDelete, titleVisibility: .visible) {
            Button("Supprimer", role: .destructive) {
                store.delete(scan)
                dismiss()
            }
            Button("Annuler", role: .cancel) {}
        }
    }

    private func line(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k)
            Spacer()
            Text(v).foregroundColor(.secondary)
        }
    }
}
