import SwiftUI
import RealityKit
import ARKit

struct ARViewContainer: UIViewRepresentable {
    let arView: ARView
    func makeUIView(context: Context) -> ARView { arView }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

/// Écran de scan : caméra + maillage temps réel, bandeau d'état, Pause / Terminer, résumé final.
/// Avec `resuming` : reprise d'un scan existant dans son repère (carte ARKit rechargée).
struct ScanView: View {
    var resuming: ScanRecord? = nil
    @EnvironmentObject var store: ScanStore
    @EnvironmentObject var settings: ScanSettings
    @Environment(\.dismiss) private var dismiss
    @StateObject private var controller = ARSessionController()
    @State private var askCancel = false
    @State private var showPreview = false
    @State private var showLivePreview = false

    var body: some View {
        ZStack {
            ARViewContainer(arView: controller.arView).ignoresSafeArea()
            VStack(spacing: 8) {
                statusBanner
                Spacer()
                bottomPanel
            }
            .padding()
        }
        .navigationBarBackButtonHidden(controller.phase != .done && controller.phase != .idle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                if controller.phase == .scanning || controller.phase == .paused {
                    Button("Annuler") { askCancel = true }
                }
            }
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                Button {
                    controller.setVeilVisible(!controller.veilVisible)
                } label: {
                    Image(systemName: controller.veilVisible ? "exclamationmark.triangle.fill" : "exclamationmark.triangle")
                }
                .tint(.red)
                .disabled(controller.phase != .scanning && controller.phase != .paused)
                Menu {
                    Picker("Visualisation", selection: Binding(get: { controller.viewMode }, set: { controller.setViewMode($0) })) {
                        ForEach(ViewMode.allCases) { m in Label(m.title, systemImage: m.icon).tag(m) }
                    }
                } label: {
                    Image(systemName: controller.viewMode.icon)
                }
                .disabled(controller.phase == .done || controller.phase == .failed)
            }
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            controller.start(settings: settings, store: store, resuming: resuming)
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            controller.stop()
        }
        .confirmationDialog(resuming == nil ? "Abandonner ce scan ?" : "Abandonner la reprise ?", isPresented: $askCancel,
                            titleVisibility: .visible) {
            Button(resuming == nil ? "Supprimer le scan" : "Abandonner (le scan précédent est conservé)", role: .destructive) {
                controller.cancel()
                dismiss()
            }
            Button("Continuer", role: .cancel) {}
        }
        .alert("Erreur", isPresented: Binding(get: { controller.errorMessage != nil && controller.phase == .failed },
                                               set: { _ in })) {
            Button("Retour") {
                controller.cancel()
                dismiss()
            }
        } message: {
            Text(controller.errorMessage ?? "")
        }
        .sheet(isPresented: $showLivePreview) {
            // le scan tel qu'il est en pause ; la session ARKit continue derrière, on reprend au même endroit
            let snap = controller.snapshot()
            NavigationStack {
                MeshPreviewView(title: "Scan en pause") {
                    (LiveMesh.read(snap.anchors), snap.veil.faces > 0 ? MeshGeometry.veilMesh(snap.veil) : nil)
                }
                    .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Retour") { showLivePreview = false } } }
            }
        }
        .sheet(isPresented: $showPreview) {
            if let url = controller.scanURL {
                NavigationStack {
                    MeshPreviewView(scan: ScanRecord(url: url, meta: controller.result, bytes: 0))
                        .toolbar { ToolbarItem(placement: .confirmationAction) { Button("OK") { showPreview = false } } }
                }
            }
        }
    }

    private var statusBanner: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle().fill(controller.trackingGood ? Color.green : Color.orange).frame(width: 10, height: 10)
                Text(controller.trackingMessage).font(.subheadline.weight(.medium))
                Spacer()
                Text(Self.clock(controller.elapsed)).font(.subheadline.monospacedDigit())
            }
            HStack(spacing: 12) {
                if controller.viewMode.drawsPoints {
                    Label(Self.compact(controller.pointCount) + " points", systemImage: "circle.grid.3x3")
                } else {
                    Label(Self.compact(controller.triangleCount) + " triangles", systemImage: "triangle")
                }
                Label("\(controller.keyframeCount) images", systemImage: "photo.stack")
                Spacer()
                Label(String(format: "%.1f Go", controller.freeGigabytes), systemImage: "internaldrive")
            }
            .font(.caption)
            if let notice = controller.cloudNotice, controller.viewMode.drawsPoints {
                Label(notice, systemImage: "exclamationmark.circle").font(.caption).foregroundColor(.orange)
            }
            if !controller.mappingStatus.isEmpty && controller.phase == .scanning {
                Label("Carte ARKit : \(controller.mappingStatus)", systemImage: "map").font(.caption).foregroundColor(.secondary)
            }
            if controller.resuming && !controller.relocalized && controller.phase == .scanning {
                Label("Reprise : revenez à un endroit déjà scanné, dans le même éclairage, et balayez lentement",
                      systemImage: "arrow.triangle.2.circlepath")
                    .font(.caption).foregroundColor(.yellow)
            }
            if controller.elapsed > 20 && controller.ceilingShare < 0.03 && controller.phase == .scanning && !controller.resuming {
                Label("Pensez à viser le plafond", systemImage: "arrow.up.to.line").font(.caption).foregroundColor(.yellow)
            }
            if controller.freeGigabytes < 1 && controller.phase == .scanning {
                Label("Espace disque faible", systemImage: "exclamationmark.triangle").font(.caption).foregroundColor(.red)
            }
            if let notice = controller.captureNotice {
                Label(notice, systemImage: "exclamationmark.circle").font(.caption).foregroundColor(.orange)
            }
            if controller.veilVisible && controller.missingArea > 0 && controller.phase != .done {
                Label(String(format: "Zones manquantes (en rouge) : %.1f m²", controller.missingArea), systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundColor(.red)
            }
            if controller.phase == .paused {
                Label("En pause : le suivi continue, vous pouvez vous déplacer et reprendre", systemImage: "pause.circle")
                    .font(.caption).foregroundColor(.secondary)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private var bottomPanel: some View {
        switch controller.phase {
        case .idle, .failed:
            EmptyView()
        case .scanning, .paused:
            VStack(spacing: 10) {
                if controller.phase == .paused {
                    Button { showLivePreview = true } label: {
                        Label("Voir le scan", systemImage: "cube").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
                HStack(spacing: 16) {
                    Button {
                        if controller.phase == .scanning { controller.pause() } else { controller.resume() }
                    } label: {
                        Label(controller.phase == .scanning ? "Pause" : "Reprendre",
                              systemImage: controller.phase == .scanning ? "pause.fill" : "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    Button {
                        controller.finish()
                    } label: {
                        Label("Terminer", systemImage: "checkmark.circle.fill").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .controlSize(.large)
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        case .finishing:
            HStack {
                ProgressView()
                Text("Export de la carte, du maillage et des images clés…").padding(.leading, 8)
            }
            .padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        case .done:
            VStack(alignment: .leading, spacing: 8) {
                Text("Scan terminé").font(.headline)
                if let m = controller.result {
                    let skipped = (m.keyframesSkipped ?? 0) > 0 ? " · \(m.keyframesSkipped ?? 0) ignorées" : ""
                    let sessions = (m.sessions ?? 1) > 1 ? " · \(m.sessions ?? 1) sessions" : ""
                    Text("\(Self.compact(m.triangles)) triangles · \(m.keyframes) images clés\(skipped) · \(Self.clock(m.durationSeconds))\(sessions)")
                        .font(.subheadline)
                    if m.worldMapBytes == nil {
                        Text("Carte ARKit non sauvegardée : ce scan ne pourra pas être repris").font(.caption).foregroundColor(.orange)
                    }
                    if let f = m.floorHeight, let c = m.ceilingHeight {
                        Text(String(format: "Hauteur sous plafond estimée : %.2f m", c - f)).font(.subheadline)
                    } else {
                        Text("Plafond ou sol non reconnu : scan_to_domain.py utilisera --height").font(.caption).foregroundColor(.secondary)
                    }
                    if let u = controller.scanURL {
                        Text(u.lastPathComponent).font(.caption.monospaced()).foregroundColor(.secondary)
                    }
                }
                HStack {
                    Button { showPreview = true } label: { Label("Voir le maillage", systemImage: "cube") }
                        .buttonStyle(.bordered)
                    Spacer()
                    Button("OK") { dismiss() }.buttonStyle(.borderedProminent)
                }
            }
            .padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        }
    }

    static func clock(_ t: TimeInterval) -> String {
        let s = Int(t.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    static func compact(_ n: Int) -> String {
        n >= 1_000_000 ? String(format: "%.1f M", Double(n) / 1e6) : n >= 1000 ? String(format: "%.0f k", Double(n) / 1e3) : "\(n)"
    }
}
