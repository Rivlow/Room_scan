import SwiftUI

/// Réglages du scan, persistants (UserDefaults).
final class ScanSettings: ObservableObject {
    @Published var captureKeyframes: Bool { didSet { defaults.set(captureKeyframes, forKey: "captureKeyframes") } }
    @Published var keyframeInterval: Double { didSet { defaults.set(keyframeInterval, forKey: "keyframeInterval") } }   // s
    @Published var minMove: Double { didSet { defaults.set(minMove, forKey: "minMove") } }                               // m
    @Published var minRotate: Double { didSet { defaults.set(minRotate, forKey: "minRotate") } }                         // degrés
    @Published var jpegQuality: Double { didSet { defaults.set(jpegQuality, forKey: "jpegQuality") } }
    @Published var showMesh: Bool { didSet { defaults.set(showMesh, forKey: "showMesh") } }
    @Published var smoothedDepth: Bool { didSet { defaults.set(smoothedDepth, forKey: "smoothedDepth") } }

    private let defaults = UserDefaults.standard

    init() {
        let d = UserDefaults.standard
        captureKeyframes = d.object(forKey: "captureKeyframes") as? Bool ?? true
        keyframeInterval = d.object(forKey: "keyframeInterval") as? Double ?? 0.25
        minMove = d.object(forKey: "minMove") as? Double ?? 0.08
        minRotate = d.object(forKey: "minRotate") as? Double ?? 8.0
        jpegQuality = d.object(forKey: "jpegQuality") as? Double ?? 0.85
        showMesh = d.object(forKey: "showMesh") as? Bool ?? true
        smoothedDepth = d.object(forKey: "smoothedDepth") as? Bool ?? false
    }
}

struct SettingsView: View {
    @EnvironmentObject var settings: ScanSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            Section {
                Toggle("Enregistrer les images clés", isOn: $settings.captureKeyframes)
                VStack(alignment: .leading) {
                    Text("Intervalle minimal : \(settings.keyframeInterval, specifier: "%.2f") s")
                    Slider(value: $settings.keyframeInterval, in: 0.1...2.0, step: 0.05)
                }
                VStack(alignment: .leading) {
                    Text("Déplacement minimal : \(Int(settings.minMove * 100)) cm")
                    Slider(value: $settings.minMove, in: 0.02...0.5, step: 0.01)
                }
                VStack(alignment: .leading) {
                    Text("Rotation minimale : \(Int(settings.minRotate))°")
                    Slider(value: $settings.minRotate, in: 2...30, step: 1)
                }
                VStack(alignment: .leading) {
                    Text("Qualité JPEG : \(Int(settings.jpegQuality * 100)) %")
                    Slider(value: $settings.jpegQuality, in: 0.5...1.0, step: 0.05)
                }
            } header: {
                Text("Images clés")
            } footer: {
                Text("Une image clé (photo 1024×768, profondeur LiDAR 256×192 en mm, confiance, pose) est enregistrée au plus "
                     + "une fois par intervalle, et seulement si le téléphone a bougé ou tourné d'au moins ces seuils. "
                     + "C'est ce dossier que lit scan_to_domain.py.")
            }
            Section {
                Toggle("Maillage visible pendant le scan", isOn: $settings.showMesh)
                Toggle("Profondeur lissée (smoothedSceneDepth)", isOn: $settings.smoothedDepth)
            } header: {
                Text("Capture")
            } footer: {
                Text("La profondeur brute est préférable pour la géométrie : la version lissée est filtrée dans le temps "
                     + "et peut traîner sur les bords.")
            }
        }
        .navigationTitle("Réglages")
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("OK") { dismiss() }
            }
        }
    }
}
