import SwiftUI

struct BenchConfig: Identifiable {
    let id = UUID()
    let n: UInt32
    let fold: UInt32
    let rate: UInt32
    var cpuMs: Double?
    var fusedMs: Double?
    var grindMs: Double?
    var speedup: Double? {
        guard let cpu = cpuMs else { return nil }
        let bestGpu = [fusedMs, grindMs].compactMap { $0 }.min()
        guard let gpu = bestGpu, gpu > 0 else { return nil }
        return cpu / gpu
    }
}

struct ContentView: View {
    @State private var configs: [BenchConfig] = []
    @State private var running = false
    @State private var currentConfig = ""
    @State private var deviceInfo = ""

    private let defaultConfigs: [(UInt32, UInt32, UInt32)] = [
        (20, 1, 1), (20, 2, 2), (20, 4, 3),
        (22, 1, 1), (22, 1, 2), (22, 2, 1), (22, 3, 2), (22, 4, 3),
        (24, 1, 1), (24, 2, 1), (24, 3, 1), (24, 4, 1),
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if !deviceInfo.isEmpty {
                    Text(deviceInfo)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 4)
                }

                if configs.isEmpty {
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "cpu")
                            .font(.system(size: 48))
                            .foregroundStyle(.blue)
                        Text("WHIR Prover Benchmark")
                            .font(.title2)
                        Text("Compares CPU vs GPU (Metal) proving time")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                } else {
                    List {
                        Section {
                            HStack {
                                Text("n").bold().frame(width: 30, alignment: .leading)
                                Text("fold").bold().frame(width: 35, alignment: .leading)
                                Text("rate").bold().frame(width: 35, alignment: .leading)
                                Text("CPU").bold().frame(width: 60, alignment: .trailing)
                                Text("GPU").bold().frame(width: 60, alignment: .trailing)
                                Text("Speed").bold().frame(width: 50, alignment: .trailing)
                            }
                            .font(.caption)
                        }

                        ForEach(configs) { config in
                            HStack {
                                Text("\(config.n)")
                                    .frame(width: 30, alignment: .leading)
                                Text("\(config.fold)")
                                    .frame(width: 35, alignment: .leading)
                                Text("\(config.rate)")
                                    .frame(width: 35, alignment: .leading)
                                Text(config.cpuMs.map { String(format: "%.0f", $0) } ?? "...")
                                    .frame(width: 60, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                                Text(bestGpuText(config))
                                    .frame(width: 60, alignment: .trailing)
                                    .foregroundStyle(.blue)
                                Text(config.speedup.map { String(format: "%.1fx", $0) } ?? "-")
                                    .frame(width: 50, alignment: .trailing)
                                    .foregroundStyle(
                                        (config.speedup ?? 0) > 1.0 ? .green : .red
                                    )
                            }
                            .font(.system(.body, design: .monospaced))
                        }
                    }
                    .listStyle(.plain)
                }

                if running {
                    HStack {
                        ProgressView()
                        Text("Running: \(currentConfig)")
                            .font(.caption)
                    }
                    .padding()
                }
            }
            .navigationTitle("WHIR Bench")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: startBenchmark) {
                        Label("Run", systemImage: "play.fill")
                    }
                    .disabled(running)
                }
                ToolbarItem(placement: .secondaryAction) {
                    ShareLink(item: exportResults())
                }
            }
        }
        .onAppear {
            if let cStr = whir_device_info() {
                deviceInfo = String(cString: cStr)
            }
        }
    }

    private func bestGpuText(_ c: BenchConfig) -> String {
        let best = [c.fusedMs, c.grindMs].compactMap { $0 }.min()
        return best.map { String(format: "%.0f", $0) } ?? "..."
    }

    private func startBenchmark() {
        running = true
        configs = defaultConfigs.map { BenchConfig(n: $0.0, fold: $0.1, rate: $0.2) }

        DispatchQueue.global(qos: .userInitiated).async {
            for i in 0..<configs.count {
                let c = configs[i]
                DispatchQueue.main.async {
                    currentConfig = "n=\(c.n) fold=\(c.fold) rate=\(c.rate)"
                }

                let cpuResult = whir_bench(c.n, c.fold, c.rate, 1)
                DispatchQueue.main.async {
                    configs[i].cpuMs = cpuResult.cpu_ms >= 0 ? cpuResult.cpu_ms : nil
                }

                let gpuResult = whir_bench(c.n, c.fold, c.rate, 2)
                DispatchQueue.main.async {
                    configs[i].fusedMs = gpuResult.fused_ms >= 0 ? gpuResult.fused_ms : nil
                }

                let grindResult = whir_bench(c.n, c.fold, c.rate, 3)
                DispatchQueue.main.async {
                    configs[i].grindMs = grindResult.grind_ms >= 0 ? grindResult.grind_ms : nil
                }
            }

            DispatchQueue.main.async {
                running = false
                currentConfig = ""
            }
        }
    }

    private func exportResults() -> String {
        var lines = [
            "# WHIR Bench Results",
            "# Device: \(deviceInfo)",
            "# Date: \(Date())",
            "",
            "n\tfold\trate\tCPU(ms)\tGPU(ms)\tspeedup",
        ]
        for c in configs {
            let cpu = c.cpuMs.map { String(format: "%.1f", $0) } ?? "fail"
            let best = [c.fusedMs, c.grindMs].compactMap { $0 }.min()
            let gpu = best.map { String(format: "%.1f", $0) } ?? "fail"
            let sp = c.speedup.map { String(format: "%.2fx", $0) } ?? "-"
            lines.append("\(c.n)\t\(c.fold)\t\(c.rate)\t\(cpu)\t\(gpu)\t\(sp)")
        }
        return lines.joined(separator: "\n")
    }
}
