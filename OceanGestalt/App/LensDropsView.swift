import SwiftUI

// ---------------------------------------------------------------------------
// LensDropsView — 2D screen-space water-drop overlay.
// Each drop uses .background(.ultraThinMaterial) so iOS composites the live
// scene through the drop with genuine blur/distortion. Shape parameters are
// randomised at spawn time so every drop has a distinct organic silhouette.
// ---------------------------------------------------------------------------

struct LensDropsView: View {

    /// Average number of new drops per second.
    var spawnRate: Double = 0.7

    @State private var drops:     [LensDrop] = []
    @State private var lastSpawn: Date       = .distantPast

    var body: some View {
        TimelineView(.animation) { timeline in
            GeometryReader { geo in
                ZStack {
                    ForEach(drops) { drop in
                        dropView(drop: drop, at: timeline.date, size: geo.size)
                    }
                }
            }
            .onChange(of: timeline.date) { _, date in
                drops.removeAll { $0.isExpired(at: date) }
                maybeSpawn(at: date)
            }
        }
        .allowsHitTesting(false)
        .ignoresSafeArea()
    }

    // MARK: - Drop view

    @ViewBuilder
    private func dropView(drop: LensDrop, at date: Date, size: CGSize) -> some View {
        let w   = drop.width
        let h   = drop.width * 1.45
        let pos = drop.position(at: date, in: size)
        let sp  = drop.shapeParams

        ZStack {
            // Frosted body — material distorts the live scene, shape gives the silhouette
            TearDropShape(params: sp)
                .fill(.white.opacity(0.04))
                .background(.ultraThinMaterial, in: TearDropShape(params: sp))
                .frame(width: w, height: h)

            // Inner brightness centred on the round bulge
            TearDropShape(params: sp)
                .fill(
                    RadialGradient(
                        colors: [.white.opacity(0.12), .clear],
                        center: UnitPoint(x: 0.5, y: 0.65),
                        startRadius: 0,
                        endRadius: w * 0.55
                    )
                )
                .frame(width: w, height: h)

            // Specular highlight — small soft ellipse, upper-left
            Ellipse()
                .fill(.white.opacity(0.55))
                .frame(width: w * 0.28, height: w * 0.18)
                .offset(x: -w * 0.14, y: -h * 0.18)
                .blur(radius: w * 0.08)
        }
        // Gradient mask fades edges so the boundary dissolves rather than cuts off
        .mask {
            TearDropShape(params: sp)
                .fill(
                    RadialGradient(
                        colors: [.white, .white.opacity(0.65)],
                        center: UnitPoint(x: 0.5, y: 0.65),
                        startRadius: 0,
                        endRadius: w * 0.58
                    )
                )
                .frame(width: w, height: h)
        }
        .frame(width: w, height: h)
        .rotationEffect(.degrees(drop.tilt))
        .scaleEffect(drop.scale(at: date))
        .position(pos)
        .opacity(drop.opacity(at: date))
    }

    // MARK: - Spawn

    private func maybeSpawn(at date: Date) {
        guard date.timeIntervalSince(lastSpawn) >= 1.0 / spawnRate else { return }
        lastSpawn = date
        let startY   = 0.05 + pow(Double.random(in: 0...1), 0.5) * 0.85
        let duration = Double.random(in: 0.5...3.0)
        drops.append(LensDrop(
            x:           Double.random(in: 0.05...0.95),
            startY:      startY,
            width:       CGFloat.random(in: 22...62),
            slideRate:   (1.0 - startY) / duration,
            lifetime:    duration,
            growDuration: Double.random(in: 0.25...0.55),
            tilt:        Double.random(in: -6...6),
            shapeParams: .random(),
            spawnDate:   date
        ))
    }
}

// MARK: - Teardrop shape
// Parameterised so each drop gets its own silhouette.
// tipX:     horizontal offset of the tip from centre (–0.5…0.5 of half-width)
// leftCtrl: Y-fractions for the left bezier's upper and lower control points
// rightCtrl:same for the right side — can differ, creating asymmetry
// leftX/rightX: how far left/right each side bulges (fraction of half-width)

private struct TearDropParams {
    var tipX:        CGFloat  // offset of tip from centre, in half-width units
    var leftCtrlY1:  CGFloat  // upper left control point Y, fraction of height
    var rightCtrlY1: CGFloat  // upper right control point Y, fraction of height
    var leftBulgeX:  CGFloat  // how far left side reaches, fraction of half-width
    var rightBulgeX: CGFloat  // how far right side reaches, fraction of half-width
    // Lower control points are fixed at Y = 1.0 (bottom of rect) so both
    // curves arrive horizontally — smooth rounded bottom, no pinch/cusp.

    static func random() -> TearDropParams {
        TearDropParams(
            tipX:        CGFloat.random(in: -0.30...0.30),
            leftCtrlY1:  CGFloat.random(in: 0.14...0.28),
            rightCtrlY1: CGFloat.random(in: 0.14...0.28),
            leftBulgeX:  CGFloat.random(in: 0.85...1.15),
            rightBulgeX: CGFloat.random(in: 0.85...1.15)
        )
    }
}

private struct TearDropShape: Shape {
    var params: TearDropParams

    func path(in rect: CGRect) -> Path {
        let h   = rect.height
        let cx  = rect.midX
        // tip can be offset left/right of centre by a fraction of the half-width
        let tip = CGPoint(x: cx + params.tipX * cx, y: rect.minY)
        let bot = CGPoint(x: cx, y: rect.maxY)
        // leftBulgeX/rightBulgeX scale the half-width: 1.0 = full rect edge, >1 extends beyond
        let leftX  = cx - params.leftBulgeX  * cx   // 0 when bulgeX = 1.0
        let rightX = cx + params.rightBulgeX * cx   // w when bulgeX = 1.0

        var p = Path()
        p.move(to: tip)

        // Left side — lower control point at Y = h so the curve arrives
        // at the bottom horizontally, giving a smooth rounded base.
        p.addCurve(
            to: bot,
            control1: CGPoint(x: leftX, y: h * params.leftCtrlY1),
            control2: CGPoint(x: leftX, y: h)
        )

        // Right side — same trick: lower control at Y = h for smooth join.
        p.addCurve(
            to: tip,
            control1: CGPoint(x: rightX, y: h),
            control2: CGPoint(x: rightX, y: h * params.rightCtrlY1)
        )

        return p
    }
}

// MARK: - Drop model

private struct LensDrop: Identifiable {
    let id          = UUID()
    let x:           Double
    let startY:      Double
    let width:       CGFloat
    let slideRate:   Double
    let lifetime:    Double
    let growDuration: Double       // seconds to expand from zero to full size
    let tilt:         Double       // degrees, slight lean left/right
    let shapeParams:  TearDropParams
    let spawnDate:    Date

    func position(at date: Date, in size: CGSize) -> CGPoint {
        let age = date.timeIntervalSince(spawnDate)
        return CGPoint(
            x: x * size.width,
            y: (startY + slideRate * age) * size.height
        )
    }

    func scale(at date: Date) -> Double {
        let t = min(1.0, date.timeIntervalSince(spawnDate) / growDuration)
        // Ease-out: decelerate into full size
        return 1.0 - pow(1.0 - t, 2.5)
    }

    func opacity(at date: Date) -> Double {
        let age       = date.timeIntervalSince(spawnDate)
        let fadeStart = lifetime * 0.82
        guard age >= fadeStart else { return 1.0 }
        return max(0, 1 - (age - fadeStart) / (lifetime * 0.18))
    }

    func isExpired(at date: Date) -> Bool {
        date.timeIntervalSince(spawnDate) >= lifetime
    }
}
