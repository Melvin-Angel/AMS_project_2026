# Phase 4 Results — What the plots are showing

These are the three output plots from running the tracker on real experimental data collected in Copenhagen harbour on 5 March 2026. Here's what each one means in plain terms.

---

## 1. `t8_satellite_map.png` — The satellite map

**What it is:** A bird's-eye view of the harbour with an actual map tile underneath. You can see the quay, the water, and the tracks drawn on top of it.

**What's on it:**
- **Dashed coloured lines** — the ground truth. These come from GNSS (GPS fixes from the vessel we know the position of) and AIS (the transponder signal from a ship broadcasting its own position). This is what *actually* happened.
- **Solid coloured lines (red and orange)** — the tracks our tracker produced that matched the ground truth. These are the ones we can evaluate.
- **Faint grey dots** — confirmed clutter tracks. The radar detected something real (a buoy, a quay wall reflection, another vessel) but we have no ground truth for it so we can't say if the tracker is right or wrong.

**What to notice:** The two coloured tracks follow the dashed ground truth lines reasonably well but with an offset of about 57–60 metres. That offset is the whole point of Phase 4 — the tracker *is* following the right target, but its position estimate is off because the sensor calibration isn't perfect.

**Why is there so much clutter?** Copenhagen harbour is a busy, cluttered environment. Radar bounces off quay walls, moored vessels, and buoys constantly. The simulation had none of that, so the tracker behaves very differently here.

---

## 2. `t8_metrics_comparison.png` — The numbers

**What it is:** A 2×3 grid of plots that lets you compare the tracker's performance on real data (red, top row) versus simulation (blue, bottom row), plus a bar chart on the right.

### Top-left: Real MOTP over time
MOTP stands for "Mean Object Tracking Precision" — basically, how far off is the tracker's estimated position from the ground truth at each moment in time, averaged over all matched tracks. Lower is better.

The curve starts around 90–120 m when the vessel is nearby (t=200–280 s), then drops to ~50 m as it moves further away. This sounds backwards but makes sense: at close range, a small angular error in the radar bearing translates to a large absolute distance. At 1 km range, even a 3° bearing offset = 52 m sideways error.

The mean is **53.4 m**.

### Bottom-left: Simulation MOTP over time
The same metric but for Scenario E (the simulation). Mostly under 10 m, with a few spikes from short false-alarm tracks that happened to get confirmed for 2–3 scans. These spikes are noise in the simulation, not real targets.

The mean is **10.1 m** — about **5× better** than real data.

### Top-middle: Real CE over time
CE = "Cardinality Error" — the absolute difference between how many targets the tracker thinks are there vs how many are actually there. 0 is perfect; higher means the tracker is either creating too many tracks or missing targets.

In real data it bounces between 1 and 7, mean = 3.4. This happens because the tracker keeps confirming clutter tracks (it only needs 2 detections in 5 scans to confirm, which is easy in a cluttered harbour), so at any given time it thinks there are more targets than there really are.

### Bottom-middle: Simulation CE over time
Plateaus at exactly 5 for most of the scenario. This is not the tracker missing targets — it's 11 confirmed tracks minus 6 true targets = 5 excess tracks, all from short false-alarm tracks created during the initial phase when returns are dense. The label says "excess = false-alarm tracks" for this reason.

### Right: RMSE bar chart
RMSE (Root Mean Squared Error) is similar to MOTP but squares the errors before averaging, which penalises large misses more heavily. These bars show the per-track RMSE for the two tracks we matched to ground truth:
- **T4 → AIS vessel (MMSI 219384000): 57.6 m**
- **T32 → GNSS vessel: 60.4 m**

The dashed blue line at 10.1 m is the simulation's average MOTP for reference. The real tracks are about 6× above it.

---

## 3. `t8_sim_vs_real_scene.png` — The side-by-side comparison

**What it is:** Two panels showing the same tracker running on two very different inputs — left is the real Copenhagen data, right is Scenario E from the simulation.

### Left panel: Real data
- Dashed lines = GNSS/AIS ground truth
- Solid coloured lines = matched tracks (Track 4 in red, Track 32 in orange)
- Grey dots = clutter tracks (confirmed but unmatched — hidden moving unmatched ones to keep it readable)

You can see the two coloured tracks following their dashed references but with a visible offset. The harbour area is also full of grey dots from clutter — the tracker is working hard but generating a lot of false positives.

### Right panel: Simulation Scenario E
- Dashed lines = true simulated target paths
- Solid coloured lines = tracker estimates for each target

The tracks lie almost directly on top of the dashed ground truths. This is because the simulation has perfect sensor geometry, Gaussian noise, no multipath, and no clutter. It's the "best case" scenario.

**The note on the title says "NED geometry ≠ Copenhagen harbour"** — this is important. The simulation scenario was not designed to look like Copenhagen. The targets move in different directions, the scale is different, and there's no map underneath. The comparison is about *tracker performance quality*, not about the paths looking the same.

---

## Summary: Why is real data so much worse?

Four reasons:

1. **Calibration** — the radar has an estimated 16° rotation offset baked in, but any residual error of even 1–2° causes 17–35 m of position error at 1 km range.

2. **Clutter** — quay walls, buoys, and moored vessels produce constant false returns. The simulation was clean.

3. **Non-Gaussian noise** — the EKF assumes errors are Gaussian (bell curve shaped). Real radar noise has heavier tails (more big outliers), so the filter's gating logic is not well tuned for it.

4. **Limited ground truth** — we only have GNSS on one vessel and AIS from one other. We can't evaluate the tracker on anything else, so the metrics are based on just 2 tracks out of ~50 confirmed.

The improvement proposed to close this gap is an **IMM (Interacting Multiple Model) filter**, which runs multiple motion models in parallel and weights them based on which best explains the current detections. This helps in cluttered environments where targets don't always follow a smooth constant-velocity path.
