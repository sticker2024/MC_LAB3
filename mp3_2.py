"""
Mini Project 3.2 - Cellular capacity and frequency-reuse planning
Mobile Communications, Unit 3 lab.

Run:  python3 mp3_2.py     (writes table, charts and printed results to ./out_3_2)
"""
import os
import math
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_3_2")
os.makedirs(OUT, exist_ok=True)

# =========================================================== 1. CONFIGURATION
TOTAL_SPECTRUM_MHZ = 25.0      # total spectrum allocation held by the operator
CHANNEL_BW_KHZ = 200.0         # bandwidth of one channel
CANDIDATE_N = [1, 3, 4, 7, 9, 12]   # at least 1, 3, 4, 7 required
AREA_KM2 = 1000.0              # fixed coverage area
R_KM = 2.0                     # baseline cell radius (centre to vertex)
SPLIT_FACTORS = [2, 4]         # radius divided by this factor in cell splitting
SPLIT_N = 7                    # reuse factor used for the splitting scenario
SECTORS = 3                    # three-sector sites (120-degree)

# =========================================================== 2. CHANNELS
total_channels = int(TOTAL_SPECTRUM_MHZ * 1000.0 // CHANNEL_BW_KHZ)


def channels_per_cell(N):
    return total_channels // N          # whole channels only


# =========================================================== 3. CELL COUNT
def hex_area(R):
    """Area of a regular hexagon with circumradius R: (3*sqrt(3)/2) R^2."""
    return 1.5 * math.sqrt(3) * R * R


def cells_in_area(R, area=AREA_KM2):
    return int(area // hex_area(R))     # approximate: whole cells that fit


def system_capacity(N, R):
    return channels_per_cell(N) * cells_in_area(R)


# =========================================================== 4. Q
def Q_of(N):
    return math.sqrt(3 * N)


# =========================================================== main table
rows = []
for N in CANDIDATE_N:
    cpc = channels_per_cell(N)
    nc = cells_in_area(R_KM)
    rows.append(dict(N=N, Q=Q_of(N), channels_per_cell=cpc, cells=nc,
                     capacity=cpc * nc))

print("=== Inputs ===")
print(f"Total spectrum        : {TOTAL_SPECTRUM_MHZ} MHz")
print(f"Channel bandwidth     : {CHANNEL_BW_KHZ} kHz")
print(f"Total channels        : {total_channels}")
print(f"Coverage area         : {AREA_KM2} km^2")
print(f"Cell radius R         : {R_KM} km  -> hexagon area {hex_area(R_KM):.3f} km^2")
print(f"Cells that fit        : {cells_in_area(R_KM)}")

print("\n=== Capacity versus reuse factor (omni cells, R = %.1f km) ===" % R_KM)
print(f"{'N':>3} | {'Q':>6} | {'channels/cell':>13} | {'cells':>6} | {'system capacity (channels)':>26}")
print("-" * 68)
for r in rows:
    print(f"{r['N']:>3} | {r['Q']:>6.2f} | {r['channels_per_cell']:>13d} | {r['cells']:>6d} | {r['capacity']:>26d}")

# =========================================================== 5. CELL SPLITTING
print(f"\n=== Cell splitting, N = {SPLIT_N} ===")
split_rows = []
base_cap = system_capacity(SPLIT_N, R_KM)
print(f"{'R (km)':>7} | {'cells':>6} | {'channels/cell':>13} | {'capacity':>9} | {'gain vs base':>12}")
print("-" * 58)
for f in [1] + SPLIT_FACTORS:
    Rn = R_KM / f
    nc = cells_in_area(Rn)
    cap = channels_per_cell(SPLIT_N) * nc
    split_rows.append(dict(factor=f, R=Rn, cells=nc, capacity=cap, gain=cap / base_cap))
    print(f"{Rn:>7.2f} | {nc:>6d} | {channels_per_cell(SPLIT_N):>13d} | {cap:>9d} | {cap / base_cap:>11.2f}x")
print("(Q is unchanged by splitting: Q depends only on N, not on R.)")

# =========================================================== 6. SECTORING
# Simplified first-tier illustration.  In an omni layout each cell has 6 first-tier
# co-channel interferers.  With 120-degree sectors a sector's directional antenna
# sees only ~2 of them (the others lie outside its beam / in its back lobe).
# Using the common far-field approximation  SIR = (1/i0) * Q^gamma  with path-loss
# exponent gamma:
GAMMA = 4.0
print(f"\n=== Sectoring illustration (path-loss exponent gamma = {GAMMA}) ===")


def sir_db(N, interferers, gamma=GAMMA):
    return 10 * math.log10((Q_of(N) ** gamma) / interferers)


print(f"{'N':>3} | {'Q':>5} | {'SIR omni (6 interferers)':>25} | {'SIR 3-sector (2 interferers)':>29} | gain")
print("-" * 82)
sec_rows = []
for N in CANDIDATE_N:
    o = sir_db(N, 6)
    s = sir_db(N, 2)
    sec_rows.append(dict(N=N, Q=Q_of(N), sir_omni=o, sir_sec=s))
    print(f"{N:>3} | {Q_of(N):>5.2f} | {o:>22.1f} dB | {s:>26.1f} dB | {s - o:+.1f} dB")

# How tight a reuse does sectoring allow for the same SIR as omni N=7 ?
target = sir_db(7, 6)
TOL_DB = 0.5   # treat SIR within 0.5 dB of the omni N=7 value as "the same quality"
print(f"\nOmni N=7 SIR target = {target:.1f} dB (tolerance {TOL_DB} dB)")
n_sec = 7
for N in CANDIDATE_N:
    if sir_db(N, 2) >= target - TOL_DB:
        print(f"Smallest candidate N meeting that SIR with 3 sectors: N = {N} "
              f"(SIR {sir_db(N, 2):.1f} dB)")
        n_sec = N
        break
cap_omni7 = channels_per_cell(7) * cells_in_area(R_KM)
# A 3-sector site still draws its pool of total_channels/N from the cluster; the pool is
# shared among the 3 sectors. So per-site capacity (and system capacity per area) scales
# as 1/N, ignoring the small trunking-efficiency loss from splitting a pool into sectors.
cap_sec = channels_per_cell(n_sec) * cells_in_area(R_KM)
print(f"Capacity omni N=7: {cap_omni7}  |  3-sector at N={n_sec}: {cap_sec}  "
      f"({cap_sec / cap_omni7:.2f}x, before trunking-efficiency loss)")

# =========================================================== 7. CHARTS & TABLE
Ns = [r["N"] for r in rows]
caps = [r["capacity"] for r in rows]
Qs = [r["Q"] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
bars = ax.bar([str(n) for n in Ns], caps, color="tab:blue")
ax.set_xlabel("reuse factor N"); ax.set_ylabel("total system capacity (channels)")
ax.set_title("Capacity versus reuse factor")
for b, c in zip(bars, caps):
    ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=8)
ax2 = ax.twinx()
ax2.plot([str(n) for n in Ns], Qs, "o-", color="tab:red")
ax2.set_ylabel("Q = sqrt(3N)", color="tab:red")
ax = axes[1]
ax.plot(Qs, caps, "o-", color="tab:green")
for q, c, n in zip(Qs, caps, Ns):
    ax.annotate(f"N={n}", (q, c), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("co-channel ratio Q"); ax.set_ylabel("total system capacity (channels)")
ax.set_title("Capacity versus Q (interference protection)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_capacity_vs_N_and_Q.png"), dpi=150)
plt.close(fig)

# levers chart: baseline, splitting, sectoring for each N
fig, ax = plt.subplots(figsize=(11, 5))
w = 0.2
x = np.arange(len(CANDIDATE_N))
base = [channels_per_cell(N) * cells_in_area(R_KM) for N in CANDIDATE_N]
sp2 = [channels_per_cell(N) * cells_in_area(R_KM / 2) for N in CANDIDATE_N]
sp4 = [channels_per_cell(N) * cells_in_area(R_KM / 4) for N in CANDIDATE_N]
ax.bar(x - 1.5 * w, base, w, label=f"baseline R={R_KM:g} km")
ax.bar(x - 0.5 * w, sp2, w, label=f"split: R={R_KM / 2:g} km")
ax.bar(x + 0.5 * w, sp4, w, label=f"split: R={R_KM / 4:g} km")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels([f"N={n}\nQ={Q_of(n):.2f}" for n in CANDIDATE_N])
ax.set_ylabel("total system capacity (channels, log scale)")
ax.set_title("Cell splitting as a capacity lever, across reuse factors")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_capacity_levers.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(Ns, [r["sir_omni"] for r in sec_rows], "o-", label="omni (6 interferers)")
ax.plot(Ns, [r["sir_sec"] for r in sec_rows], "s-", label="3-sector (2 interferers)")
ax.axhline(target, color="gray", ls="--", lw=0.8)
ax.text(Ns[-1], target + 0.3, "omni N=7 SIR", ha="right", fontsize=8)
ax.set_xlabel("reuse factor N"); ax.set_ylabel("approx. SIR (dB)")
ax.set_title(f"Sectoring raises SIR at the same N (gamma={GAMMA:g})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_sectoring_sir.png"), dpi=150)
plt.close(fig)

with open(os.path.join(OUT, "table_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "Q", "channels_per_cell", "cells_at_R", "capacity_baseline",
                f"capacity_R/2", f"capacity_R/4", "SIR_omni_dB", "SIR_3sector_dB"])
    for r, s2, s4, sr in zip(rows, sp2, sp4, sec_rows):
        w.writerow([r["N"], f"{r['Q']:.3f}", r["channels_per_cell"], r["cells"],
                    r["capacity"], s2, s4, f"{sr['sir_omni']:.1f}", f"{sr['sir_sec']:.1f}"])
print(f"\nOutputs written to {OUT}")
