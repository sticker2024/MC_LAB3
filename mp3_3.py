"""
Mini Project 3.3 - Device connection and handover as a finite-state machine
Mobile Communications, Unit 3 lab.

States
  IDLE                no connection; camps on the strongest acceptable cell
  RACH_PREAMBLE       step 1: preamble (Msg1)
  RACH_RAR            step 2: Random Access Response with timing advance (Msg2)
  RACH_CONNREQ        step 3: RRC connection request (Msg3)
  RACH_CONTENTION     step 4: contention resolution (Msg4)
  CONNECTED           active link to a serving cell
  HANDOVER            handover in progress (source -> target)
Failure path: Radio Link Failure (RLF) from CONNECTED or HANDOVER back to IDLE.

Run:  python3 mp3_3.py       (writes logs, tables, figures to ./out_3_3)
"""
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_3_3")
os.makedirs(OUT, exist_ok=True)

# ============================================================ PARAMETERS
DT_S = 0.2                    # seconds per sample
SPEED_MPS = 20.0              # device speed (72 km/h)
CELL_POS = {"A": 0.0, "B": 1000.0, "C": 2000.0}   # cells on a line, 1 km apart
START_X, END_X = 50.0, 2000.0
T = int((END_X - START_X) / (SPEED_MPS * DT_S))

RLF_THRESH = -110.0           # dBm: serving below this counts as out-of-sync
RLF_COUNT = 3                 # consecutive out-of-sync samples before RLF is declared
CAMP_MIN = -105.0             # dBm needed to camp on a cell from IDLE
RACH_MIN = -110.0             # dBm needed for a preamble to be detected
RACH_MAX_ATTEMPTS = 4
IDLE_WAIT = 5                 # samples spent in IDLE after RLF before cell selection
HO_DURATION = 3               # samples for handover execution (command + target RACH)
PP_WINDOW = 40                # samples (8 s): handover back to a cell left within this = ping-pong
PL_EXP = 33.0                 # path-loss slope (dB per decade)
TA_UNIT_M = 78.12             # one timing-advance step ~ 16 Ts round trip = 78.12 m
FIRST_ATTACH_COLLISIONS = 1   # scripted: first attach loses contention once (shows retry path)

# ============================================================ SCENARIO
def build_scenario(seed=11):
    """RSRP series for cells A, B, C along a straight road.
    path loss  : -45 - 33*log10(d/10)  dBm
    shadowing  : AR(1), sigma 3 dB, rho 0.9, independent per cell
    meas. noise: 1.5 dB white
    Consecutive cells are 1 km apart, so the A/B and B/C overlaps occur around
    x = 500 m and x = 1500 m."""
    rng = np.random.default_rng(seed)
    x = START_X + SPEED_MPS * DT_S * np.arange(T)
    sig = {}
    for name, pos in CELL_POS.items():
        d = np.maximum(np.abs(x - pos), 20.0)
        pl = -45.0 - PL_EXP * np.log10(d / 10.0)
        sh = np.zeros(T)
        for k in range(1, T):
            sh[k] = 0.9 * sh[k - 1] + math.sqrt(1 - 0.81) * 3.0 * rng.normal()
        sig[name] = pl + sh + rng.normal(0, 1.5, T)
    return x, sig


# ============================================================ POLICIES
# Interchangeable: policy(serving, rsrp, mem) -> target cell name or None
#   serving : name of current serving cell
#   rsrp    : dict cell -> RSRP (dBm) at this sample
#   mem     : mutable dict the policy owns (streak counters); reset on each attach/HO
def make_naive():
    def policy(serving, rsrp, mem):
        best = max((c for c in rsrp if c != serving), key=lambda c: rsrp[c])
        return best if rsrp[best] > rsrp[serving] else None
    policy.label = "naive (no margin, no TTT)"
    return policy


def make_hyst_ttt(margin_db, ttt):
    def policy(serving, rsrp, mem):
        best = max((c for c in rsrp if c != serving), key=lambda c: rsrp[c])
        if rsrp[best] > rsrp[serving] + margin_db:
            if mem.get("cand") == best:
                mem["streak"] += 1
            else:
                mem["cand"], mem["streak"] = best, 1
            if mem["streak"] >= ttt:
                return best
        else:
            mem["cand"], mem["streak"] = None, 0
        return None
    policy.label = f"hysteresis {margin_db:g} dB + TTT {ttt} samples"
    return policy


# ============================================================ STATE MACHINE
IDLE, PRE, RAR, MSG3, CRES, CONN, HO = (
    "IDLE", "RACH_PREAMBLE", "RACH_RAR", "RACH_CONNREQ", "RACH_CONTENTION",
    "CONNECTED", "HANDOVER")


def run(sig, x, policy):
    log = []                   # (t, from_state, to_state, cell, trigger)
    state, cell = IDLE, None   # cell: serving cell (CONNECTED/HANDOVER) or RACH cell
    target = None
    mem = {}
    attempts = 0
    collisions_left = FIRST_ATTACH_COLLISIONS
    idle_since = -10 ** 9
    ho_timer = 0
    oos = 0
    ho_events, rlf_events, attach_events = [], [], []

    def go(t, new, c, trig):
        nonlocal state
        log.append((t, state, new, c, trig))
        state = new

    for t in range(T):
        r = {c: float(sig[c][t]) for c in sig}

        if state == IDLE:
            if t - idle_since >= IDLE_WAIT:
                best = max(r, key=r.get)
                if r[best] >= CAMP_MIN:
                    cell, attempts = best, 0
                    go(t, PRE, cell, f"cell selection: {best} at {r[best]:.1f} dBm; "
                                     f"data to send -> send preamble (Msg1)")

        elif state == PRE:
            if r[cell] >= RACH_MIN:
                go(t, RAR, cell, "preamble detected -> Random Access Response (Msg2)")
            else:
                attempts += 1
                if attempts >= RACH_MAX_ATTEMPTS:
                    idle_since = t
                    go(t, IDLE, None, "RACH failed: max attempts reached")

        elif state == RAR:
            ta = int(round(abs(x[t] - CELL_POS[cell]) / TA_UNIT_M))
            go(t, MSG3, cell, f"RAR received, timing advance = {ta} steps -> RRC connection request (Msg3)")

        elif state == MSG3:
            go(t, CRES, cell, "Msg3 sent -> awaiting contention resolution (Msg4)")

        elif state == CRES:
            if collisions_left > 0:
                collisions_left -= 1
                attempts += 1
                go(t, PRE, cell, "contention lost (identity mismatch) -> back off, new preamble")
            else:
                mem = {}
                oos = 0
                attach_events.append((t, cell))
                go(t, CONN, cell, "contention resolved (Msg4) -> RRC connected")

        elif state in (CONN, HO):
            # RLF monitoring on the serving (source) cell, both while connected and mid-handover
            oos = oos + 1 if r[cell] < RLF_THRESH else 0
            if oos >= RLF_COUNT:
                rlf_events.append((t, cell, state))
                idle_since = t
                go(t, IDLE, None,
                   f"RADIO LINK FAILURE: {cell} {r[cell]:.1f} dBm < {RLF_THRESH:.0f} dBm "
                   f"for {RLF_COUNT} samples (too-late handover)"
                   + (f"; handover to {target} aborted" if state == HO else ""))
                continue
            if state == CONN:
                tgt = policy(cell, r, mem)
                if tgt:
                    target, ho_timer = tgt, 0
                    go(t, HO, cell, f"policy fired: {tgt} {r[tgt]:.1f} vs {cell} {r[cell]:.1f} dBm "
                                    f"-> handover command to {tgt}")
            else:  # HO
                ho_timer += 1
                if ho_timer >= HO_DURATION:
                    if r[target] >= RACH_MIN:
                        ho_events.append((t, cell, target))
                        old, cell = cell, target
                        mem, oos = {}, 0
                        go(t, CONN, cell, f"handover complete: {old} -> {cell} (target RACH ok)")
                    else:
                        rlf_events.append((t, cell, HO))
                        idle_since = t
                        go(t, IDLE, None, f"handover failure: target {target} {r[target]:.1f} dBm too weak "
                                          f"-> RLF")
    return log, ho_events, rlf_events, attach_events


def pingpongs(ho_events):
    """Handover into a cell that was left within PP_WINDOW samples earlier."""
    count, pairs = 0, []
    left = {}
    for (t, src, dst) in ho_events:
        if dst in left and t - left[dst] <= PP_WINDOW:
            count += 1
            pairs.append((t, src, dst))
        left[src] = t
    return count, pairs


def serving_timeline(log):
    s = [None] * T
    cur, i = None, 0
    for t in range(T):
        while i < len(log) and log[i][0] <= t:
            _, _, new, c, _ = log[i]
            cur = c if new in (CONN, HO) else None
            i += 1
        s[t] = cur
    return s


# ============================================================ RUN
x, sig = build_scenario()
POLICIES = [make_naive(), make_hyst_ttt(3, 5), make_hyst_ttt(8, 60)]
summary, timelines = [], []
print(f"Scenario: {T} samples x {DT_S}s = {T*DT_S:.0f} s, device {SPEED_MPS} m/s from x={START_X:.0f} to {END_X:.0f} m")
print(f"Cells: {CELL_POS};  RLF below {RLF_THRESH:.0f} dBm for {RLF_COUNT} samples")
for p in POLICIES:
    log, hos, rlfs, attaches = run(sig, x, p)
    pp, pp_list = pingpongs(hos)
    timelines.append(serving_timeline(log))
    summary.append(dict(policy=p.label, handovers=len(hos), rlf=len(rlfs), pingpong=pp,
                        attaches=len(attaches), ho_events=hos, rlf_events=rlfs, log=log))
    fn = os.path.join(OUT, "log_" + p.label.split("(")[0].strip().replace(" ", "_").replace("+", "plus")
                      .replace("__", "_") + ".txt")
    fn = os.path.join(OUT, f"log_policy{POLICIES.index(p) + 1}.txt")
    with open(fn, "w") as f:
        f.write(f"Policy: {p.label}\n")
        f.write(f"{'t':>4} {'time(s)':>7} {'x(m)':>6}  {'from':<16}-> {'to':<16} {'cell':<4} trigger\n")
        for (t, a, b, c, trig) in log:
            f.write(f"{t:>4} {t * DT_S:>7.1f} {x[t]:>6.0f}  {a:<16}-> {b:<16} {str(c or '-'):<4} {trig}\n")
        f.write(f"\nHandovers completed: {len(hos)}   RLF: {len(rlfs)}   Ping-pongs: {pp}\n")
        if pp_list:
            f.write("Ping-pong handovers (t, from, to): " + ", ".join(map(str, pp_list)) + "\n")

print(f"\n{'Policy':<38}| {'Handovers':>9} | {'RLF':>3} | {'Ping-pong':>9} | {'Attaches':>8}")
print("-" * 80)
for s in summary:
    print(f"{s['policy']:<38}| {s['handovers']:>9d} | {s['rlf']:>3d} | {s['pingpong']:>9d} | {s['attaches']:>8d}")

with open(os.path.join(OUT, "table_policy_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy", "completed_handovers", "RLF", "ping_pong", "RACH_attaches"])
    for s in summary:
        w.writerow([s["policy"], s["handovers"], s["rlf"], s["pingpong"], s["attaches"]])

# log excerpt for the report: first 14 lines of the hysteresis+TTT run
print("\nLog excerpt (hysteresis 3 dB + TTT 5):")
for (t, a, b, c, trig) in summary[1]["log"][:14]:
    print(f"  t={t:>3} {a:<16}-> {b:<16} {str(c or '-'):<3} {trig}")

# ============================================================ FIGURES
# (1) signals + serving timeline per policy
fig, axes = plt.subplots(len(POLICIES) + 1, 1, figsize=(12, 12), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1.2]})
cols = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
for c in sig:
    axes[0].plot(x, sig[c], color=cols[c], lw=0.9, label=f"cell {c}")
axes[0].axhline(RLF_THRESH, color="red", ls="--", lw=0.8, label="RLF threshold")
axes[0].set_ylabel("RSRP (dBm)"); axes[0].set_ylim(-125, -60)
axes[0].set_title("Scripted three-cell scenario: device moves A -> B -> C")
axes[0].legend(ncol=4, loc="upper center")
for ax, s, tl in zip(axes[1:], summary, timelines):
    for k in range(T):
        c = tl[k]
        ax.axvspan(x[k], x[k] + SPEED_MPS * DT_S, color=cols.get(c, "lightgray"), lw=0)
    for (t, a, b) in s["ho_events"]:
        ax.axvline(x[t], color="k", lw=0.8)
    for (t, c, st) in s["rlf_events"]:
        ax.axvline(x[t], color="red", lw=2)
    ax.set_yticks([]); ax.set_ylabel("serving", rotation=0, labelpad=28)
    ax.set_title(f"{s['policy']}: {s['handovers']} HO, {s['rlf']} RLF, {s['pingpong']} ping-pong "
                 f"(black = handover done, red = RLF, grey = IDLE/RACH)", fontsize=9)
axes[-1].set_xlabel("device position x (m)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_scenario_and_serving_timelines.png"), dpi=150)
plt.close(fig)

# (2) state diagram
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 12.2); ax.set_ylim(0, 8); ax.axis("off")


def box(x0, y0, w, h, text, fc, fs=9):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05", fc=fc, ec="k"))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs)


def path(pts, label=None, lx=None, ly=None, color="k", fs=7.5, ha="center"):
    """Polyline with an arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.3, shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx, ly, label, fontsize=fs, ha=ha, va="center", color=color,
                bbox=dict(fc="white", ec="none", pad=0.8))


box(0.3, 3.3, 1.9, 1.2, "IDLE\n(no connection)", "#e8e8e8")
ax.add_patch(FancyBboxPatch((4.0, 2.0), 3.4, 4.4, boxstyle="round,pad=0.05", fc="#fff3d6", ec="k"))
ax.text(5.7, 6.15, "RANDOM ACCESS (RACH)", ha="center", fontsize=9, weight="bold")
labels = ["1. PREAMBLE (Msg1)", "2. RAR + timing advance (Msg2)",
          "3. CONNECTION REQUEST (Msg3)", "4. CONTENTION RESOLUTION (Msg4)"]
ys = [5.0, 4.1, 3.2, 2.3]
for yy, lab in zip(ys, labels):
    box(4.3, yy, 2.8, 0.7, lab, "white", fs=8.5)
for yy in ys[:-1]:
    path([(5.7, yy), (5.7, yy - 0.2)])
box(9.2, 3.3, 2.2, 1.2, "CONNECTED\n(serving cell)", "#d6ecd6")
box(9.2, 0.6, 2.2, 1.2, "HANDOVER IN\nPROGRESS", "#d6e4f5")

# IDLE <-> RACH
path([(2.2, 4.2), (4.0, 4.2)], "attach / data to send /\nre-establishment", 3.1, 4.65)
path([(4.0, 3.6), (2.2, 3.6)], "max attempts\nreached", 3.1, 3.15)
# RACH retry loop on step 1
path([(7.1, 5.55), (7.8, 5.55), (7.8, 5.15), (7.1, 5.15)],
     "preamble not heard / contention\nlost: back off, retry (< 4 tries)", 7.95, 5.95, ha="left")
# RACH -> CONNECTED
path([(7.1, 2.65), (9.2, 3.6)], "Msg4 ok", 8.1, 2.85)
# CONNECTED <-> HANDOVER
path([(9.8, 3.3), (9.8, 1.8)], "policy fires\n(naive or\nhyst + TTT)", 9.0, 2.55)
path([(10.8, 1.8), (10.8, 3.3)], "target RACH ok,\nHO complete", 11.75, 2.55)
# RLF paths (red)
path([(10.3, 4.5), (10.3, 7.4), (1.25, 7.4), (1.25, 4.5)],
     "RLF: serving RSRP < -110 dBm for 3 samples (too-late handover)", 5.8, 7.4, color="red")
path([(9.2, 0.9), (1.25, 0.9), (1.25, 3.3)],
     "RLF mid-handover (or target too weak at completion)", 5.2, 0.9, color="red")
ax.set_title("Connection / handover finite-state machine (red = Radio Link Failure path)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_state_diagram.png"), dpi=150)
plt.close(fig)

# (3) transition table
TRANSITIONS = [
    ("IDLE", "RACH_PREAMBLE", "Data to send / re-establishment; strongest cell >= -105 dBm (after 5-sample wait following RLF)"),
    ("RACH_PREAMBLE", "RACH_RAR", "Preamble detected (cell RSRP >= -110 dBm)"),
    ("RACH_PREAMBLE", "IDLE", "4 attempts without detection"),
    ("RACH_RAR", "RACH_CONNREQ", "RAR with timing advance received"),
    ("RACH_CONNREQ", "RACH_CONTENTION", "Msg3 (RRC connection request) sent"),
    ("RACH_CONTENTION", "CONNECTED", "Contention resolved (Msg4)"),
    ("RACH_CONTENTION", "RACH_PREAMBLE", "Contention lost -> back off and retry"),
    ("CONNECTED", "HANDOVER", "Policy fires (naive, or hysteresis margin held for TTT samples)"),
    ("HANDOVER", "CONNECTED", "Execution done (3 samples) and target RSRP >= -110 dBm"),
    ("CONNECTED", "IDLE", "RLF: serving RSRP < -110 dBm for 3 consecutive samples"),
    ("HANDOVER", "IDLE", "RLF on source during execution, or target too weak"),
]
with open(os.path.join(OUT, "table_state_transitions.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["from", "to", "trigger"]); w.writerows(TRANSITIONS)
print(f"\nOutputs written to {OUT}")
