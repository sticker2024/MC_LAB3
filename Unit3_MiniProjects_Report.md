# Mobile Communications – Unit 3 Mini Projects: Results Report

All code is in `mp3_1.py`, `mp3_2.py`, `mp3_3.py` (Python 3, NumPy, Matplotlib only). Each script writes its figures and tables to `out_3_1/`, `out_3_2/`, `out_3_3/`. Random seeds are fixed, so every number below is reproducible.

---

## Mini Project 3.1 – Frequency reuse and handover decision logic

**Files:** `out_3_1/fig1_reuse_layouts_N1_3_4_7.png` (four layouts), `fig1b_…N9_12_extension.png` (Lab Task 1), `fig2_naive_handover.png`, `fig3_three_configs_stacked.png`, `fig4_param_sweep_MRO.png` (Lab Task 3), `table_Q_vs_N.csv`, `table_handover_counts.csv`.

**Q versus N** (measured nearest same-group distance matches √(3N)·R in every layout; the script asserts this)

| N | i, j | Q = √(3N) | Groups seen |
|---|------|-----------|-------------|
| 1 | 1, 0 | 1.73 | 1 |
| 3 | 1, 1 | 3.00 | 3 |
| 4 | 2, 0 | 3.46 | 4 |
| 7 | 2, 1 | 4.58 | 7 |
| 9 (ext.) | 3, 0 | 5.20 | 9 |
| 12 (ext.) | 2, 2 | 6.00 | 12 |

**Handover comparison** (300 samples, seed 7, noise σ = 3 dB, trend ±0.12 dB/sample, nominal crossover at sample 150)

| Configuration | Handovers | Min serving RSRP | Samples with serving < −100 dBm |
|---|---|---|---|
| Naive (0 dB, no TTT) | 11 | −102.7 dBm | 4 |
| Hysteresis only (3 dB) | 5 | −103.4 dBm | 5 |
| Hysteresis 3 dB + TTT 5 | 1 (at sample 154) | −105.9 dBm | 25 |

Mean over 40 other random seeds: 18.6, 6.75 and 1.00 handovers respectively, with 2.1, 2.8 and 9.0 low-signal samples, so the ordering is not a fluke of one seed.

**Interpretation (≈230 words)**

Q = √(3N) rises with the reuse factor: 1.73 for N = 1, 3.00 for N = 3, 3.46 for N = 4 and 4.58 for N = 7, and the generated layouts confirm that the nearest same-numbered hexagons sit exactly √(3N)·R apart. Small N packs co-channel cells close together, which gives high capacity but high co-channel interference; N = 7 separates them most, at the cost of reusing each frequency in fewer cells. The extension to N = 9 and N = 12 (Q = 5.20 and 6.00) continues the same trend with diminishing capacity. In the handover simulation the naive trigger produced 11 handovers, nearly all between samples 105 and 145 where the two RSRP curves overlap. This is the ping-pong effect: noise keeps flipping which cell is momentarily stronger. A 3 dB hysteresis margin cut the count to 5, and adding a 5-sample time-to-trigger reduced it to a single handover at sample 154. The price is the too-late-handover risk: with hysteresis plus TTT the serving RSRP stayed below −100 dBm for 25 samples (minimum −105.9 dBm), against 4 samples for the naive trigger, because the device waited longer on a fading serving cell. The parameter sweep shows this trade-off getting worse for aggressive settings, for example 8 dB with 15 samples gave about 37 low-signal samples, which is how over-tuning leads toward radio link failure.

---

## Mini Project 3.2 – Capacity and frequency-reuse planning

**Inputs:** 25 MHz total spectrum, 200 kHz channels (125 channels), 1000 km² area, R = 2 km (hexagon area 10.39 km², 96 cells). Whole channels only, so channels per cell is ⌊125 / N⌋.

**Files:** `out_3_2/table_summary.csv`, `fig1_capacity_vs_N_and_Q.png`, `fig2_capacity_levers.png`, `fig3_sectoring_sir.png`.

| N | Q | Channels/cell | Cells | System capacity (channels) |
|---|---|---|---|---|
| 1 | 1.73 | 125 | 96 | 12,000 |
| 3 | 3.00 | 41 | 96 | 3,936 |
| 4 | 3.46 | 31 | 96 | 2,976 |
| 7 | 4.58 | 17 | 96 | 1,632 |
| 9 | 5.20 | 13 | 96 | 1,248 |
| 12 | 6.00 | 10 | 96 | 960 |

**Cell splitting (N = 7):** R = 1 km → 384 cells, 6,528 channels (4.00×); R = 0.5 km → 1,539 cells, 26,163 channels (16.03×, the extra 0.03 is rounding of whole cells). Q is unchanged.

**Sectoring illustration** (first-tier approximation SIR ≈ Q^γ / i₀, γ = 4): omni cells see 6 first-tier interferers, a 120° sector sees about 2, a gain of 10·log₁₀(3) = 4.8 dB at every N. Omni N = 7 gives 18.7 dB; three-sector N = 4 gives 18.6 dB, essentially the same quality, and capacity rises from 1,632 to 2,976 channels (1.82×, before trunking-efficiency loss).

**Written analysis (≈400 words)**

The table shows the basic trade-off. With a fixed 125 channels, capacity is inversely proportional to N: moving from N = 7 to N = 3 raises capacity from 1,632 to 3,936 channels, and N = 1 reaches 12,000. The price is co-channel protection, measured by Q = √(3N), which falls from 4.58 at N = 7 to 3.00 at N = 3 and 1.73 at N = 1. With a path-loss exponent of 4, omni N = 1 gives roughly 1.8 dB of signal-to-interference ratio, which is unusable for narrowband channels, while N = 3 gives about 11 dB and N = 7 about 19 dB. Choosing N therefore means choosing the smallest value whose SIR the radio technology can tolerate.

For dense urban deployment demand, not coverage, is the limit, so the operator wants the smallest workable N, typically 3 or 4 with sectored sites, combined with small cells. For rural deployment the opposite holds: traffic is light, cells are large, and spectrum is not the bottleneck. A larger N such as 7 buys robust SIR and range at no real cost because each cell has enough channels for few users.

Once N is already small, further reduction is not available: dropping from N = 3 to N = 1 destroys SIR. Splitting and sectoring add capacity without that penalty. Cell splitting multiplies capacity by the square of the radius ratio (4× for halving R, about 16× for quartering it) and leaves Q, and therefore SIR, unchanged, because Q depends only on N. Sectoring works through the directional-beam argument of Section 3.1: a 120° antenna illuminates only a third of the surroundings and hears roughly two of the six first-tier co-channel cells, a 4.8 dB SIR gain. That gain lets N = 4 with three sectors match the interference quality of omni N = 7 while delivering about 1.8 times the capacity.

Both levers have costs. Splitting needs new sites, backhaul and re-planning, and smaller cells mean more handovers, which is why the handover tuning in Project 3.1 matters more as cells shrink. Sectoring divides each site's channel pool across three sectors, which reduces trunking efficiency, and adds sector-to-sector handovers. The model here is deliberately simple: it counts channels, ignores traffic (Erlang) dimensioning, uses a first-tier-only SIR approximation and assumes perfect hexagons, so the absolute numbers are indicative and the trends are the point.

---

## Mini Project 3.3 – Connection and handover as a finite-state machine

**Files:** `out_3_3/fig2_state_diagram.png`, `table_state_transitions.csv`, `fig1_scenario_and_serving_timelines.png`, `log_policy1.txt` (naive), `log_policy2.txt` (hysteresis 3 dB + TTT 5), `log_policy3.txt` (hysteresis 8 dB + TTT 60), `table_policy_summary.csv`.

**Design in brief:** states IDLE, RACH (four sub-states: preamble, RAR with timing advance, connection request, contention resolution), CONNECTED and HANDOVER. Policies are interchangeable functions `policy(serving, rsrp, mem) → target or None`. RLF is declared when serving RSRP stays below −110 dBm for 3 consecutive samples, in CONNECTED or HANDOVER, and returns the device to IDLE; it then re-attaches through RACH. A scripted contention loss on the first attach shows the RACH retry path. Scenario: cells A, B, C at 0, 1000, 2000 m, device at 20 m/s, 0.2 s samples (487 samples), path loss slope 33 dB/decade, AR(1) shadowing σ = 3 dB plus 1.5 dB measurement noise.

| Policy | Handovers | RLF | Ping-pong | RACH attaches |
|---|---|---|---|---|
| Naive | 16 | 0 | 13 | 1 |
| Hysteresis 3 dB + TTT 5 | 2 | 0 | 0 | 1 |
| Hysteresis 8 dB + TTT 60 | 0 | 2 | 0 | 3 |

Log excerpt (hysteresis 3 dB + TTT 5):

```
t=  0 IDLE            -> RACH_PREAMBLE    A  cell selection: A at -67.6 dBm; send preamble (Msg1)
t=  4 RACH_CONTENTION -> RACH_PREAMBLE    A  contention lost -> back off, new preamble
t=  8 RACH_CONTENTION -> CONNECTED        A  contention resolved (Msg4) -> RRC connected
t=146 CONNECTED       -> HANDOVER         A  policy fired: B -100.1 vs A -106.8 dBm
t=149 HANDOVER        -> CONNECTED        B  handover complete: A -> B
```

**Comparison report (≈310 words)**

The naive policy handed over whenever a neighbour was momentarily stronger. In the A/B overlap (x ≈ 450–650 m) and the B/C overlap (x ≈ 1340–1700 m) it completed 16 handovers, 13 of which were ping-pongs, meaning a return to a cell left within 8 seconds. It never lost the link, because it always moved early, but it generated a large signalling load for no benefit.

Hysteresis of 3 dB combined with a 5-sample time-to-trigger removed all ping-pong. The margin ensures that a neighbour must be clearly better, and the TTT ensures that this holds for 1 second rather than for a single noisy sample. It produced exactly the two intended handovers, A to B at x = 634 m and B to C at x = 1578 m. These are 134 m and 78 m beyond the geometric midpoints, and the serving RSRP at trigger was −106.8 and −108.7 dBm, close to the −110 dBm RLF threshold but still above it. That shows the cost of this filtering: handovers become later.

The over-tuned setting (8 dB, 60 samples = 12 s) shows that cost taken too far. The required advantage never persisted for 60 consecutive samples before the serving cell fell below −110 dBm, so no handover happened at all and the device suffered two radio link failures, at x = 814 m and x = 1834 m. Each caused about 1.8 seconds of outage while the device waited in IDLE and re-attached through RACH, which is the too-late handover failure.

In summary, a larger margin and a longer TTT reduce ping-pong but delay handover, and both must be chosen against the fading and the device speed. Here 3 dB with 5 samples sat in the working range; this is a single seeded scenario, so the precise counts would change with another seed even though the ordering is stable.
