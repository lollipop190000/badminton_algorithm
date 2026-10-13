# Parity-aware Markov Q, C, K, and V

## State and parity

The fixed reference player is canonical `player_A`. This assignment is based on
player-name ordering, not on who won the match. The court-pattern state remains

\[
i=(z_{t-3},z_{t-2},z_{t-1}),
\]

but it is augmented with

\[
r=0:\ A-B-A,\qquad r=1:\ B-A-B.
\]

The labels describe the hitters of the three completed shots in the state. The
current action is therefore hit by B under parity 0 and by A under parity 1.
Every valid transition flips parity:

\[
(i,r)\longrightarrow(j,1-r).
\]

There are 4,096 court patterns and 8,192 augmented states. A rally containing a
nonalternating hitter annotation is excluded in full, because its parity cannot
be represented by the two off-diagonal blocks.

All terminal probabilities use the fixed reference-player perspective:

\[
w_{i,r}^{(s)}=P(\text{A wins immediately},s\mid i,r),
\]

\[
l_{i,r}^{(s)}=P(\text{A loses immediately},s\mid i,r).
\]

## Movement and energy cost

For a shot at row \(t\), movement is taken from row \(t+1\), so it measures the
interval caused by the current shot rather than the movement leading into that
shot. `self` is always player A and `opp` is always player B.

For each parity-labelled transition:

\[
C_{ij,r,p}^{(s)}=19.985\left[
\frac{n_{ijsr}}{n_{ijsr}+\tau}\bar d_{ij,r,p}^{(s)}+
\frac{\tau}{n_{ijsr}+\tau}\hat d_{ij,r,p}^{(s)}
\right].
\]

The prior distance \(\hat d\) is the train-only parity/shot/next-zone mean,
backed off to parity/shot and then the parity-wide mean. A common \(\tau\) is
selected by validation RMSE. The current result selects `tau=1.0`.

The coefficient is the arithmetic mean of the reported side-step and
running-step net energy costs, 19.59 and 20.38 J kg^-1 m^-1
([PMID 40716493](https://pubmed.ncbi.nlm.nih.gov/40716493/)). The tracking data
contain straight-line displacement between annotated stroke positions, not the
full footwork trajectory. Consequently, exported `C` is an energy-equivalent
movement proxy, not directly measured metabolic energy.

## Kernel and value

For each supplied real \((p,q)\) pair,

\[
K_{ij,r}^{(s)}(p,q)=Q_{ij,r}^{(s)}
e^{-pC_{ij,r,\mathrm{self}}^{(s)}-qC_{ij,r,\mathrm{opp}}^{(s)}}.
\]

The aggregate matrix contains only off-diagonal parity blocks:

\[
K=\begin{pmatrix}0&K_{01}\\K_{10}&0\end{pmatrix}.
\]

Values are solved without constructing a dense 8,192-by-8,192 matrix, using
the structurally valid 16 outgoing zone edges per state:

\[
V=w+KV.
\]

At `p=q=0`, \(K=Q\), so the result is the ordinary fixed-reference-player EPV.
Positive `p` or `q` values currently produce energy-discounted sensitivity
values. They are not yet inverse-Laplace stamina probabilities because
\(\widetilde w(F_{self},F_{opp})\) and a numerical inverse transform have not
been specified.

## Run

```powershell
.\.venv-gru\Scripts\python.exe -m src.estimate_parity_markov `
  --split-root data\splits `
  --output-root outputs\markov_parity
```

The default sensitivity grid is
`p,q in {0, 0.005, 0.01, 0.02}` in inverse J/kg. Alternative nonnegative grids
can be supplied with `--p-values` and `--q-values`; both grids must include zero
so the baseline EPV is retained.

## Outputs

| File | Meaning |
|---|---|
| `state_parity_index.parquet` | Mapping for all 8,192 `(i,r)` states |
| `q_shot_parity.parquet` | Raw and smoothed \(Q_{ij,r}^{(s)}\) |
| `terminal_probabilities_parity.parquet` | \(w_{i,r}^{(s)}\) and \(l_{i,r}^{(s)}\) |
| `transition_costs.parquet` | Counts, observed/prior/shrunk distances, and both-player costs |
| `kernel_components.parquet` | Joined \(Q^{(s)}\) and \(C^{(s)}\) inputs for arbitrary \(p,q\) |
| `kernel_aggregate.parquet` | Aggregate \(K_{ij,r}(p,q)\) for every requested scenario |
| `state_values.parquet` | Baseline EPV and positive-parameter sensitivity values |
| `diagnostics.json` | Definitions, selections, held-out metrics, and numerical invariants |

The previous non-parity outputs under `outputs/markov_q` remain available for
audit history but must not be used in the parity-aware EPV model.
