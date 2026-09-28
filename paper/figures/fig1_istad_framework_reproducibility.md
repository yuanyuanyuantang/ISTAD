# Figure 1 statistics and reproducibility report

## Figure class and statistical scope

Figure 1 is a method schematic, not a quantitative result figure. It contains no observed
samples, fitted effect sizes, uncertainty intervals, hypothesis tests, or significance
claims. Consequently, sample size, error-bar definition, statistical test, multiple-testing
correction, and significance thresholds are **not applicable**. No data points were removed,
filtered, or downsampled to create the figure.

## Scientific provenance

The schematic was traced to the frozen method description and implementation:

- Causal ridge VAR innovations, robust evidence, ECDF calibration, stability gate, and
  rank-safe fusion: `../ISTAD_KBS_draft.md`, `../../code/ISTAD/utils/innovation.py`, and
  `../../code/ISTAD/exp/exp_anomaly_detection.py`.
- HGAT-Lite projection, sparse dynamic incidence, and node–edge–node propagation:
  `../../code/ISTAD/models/istad_layers/hypergraph_attention.py`.
- B-spline basis used by the optional bounded residual mapping:
  `../../code/ISTAD/models/istad_layers/spline_ops.py`.

Panel a depicts the main ISTAD method: causal VAR innovations, HGAT-Lite relation learning,
and rank-safe fusion. Panel b expands only the HGAT-Lite mechanism. Panel c is explicitly an
ablation extension: the bounded residual BR-KAN mapping replaces the linear relation
projection but does not replace the main method.

## Reproduction

From the repository root, run:

```bash
python paper/figures/draw_istad_framework.py
```

Required runtime packages are Matplotlib and its standard numerical dependencies. The script
sets the font hierarchy, accessible palette, exact canvas size, and export parameters before
drawing. It writes all three formats beside the script:

- `fig1_istad_framework.pdf`
- `fig1_istad_framework.svg`
- `fig1_istad_framework.png`

The diagram uses only deterministic vector primitives and therefore requires no random seed.
The PNG was rendered with Matplotlib 3.10.8 at 300 dpi in the recorded environment. The
provided PDF is the preferred submission asset; the SVG is the preferred editable source.

