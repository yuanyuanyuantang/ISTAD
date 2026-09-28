# Figure 1 quality-assurance record

## Deliverable

- Figure: `fig1_istad_framework`
- Intended use: Knowledge-Based Systems manuscript, double-column method figure
- Canvas: 183.0 mm × 143.8 mm
- Exports: editable SVG, vector PDF, and 300 dpi PNG
- Generator: `draw_istad_framework.py`

## Automated source checks

The Academic Figure Skill checker completed all four modules without failure:

| Check | Result | Evidence |
|---|---:|---|
| Accessible palette | PASS | No default, rainbow, or red–green-only encoding |
| Font family and size | PASS | Sans-serif family declared; all explicit sizes are at least 5 pt |
| Dimensions | PASS | 183.0 mm double-column width; height below 247 mm |
| Export | PASS | PDF and SVG vector outputs; PNG at 300 dpi; PDF TrueType embedding enabled |

## Visual inspection

| Item | Result | Notes |
|---|---:|---|
| Logical flow | PASS | Panel a follows input → two evidence paths → gate → fusion → score |
| Main/extension boundary | PASS | BR-KAN is confined to the dashed orange ablation panel |
| Clipping and overlap | PASS | Text, equations, arrows, and boxes remain within their panel bounds |
| Typography consistency | PASS | One sans-serif hierarchy is used across all three panels |
| Color independence | PASS | Branch names, direct labels, and solid/dashed strokes preserve meaning in grayscale |
| Panel balance | PASS | Non-background densities: a 33.94%, b 10.69%, c 19.51%; all within the 1.5–50% screening interval |
| Editable text | PASS | SVG retains text elements; PDF fonts are embedded |

## Output verification

- PDF: one page, 518.74 × 407.62 pt (183.0 × 143.8 mm).
- PNG: 2161 × 1698 pixels at approximately 300 dpi (182.97 × 143.76 mm from raster metadata).
- A grayscale preview was inspected separately; all paths and labels remain distinguishable.
- Final verdict: **READY for manuscript integration**.

