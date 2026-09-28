# V4-HG efficiency benchmark

> Device: NVIDIA RTX A6000; PyTorch 2.6.0+cu124; 30 warm-up + 100 measured passes.
> Scope: model forward with dynamic incidence output; excludes data loading and VAR fitting.

| Dataset | Parameters | HGAT parameters | Checkpoint | Batch | ms/batch | us/window | Peak allocated | Epoch train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXA | 4,771 | 274 | 25.2 KiB | 128 | 3.838±0.083 | 29.99 | 62.7 MiB | 0.840±0.039 s |
| PSM | 7,303 | 346 | 35.2 KiB | 128 | 3.150±0.360 | 24.61 | 65.9 MiB | 65.200±1.099 s |
| SMD | 14,522 | 506 | 63.3 KiB | 128 | 7.027±0.020 | 54.90 | 198.1 MiB | 4.791±0.041 s |
| SWAT | 44,867 | 618 | 181.9 KiB | 64 | 4.533±0.014 | 70.83 | 141.9 MiB | 3.964±0.037 s |

The epoch times are observational values from the three completed training runs;
they are not a controlled cross-model speed comparison. PSM is much longer than
the other training loaders, so epoch time must not be compared across datasets.
