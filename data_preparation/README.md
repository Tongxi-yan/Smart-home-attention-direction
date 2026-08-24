# Data preparation

This module keeps the available original data, the final model inputs and the code
that converts the beginning of the pipeline into the final format.

```text
data_preparation/
├── raw_data/       23 available raw folders from a 25-recording study
├── final_data/     25 primary final datasets
├── merge_data.py   skeleton + device position + ELAN label alignment
├── smooth_data.py  interpolation, Gaussian smoothing and validation
├── schema.py       canonical joints, graph nodes, columns and labels
└── manifest.json   machine-readable counts and sizes
```

The study collected 25 recordings. `Data18` and `Data19` were not present in the
supplied raw workspace, so 23 raw folders are available. All 25 final processed
datasets are retained. Missing raw sessions are not fabricated.

## ELAN labels

Synchronized RGB videos were manually annotated in ELAN as `Off`, `On Device 1`
(lamp), or `On Device 2` (speaker). Intervals begin at the earliest visible attention
cue and end after attention moves away. They were aligned to the 30 fps recordings and
stored as frame-level `label_device1`, `label_device2` and `label_3` columns.

## Deliberately excluded

- intermediate merge and smoothing copies;
- preprocessing validation copies;
- generated window caches;
- train/validation/test dataset copies;
- repeated experiment directories;
- separate ELAN project files.

Feature construction continues in `../source_code/feature_generation/`. Window and
segment partitions are generated at runtime by `../source_code/data_splitting/`.

Raw RGB videos may identify participants or environments. Keep the repository private
unless participant consent and institutional policy allow public release.
