# Data and weights

This repository does not redistribute MM-Fi recordings or X-Fi model weights.

## Official sources

- MM-Fi project page: <https://ntu-aiot-lab.github.io/mm-fi>
- MM-Fi toolbox: <https://github.com/ybhbingo/MMFi_dataset>
- X-Fi source: <https://github.com/NTUMARS/X-Fi>

The URLs recorded during the experiment are also listed in
[`configs/mmfi_official_sources.json`](configs/mmfi_official_sources.json).
Always check the current terms on the official pages before downloading or
redistributing any asset. The recorded MM-Fi dataset license is CC BY-NC 4.0.

## Expected local layout

After obtaining the assets from their official sources, arrange the relevant
parts locally as follows:

```text
xfi-multimodal-reliability/
├── assets/
│   └── xfi_weights/
│       └── MMFi_HAR/
│           └── <released-full-checkpoint>.pt
├── data/
│   └── aligned_points/
│       ├── E01/S01/A01/lidar/frame001.bin
│       ├── E01/S01/A01/mmwave/frame001.bin
│       └── ...
└── third_party/
    ├── X-Fi/
    │   └── MMFi_HAR/
    └── MMFi_dataset/
```

The point loader expects synchronized frame IDs and rejects recordings whose
LiDAR and mmWave frame sets differ. LiDAR rows contain three float64 values;
mmWave rows contain five float64 values.

## Why these files are excluded

- They are large and unsuitable for ordinary Git history.
- Dataset and checkpoint redistribution may be governed by separate licenses.
- Raw assets are not required to inspect the protocol, code, tests, aggregate
  tables, or selected figures in this repository.

The `.gitignore` intentionally blocks common data, checkpoint, environment, and
raw-inference paths to reduce the risk of accidental publication.

