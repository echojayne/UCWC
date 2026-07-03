This is a tiny pathloss-grid fixture for testing the RadioMapSeer-style adapter.

Real RadioMapSeer data should be stored outside source control and referenced
from `configs/*/simulation.yaml` through `channel.radio_map.dataset_root`.
The production adapter supports CSV, NPY/NPZ, and grayscale PNG/TIFF/JPEG maps.
