# CIFAR10 ViT-B/16 Full-Grid Semantic Link Review

## Completeness

- Rows: 3024 / expected 3024.
- Dimensions: 12 layers, 3 quantization settings, 3 LDPC rates, 4 QAM orders, 7 SNR points.
- Batch size: [10000]; feature_dim: [512].
- NaN cells: 0; duplicate config keys: 0.
- The `config_id` prefix was corrected from `cifar10_vitb16_1000_` to `cifar10_vitb16_10000_` in both CSV and SQLite metadata.

## Metric Sanity

- `semantic_score` range: 0.0058 to 0.9886; mean 0.2934.
- `bit_error_rate` range: 0 to 0.4744; mean 0.1978.
- `block_error_rate` range: 0 to 1.0000; mean 0.6753.
- `original_accuracy` range: 0.3169 to 0.9434; this matches the layer-head trend from shallow to deep ViT layers.
- Adjacent-SNR monotonic check across 432 layer/q/rate/QAM groups: semantic-score decreases occurred in 98 groups; BER increases occurred in 0 groups. Small local deviations are possible because each configuration uses its own random channel/noise seed, but the aggregate SNR trend is correct.

## Normality Assessment

The result is internally consistent and broadly normal for this link model. Low SNR and high-order QAM produce saturated LDPC block errors and low semantic recovery. High SNR, especially q=8 and low-order QAM, recovers semantic features close to the quantized clean-link ceiling. The recovered classification accuracy follows the communication quality and the direct layer-head accuracy ceiling, rather than indicating a dataset-label problem.

## Generated Artifacts

- `summary_by_snr_qbit.csv`
- `summary_by_mode_snr.csv`
- `summary_by_layer_snr_qbit.csv`
- `typical_cases.csv`
- `figures/snr_quantization_trends.png`
- `figures/ber_snr_by_qam.png`
- `figures/layer12_q8_mode_accuracy_heatmap.png`
- `figures/typical_cases_dashboard.png`
