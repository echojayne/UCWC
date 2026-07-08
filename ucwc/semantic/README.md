# Semantic

This module owns the first executable semantic-communication link used to build
initial experience tables and later test real semantic codecs.

## Components

- `models.py`: the primary codec path. It loads a variable-depth CLIP
  `ViT-B/16` encoder and RIDAS' `decoder_cifar100.pth` as a `512 -> 100`
  receiver-side decoder.
- `clip_vit.py`: local CLIP ViT visual source copied/adapted from
  `/home/users/dky/RKDSC/CLIP-main/clip/model.py`, modified so
  `encoder_depth` can stop after an intermediate transformer block.
- `quantization.py`: uniform feature quantization and bit packing.
- `ldpc.py`: a real binary LDPC code with sparse parity-check matrix,
  systematic encoding, and iterative sum-product decoding.
- `qam.py`: Gray-coded square QAM with exact log-likelihood soft demodulation.
- `channels.py`: AWGN and Rayleigh discrete-time channel simulations.
- `link.py`: bitstream and semantic-feature end-to-end transmission.
- `evaluation.py`: grid evaluation for semantic score, BER, LDPC convergence,
  and encode/decode latency.
- `storage.py`: SQLite/CSV persistence for result rows.
- `weights.py`: RIDAS decoder/classifier weight helpers.

The LDPC/QAM/channel path is not a throughput proxy. It encodes feature bits,
modulates coded bits, simulates noisy complex symbols, demodulates soft LLRs,
and decodes LDPC blocks before reconstructing semantic features.

RIDAS' `clip/` tree provides the receiver-side decoder weights used here. The
project stores those small weights under
`data/semantic_catalogs/weights/ridas/`; the CLIP ViT source is local to this
module so depth-controlled encoder evaluation is reproducible.

## Entry Points

```bash
python scripts/run_semantic_link_smoke.py --output-dir outputs/semantic_link_smoke
```

For a depth sweep:

```bash
python scripts/run_semantic_link_smoke.py \
  --encoder-depths 3 6 9 12 \
  --quantization-bits 4 \
  --snr-db 20 \
  --output-dir outputs/semantic_depth_sweep
```

The smoke runner writes:

- `semantic_results.sqlite`
- `semantic_experience.csv`
