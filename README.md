# PILOT: Prototype-Informed Flow Matching for One-Step Sequential Recommendation

This repository provides the implementation and processed datasets for **PILOT: Prototype-Informed Flow Matching for One-Step Sequential Recommendation**, accepted by **WISE 2026**.

PILOT is a sequential recommendation framework that combines a **collaborative prototype prior** with **flow matching** to generate the next-item representation through **one-step inference**. The implementation contains the complete training, validation, model-selection, and full-catalog evaluation pipeline for Amazon Beauty, Steam, and Yelp.

---

## Repository Structure

```text
PILOT/
├── README.md
├── LICENSE
├── datasets/
│   ├── readme.md
│   └── data/
│       ├── amazon_beauty/
│       │   └── dataset.pkl
│       ├── steam/
│       │   └── dataset.pkl
│       └── yelp/
│           └── dataset.pkl
└── src/
    ├── main.py
    ├── model.py
    ├── pilot.py
    ├── trainer.py
    └── utils.py
```

### Main Components

| Path | Description |
| --- | --- |
| `src/main.py` | Command-line entry point. It parses hyperparameters, fixes the random seed, loads the processed dataset, builds PILOT, runs training and evaluation, and optionally writes final metrics to a JSONL file. |
| `src/model.py` | PILOT wrapper. It defines item embeddings, history encoding, the collaborative prototype prior, the scoring function, and the weighted training objective. |
| `src/pilot.py` | Generative core. It implements the Transformer blocks, FiLM-conditioned flow network, time embedding, rectified-path sampling, and one-step inference at `t=0`. |
| `src/trainer.py` | Training loop, validation-based model selection, early stopping, in-memory best-state restoration, HR/NDCG computation, and full-catalog evaluation. |
| `src/utils.py` | PyTorch datasets and dataloaders for the leave-one-out sequential recommendation protocol. |
| `datasets/data/` | Processed Amazon Beauty, Steam, and Yelp datasets used directly by the training and evaluation scripts. |
| `datasets/readme.md` | Note on the preprocessing protocol and its relation to ICLRec and DuoRec. |

---

## Environment and Dependencies

The implementation uses common Python packages:

- Python 3.x
- PyTorch
- NumPy
- `tqdm` (optional; the code falls back to a plain iterator if it is unavailable)

A minimal environment can be installed with:

```bash
pip install torch numpy tqdm
```

For GPU training, install a PyTorch build compatible with the local CUDA environment. A modern GPU is recommended for Steam and Yelp because evaluation performs full-catalog scoring over large item sets.

If `--device cuda` is requested but CUDA is unavailable, the program automatically falls back to CPU execution.

---

## Data Format

PILOT includes processed versions of three public sequential recommendation datasets:

- **Amazon Beauty**
- **Steam**
- **Yelp**

The corresponding files are:

```text
datasets/data/amazon_beauty/dataset.pkl
datasets/data/steam/dataset.pkl
datasets/data/yelp/dataset.pkl
```

Each `dataset.pkl` is loaded directly by `src/main.py` and contains:

- `train`: user-to-history dictionary for training;
- `val`: held-out validation item dictionary;
- `test`: held-out test item dictionary;
- `umap`: user ID mapping;
- `smap`: item ID mapping.

The model sets the item vocabulary size as:

```python
len(data_raw['smap']) + 1
```

where item index `0` is reserved for sequence padding.

### Sequential Recommendation Protocol

The dataloaders implement a standard leave-one-out setting.

For a training sequence

```text
[i1, i2, i3, i4]
```

prefix/next-item pairs are generated conceptually as:

```text
[i1]         -> i2
[i1, i2]     -> i3
[i1, i2, i3] -> i4
```

Sequences are left-padded with `0` or truncated to `--max_len`.

For validation, the training sequence is used as the historical sequence and the held-out validation item is used as the target.

For testing, the validation item is appended to the training history before predicting the held-out test item.

The preprocessing protocol follows the public settings referenced in `datasets/readme.md`, including ICLRec and DuoRec.

---

## Main Commands

Run commands from the repository root.

### Amazon Beauty

```bash
python src/main.py --dataset amazon_beauty --device cuda
```

### Steam

```bash
python src/main.py --dataset steam --device cuda
```

### Yelp

```bash
python src/main.py --dataset yelp --device cuda
```

The optional `--result_file` argument appends the final selected test metrics to a JSONL file:

```bash
python src/main.py \
    --dataset amazon_beauty \
    --device cuda \
    --result_file results/pilot.jsonl
```

The output directory is created automatically when necessary.

---

## Model Overview

PILOT contains two main components:

1. **Collaborative Prototype Prior**
2. **One-Step Flow Matching**

### Collaborative Prototype Prior

Given an embedded interaction history, PILOT summarizes the valid historical items using both:

- the mean representation of the valid history;
- the representation of the most recent valid item.

The two representations are averaged to obtain a base history summary.

When prototype modeling is enabled, the summary attends to a learnable prototype bank through multi-head attention. The prototype readout is added to the history summary through a learnable feature-wise scaling vector and normalized to form the initial latent representation `x0`.

Prototype modeling can be disabled with:

```bash
--use_prototype 0
```

In that case, the normalized history summary is used directly as `x0`.

### One-Step Flow Matching

During training, the target next-item embedding is denoted by `x1`. PILOT samples a time step `t` and constructs an interpolated latent state between the prior representation `x0` and the target representation `x1`:

```text
x_t = (1 - t) x0 + t x1
```

Training-time Gaussian noise can be added to the interpolated state through `--noise_std`.

The flow network conditions on:

- historical item representations;
- the current latent state;
- the sampled time step.

Time is represented with sinusoidal timestep embeddings, while the latent condition modulates the sequence representations through FiLM conditioning.

At inference time, PILOT first builds the collaborative prototype prior from the input sequence and then evaluates the flow network once at `t=0`:

```text
Interaction History
        │
        ▼
 Item Embeddings
        │
        ▼
Prototype Prior x0
        │
        ▼
Flow Network at t = 0
        │
        ▼
Next-item Representation
        │
        ▼
Full-catalog Ranking
```

The inference path therefore does not require an iterative ODE solver or multi-step sampling procedure.

---

## Training Objective

The total training objective contains three loss terms:

```text
L = λ_flow  · L_flow
  + λ_ce    · L_ce
  + λ_prior · L_prior
```

where:

- `L_flow` is the flow-matching reconstruction objective;
- `L_ce` is the next-item cross-entropy objective applied to the flow prediction;
- `L_prior` is an auxiliary next-item cross-entropy objective applied to the prototype-prior representation.

The corresponding command-line arguments are:

```text
--flow_weight
--ce_weight
--prior_ce_weight
```

with default values:

```text
1.0
0.5
0.3
```

respectively.

The same `t=0` branch used for one-step inference is supervised during training together with a randomly sampled point on the flow path, keeping the train-time and inference-time computations aligned.

---

## Default Hyperparameters

The following defaults correspond to the current `src/main.py`:

| Group | Argument | Default |
| --- | --- | ---: |
| Data | `--dataset` | `amazon_beauty` |
| Data | `--random_seed` | `2026` |
| Data | `--max_len` | `50` |
| Data | `--batch_size` | `512` |
| Data | `--metric_ks` | `10 20` |
| Runtime | `--device` | `cuda` |
| Backbone | `--hidden_size` | `128` |
| Backbone | `--num_blocks` | `4` |
| Backbone | `--dropout` | `0.1` |
| Backbone | `--emb_dropout` | `0.3` |
| Optimization | `--lr` | `0.001` |
| Optimization | `--epochs` | `500` |
| Optimization | `--eval_interval` | `1` |
| Optimization | `--patience` | `5` |
| Prototype prior | `--use_prototype` | `1` |
| Prototype prior | `--num_prototypes` | `192` |
| Prototype prior | `--prior_heads` | `4` |
| Flow path | `--noise_std` | `0.1` |
| Flow path | `--eps` | `0.001` |
| Flow path | `--s_modsamp` | `1.0` |
| Loss | `--flow_weight` | `1.0` |
| Loss | `--ce_weight` | `0.5` |
| Loss | `--prior_ce_weight` | `0.3` |

The optimizer is **Adam**.

### Parameter Constraints

- `hidden_size` must be divisible by `prior_heads`.
- The Transformer flow backbone uses four attention heads, so `hidden_size` must also be divisible by `4`.
- `metric_ks` should include `20` because model selection uses validation `NDCG@20`.

---

## Evaluation Protocol

PILOT uses full-catalog ranking for validation and testing.

For each user, the predicted representation is multiplied by the complete item embedding table:

```text
scores = representation × item_embedding_table^T
```

The default evaluation metrics are:

- `HR@10`
- `NDCG@10`
- `HR@20`
- `NDCG@20`

The implementation does not use negative sampling at evaluation time.

### Model Selection

The model is selected according to validation `NDCG@20`.

After every `--eval_interval` epochs:

- the model is evaluated on the validation split;
- an improved validation `NDCG@20` replaces the best state retained in memory;
- otherwise the early-stopping counter is increased.

Training stops after `--patience` consecutive validation evaluations without improvement. The best validation state is then restored directly from memory and evaluated on the test split.

The optional JSONL output written through `--result_file` stores the selected epoch, best validation `NDCG@20`, dataset name, random seed, timestamp, and final test metrics.

---

## Ablation Settings

The main components can be ablated by changing a single argument while keeping the remaining configuration unchanged.

### Without Prototype Read-out

```bash
python src/main.py --dataset amazon_beauty --use_prototype 0
```

### Without Flow-Matching Loss

```bash
python src/main.py --dataset amazon_beauty --flow_weight 0
```

### Without Next-Item Cross-Entropy Loss

```bash
python src/main.py --dataset amazon_beauty --ce_weight 0
```

### Without Prior Anchoring Loss

```bash
python src/main.py --dataset amazon_beauty --prior_ce_weight 0
```

---

## Reproducibility Notes

For reproducible comparison:

- use the same processed data files without changing the user/item mappings;
- keep the maximum sequence length fixed at `50` unless sequence length itself is being studied;
- use the same evaluation cutoffs for compared methods;
- select each run according to validation `NDCG@20`;
- report test performance only after validation-based model selection;
- use full-catalog evaluation consistently across compared methods.

The implementation fixes the random-number generators for Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA. It also enables deterministic cuDNN behavior and disables cuDNN benchmarking.

Exact results may still vary across hardware, PyTorch/CUDA versions, and low-level kernels.

---

## Paper

**PILOT: Prototype-Informed Flow Matching for One-Step Sequential Recommendation**

Accepted by **WISE 2026**.

---

## Citation

If you use PILOT, its processed datasets, or the implementation in your research, please cite our paper.

The complete BibTeX entry will be updated after the official WISE 2026 proceedings metadata becomes available.

```bibtex
@inproceedings{pilot2026,
  title     = {PILOT: Prototype-Informed Flow Matching for One-Step Sequential Recommendation},
  author    = {Wentao Gan, Fei Cao, Bangzuo Zhang},
  booktitle = {Proceedings of the International Conference on Web Information Systems Engineering (WISE)},
  year      = {2026}
}
```

---

## License

This project is released under the MIT License.

See [LICENSE](LICENSE) for details.
