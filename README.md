# Synthetic Data Generation for Low-Resolution Face Recognition

Code for **"Improving Low-Resolution Face Recognition under Limited Data: How Synthetic Data Generation Can Close the Domain Gap"** at IJCB 2026 Focus Session "Generative AI for Fair and Secure Biometrics under Limited Data".

Luis S. Luevano, Ünsal Öztürk, Hatef Otroshi Shahreza, Anjith George, Sébastien Marcel

--------

Project page: <https://idiap.ch/paper/synth-lrfr>

Face recognition in surveillance settings has to deal with faces whose usable region falls well below the standard 112×112 input. Labelled high-resolution (HR) training data is abundant; labelled *native* low-resolution (LR) data, and above all *paired* native-LR/HR data, is not. The usual workaround is to **synthesize** LR data from HR faces — but how much synthesis effort actually pays off in recognition accuracy is unclear.

This repository implements and compares five adaptation strategies on a compact, edge-oriented backbone ([EdgeFace](https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface)), spanning three levels of synthesis effort, and evaluates all of them on **real** native-LR data as well as on the usual synthetic benchmarks.

The headline result is a **synthetic–real gap**: the degradation setting that is optimal on synthetic benchmarks (28 px, cubic-down/area-up) is the *worst* of the family on real LR, where the milder 56 px setting wins. More synthesis effort does not help monotonically, and a learned super-resolution front-end never beats simply feeding a strong frozen backbone the aligned LR image.

---

## Contents

| | |
|---|---|
| [Installation](#installation) | environment, dependencies, aligner setup |
| [Configuration](#configuration) | the environment variables everything reads |
| [Data preparation](#data-preparation) | building the synthetic LR datasets |
| [Training](#training) | the five adaptation strategies |
| [Evaluation](#evaluation) | synthetic benchmarks, IJB-C, TinyFace |
| [Reproducing the paper](#reproducing-the-paper) | table-by-table command map |
| [Repository layout](#repository-layout) | description of the respository contents |
| [License and attribution](#license-and-attribution) | Licensing information for this and other submodules. |

---

## Installation

```bash
conda create -n edgeface python=3.10
conda activate edgeface

# Match the CUDA version of your machine.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirement.txt
```

Training with `config.dali = True` (the default for the interpolation recipes) additionally needs [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html):

```bash
pip install nvidia-dali-cuda120
```

The evaluation entry points pull in a few extra packages (menpo for the IJB-C report, insightface for one of the aligners, ptflops for the complexity table):

```bash
pip install -r requirements_eval.txt
```

### Pretrained weights

The pretrained EdgeFace-Base backbone (frozen inside the SR and PDT pipelines,
and the `base, direct` baseline) is **not** part of this repository. Fetch it
from Hugging Face into `checkpoints/`, where the configs expect it:

```bash
bash scripts/fetch_checkpoints.sh
```

> **Note on licenses:** the weights are distributed at
> [Idiap/EdgeFace-Base](https://huggingface.co/Idiap/EdgeFace-Base) under
> **CC-BY-NC-SA-4.0**, which is more restrictive than this repository's code
> license (BSD-3-Clause). Every other EdgeFace variant, including the quantized
> ones, is available from the official release
> ([GitLab](https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface),
> [GitHub](https://github.com/otroshi/edgeface)).

### Differentiable Face Aligner

TinyFace is reported under two alignment pipelines, and the choice moves absolute accuracy by a large margin. The DFA pipeline uses the CVLface aligners, which are third-party models hosted on the Hugging Face Hub and are **not** redistributed here. Fetch them once:

```bash
bash scripts/setup_dfa.sh          # ResNet-50 aligner, what the paper uses
bash scripts/setup_dfa.sh --all    # also the MobileNet variant
```

This clones the repositories at the exact revisions used for the paper. Running the aligner needs `omegaconf` (included in `requirements_eval.txt`). Requires [git-lfs](https://git-lfs.com).

---

## Configuration

The repository contains no machine-specific absolute paths. Everything is resolved from a handful of environment variables, read both by [`configs/paths.py`](configs/paths.py) and by the SLURM launchers.

```bash
cp slurm/env.sh.example slurm/env.sh   # git-ignored; edit for your machine
```

| Variable | Default | Meaning |
|---|---|---|
| `LRFR_DATA_ROOT` | `./data` | root of the preprocessed `.rec` / `.idx` / `.bin` datasets |
| `TINYFACE_ROOT` | `$LRFR_DATA_ROOT/tinyface` | root of the TinyFace distribution |
| `IJBC_ROOT` | `$LRFR_DATA_ROOT/IJBC` | root of the IJB-C distribution |
| `STAGE_DIR` | `$LRFR_DATA_ROOT/staging` | where launchers assemble per-run dataset views |
| `CONDA_BIN` | `$HOME/miniforge3/bin/conda` | conda binary used by the launchers |
| `CONDA_ENV` / `CONDA_ENV_EVAL` | `edgeface` | environments for training / evaluation |
| `NPROC_PER_NODE` | `4` | GPUs per node for `torchrun` |
| `LOAD_CLUSTER_MODULES` | `0` | set to `1` where CUDA/cuDNN come from Lmod modules |

`STAGE_DIR` exists because the KD and PDT recipes read an LR and an HR record side by side under specific names; the launchers build those views with hard links, so nothing is duplicated on the same filesystem. On a cluster, point it at node-local storage to keep training off the shared filesystem.

`#SBATCH --account` and `--constraint` are deliberately absent from the launchers — SLURM parses those before `env.sh` is read. Pass them on the command line:

```bash
sbatch --account=<your-account> --constraint="rtx3090|v100" \
    slurm/train_edgeface_lr.run cubic area 56
```

Every launcher also runs as a plain shell script outside SLURM, in which case the array/job variables fall back to single-task defaults.

---

## Data preparation

All LR training data is derived from [WebFace4M](https://www.face-benchmark.org/), which was pre-processed using the InsightFace template in MXNetRecordIO format. Place the pre-processed `train.rec` / `train.idx` / `train.lst` and the desired verification `.bin` files under `$LRFR_DATA_ROOT/webface4m/` (more info on how to create bin files from scratch [here](https://gist.github.com/fengyuentau/53f4200b3f721943f8a714433b9cc685)).

Throughout, a degradation setting is written **`s ↓d/↑u`**: downsample to `s` px with interpolation `d`, upsample back to 112 px with `u`. So `28 ↓c/↑a` is cubic-down, area-up.

### 1. Global shuffle index

One reproducible shuffle order for the whole dataset, shared by every array task:

```bash
python scripts/generate_shuffle_index.py \
    --rec    "$LRFR_DATA_ROOT/webface4m/train.rec" \
    --output "$LRFR_DATA_ROOT/webface4m/shuffled_index.txt"
```

### 2. A `passthrough` copy (the HR reference)

`passthrough` mode copies raw JPEG bytes without re-encoding, so the HR record used by KD, PDT and SR training is bit-identical to the source:

```bash
sbatch slurm/preprocess_rec.run passthrough
sbatch slurm/merge_recs.run passthrough_webface4m
```

### 3. Low effort — interpolation degradation

`scripts/preprocess_rec.py` controls the downsampling interpolation (`--interp-down`), the upsampling interpolation (`--interp-up`) and the target resolution (`--downsample-size`) independently:

```bash
sbatch slurm/preprocess_rec.run downsample cubic area 56
sbatch slurm/merge_recs.run processed_downsample_56_cubic_area_webface4m
```

Of the `{area,cubic,linear}²` combinations, four resolution/interpolation settings converged reliably and form the paper's comparison set: `56 ↓c/↑a`, `28 ↓a/↑c`, `28 ↓c/↑a` and `14 ↓a/↑c`.

To submit many combinations at once, along with their chained merge jobs:

```bash
python data_slurm_submission.py
```

### 4. Medium effort — Real-ESRGAN degradation

A high-order stochastic degradation (random blur kernel, random resize, Gaussian or Poisson noise, JPEG cycle; a second-order block with probability 0.8; a final resize to 32×32 with optional sinc low-pass and JPEG), stored natively at 32 px alongside the 112 px source:

```bash
sbatch slurm/preprocess_rec_esrgan.run 32
sbatch slurm/merge_recs.run processed_noup_32_esrgan_passthrough_webface4m
```

### 5. Degraded verification sets

The synthetic benchmarks are built by degrading the verification pairs of the standard `.bin` files, in two modalities: **HR→LR** (one image degraded) and **LR→LR** (both):

```bash
python scripts/generate_lr_bin.py \
    --bin-dir "$LRFR_DATA_ROOT/webface4m" \
    --size 28 --downsample cubic --upsample area --mode both
```

`--mode both` degrades both images and writes `<stem>_28_cubic_area_lr2lr.bin`;
`--mode second` degrades only the second image, giving the HR→LR set
`<stem>_28_cubic_area_hr2lr.bin`. Those stems are what the `val_targets` lists in
the configs refer to.

For the native 32 px models the bins must themselves be at 32 px, degraded the same way as the training data:

```bash
python scripts/generate_lr_bin.py --bin-dir "$LRFR_DATA_ROOT/webface4m" \
    --method esrgan --output-size 32 --mode both
```

---

## Training

All compact models are EdgeFace-S (3.65 M parameters) trained from scratch for 100 epochs with CosFace, PartialFC and AdamW. Each launcher writes to the directory named by `config.output`.

### Baseline — HR-trained (direct feed)

The reference every adaptation is compared against: an HR-trained backbone fed the aligned LR face with no front-end and no LR-specific training. As the paper shows, this is a hard baseline.

```bash
sbatch slurm/train_edgeface_hr.run
```

### Low effort — interpolation augmentation

Retrain EdgeFace-S end-to-end on the degraded record. No HR data, no teacher; LR exposure is pure augmentation.

```bash
sbatch slurm/train_edgeface_lr.run cubic area 56      # best on real LR
sbatch slurm/train_edgeface_lr.run cubic area 28      # best on synthetic LR
```

### Knowledge distillation

A frozen HR-trained EdgeFace-S teacher supervises an LR student of identical architecture. On top of the CosFace loss the student matches the teacher's L2-normalised embeddings and sampled PartialFC weights.

Two earlier runs must exist, and **both sets of checkpoints are loaded**:

- **Teacher** — the HR baseline (`slurm/train_edgeface_hr.run`); `train_v2_kd.py` loads `edgeface_s_gamma_05/checkpoint_gpu_{rank}.pt` frozen, via `config.teacher_checkpoint`.
- **Student** — warm-started from the LR backbone of the same setting (`slurm/train_edgeface_lr.run` with the same `<down> <up> <size>`). On the first launch the launcher copies the LR run's `checkpoint_gpu_*.pt` into the KD output directory, where `config.resume = True` picks them up; distillation then continues from the LR run's epoch counter (100 → `num_epoch = 200`, i.e. ~100 distillation epochs on top of the LR init).

The checkpoints are per-rank shards, so KD must run on the same number of GPUs as the runs that produced them (4 throughout).

```bash
sbatch slurm/train_edgeface_lr.run cubic area 56   # student init (if not already trained)
sbatch slurm/train_edgeface_kd.run cubic area 56
```

### Prepended Domain Transformer

Freeze the HR backbone entirely and adapt only the probe, with a translator trained under a contrastive loss over PK-sampled LR/HR batches. Requires the HR baseline first.

```bash
sbatch slurm/train_edgeface_pdt.run cubic area 56
```

### Medium effort — native 32 px stem

EdgeFace-S with its 4×4 stride-4 stem replaced by a 1×1 convolution, trained directly on the Real-ESRGAN-degraded 32 px record with no upsampling anywhere.

```bash
sbatch slurm/train_edgeface_noup32_esrgan.run
```

### High effort — learned SR front-end

Move upsampling *inside* the model: two ×2 sub-pixel (PixelShuffle) stages take 32→64→128 px, then four valid-padded k=5 convolutions trim to 112 px. No interpolation function is used anywhere. Two feature-block variants are provided: a plain sub-pixel stack (`espcn`, ≈0.86 M parameters) and an RRDB trunk (`resrgan`, ≈6.5 M).

The front-end sits in front of a PDT translator and a **frozen** EdgeFace-base, and is trained in two stages.

```bash
# Stage 1: SR pretraining on (LR-32, HR-112) pairs, with an L1 term plus an
# identity-aware cosine term through the frozen backbone embedding.
sbatch slurm/train_sr_pretrain.run espcn

# Stage 2: warm-start from stage 1, then train upsampler and PDT jointly under
# the contrastive loss. Backbone stays frozen throughout.
sbatch slurm/train_pdt_up32.run espcn
```

### Long runs

`slurm/monitor_and_relaunch.sh` resubmits a job with `--resume` each time it finishes, up to `--max-relaunches` times:

```bash
./slurm/monitor_and_relaunch.sh -n 6 slurm/train_edgeface_lr.run cubic area 56
```

Monitor options go **before** the script name; script arguments go after.

---

## Evaluation

### Synthetic verification benchmarks

LFW, CFP-FP and AgeDB-30, at full resolution and degraded to 56/28/14/7 px in both modalities:

```bash
sbatch slurm/evaluate_model.run \
    configs/edgeface_s_gamma_05_lr_56_cubic_area.py \
    edgeface_s_gamma_05_lr_56_cubic_area/model.pt
```

Or directly, without SLURM:

```bash
python evaluate_model.py \
    --config configs/edgeface_s_gamma_05_lr_56_cubic_area.py \
    --checkpoint edgeface_s_gamma_05_lr_56_cubic_area/model.pt \
    --val-targets lfw cfp_fp agedb_30 lfw_28 lfw_28_lr2lr \
    --data-dir "$LRFR_DATA_ROOT/webface4m"
```

### IJB-C

We used the IJB-C test suite to run the evaluations. More info [here](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

```bash
sbatch slurm/evaluate_model_ijbc.run \
    edgeface_s_gamma_05_lr_56_cubic_area/model.pt edgeface_s_gamma_05
```

`plot_roc_multi.py` overlays several score files to produce the FMR / 1−FNMR
figure.

### TinyFace — real native LR

This is the evaluation that exposes the synthetic–real gap, and the one to trust when comparing adaptation strategies. Identification against a ~157k-image gallery, reported as mAP and Rank-*k*.

The protocol's `.mat` pair files (`gallery_match_img_ID_pairs.mat`, `probe_img_ID_pairs.mat`) are **not** redistributed here — they ship with the [TinyFace](https://qmul-tinyface.github.io/) download, and the evaluation reads them from `$TINYFACE_ROOT/Face_Identification_Evaluation` (override with `--mat-dir`). The Python ports of the protocol's MATLAB scoring code live under `eval_tinyface/`.

Align first. **Any TinyFace comparison must fix the aligner** — moving from padded crops to DFA raises every model substantially.

```bash
for split in Train Probe Gallery_Match Gallery_Distractor; do
    sbatch slurm/align_tinyface.run "$split" dfa-resnet50 112
done
```

Then evaluate:

```bash
sbatch slurm/evaluate_model_tinyface.run \
    configs/edgeface_s_gamma_05_lr_56_cubic_area.py \
    edgeface_s_gamma_05_lr_56_cubic_area/model.pt \
    face_alignment/tinyface_alignment/aligned_dfa-resnet50_112
```

For the SR front-end the probes go through the learned upsampler first, starting from native 32 px crops:

```bash
for split in Probe Gallery_Match Gallery_Distractor; do
    sbatch slurm/align_tinyface.run "$split" dfa-resnet50 32
done
sbatch slurm/upsample_tinyface.run espcn
sbatch slurm/evaluate_model_tinyface.run \
    configs/edgeface_base_PDT_up32_espcn.py \
    edgeface_base_PDT_up32_espcn/model.pt \
    face_alignment/tinyface_alignment/upsampled_112_espcn
```

### Demographic fairness

The RFW fairness assessment is documented in [`fairness/`](fairness/README.md).
The scoring script is not part of this release yet; the protocol is written up
there so the numbers can be reproduced.

### Qualitative figures

```bash
# SR reconstructions: input / SR output / HR target
python visualize_upsampler.py --config configs/edgeface_base_PDT_up32_espcn.py \
    --stage1-checkpoint sr_pretrain_espcn/sr_pretrained.pt --n-samples 8 \
    --rec "$LRFR_DATA_ROOT/LR_for_PDT_processed_noup_32_esrgan_webface4m/" \
    --rec-hr "$LRFR_DATA_ROOT/HR_for_PDT_passthrough_webface4m/" \
    --output-dir viz/espcn

# PDT translator outputs on native-LR TinyFace probes
python visualize_pdt_samples.py --config configs/edgeface_s_gamma_05_PDT_28_cubic_area.py \
    --checkpoint edgeface_s_gamma_05_PDT_28_cubic_area/model.pt \
    --source tinyface --tinyface-dir face_alignment/tinyface_alignment/Probe_aligned_dfa-resnet50_112 \
    --num-samples 16 --output-dir viz/pdt
```

---

## Reproducing the paper

Trained model checkpoints are **not** distributed; the only pretrained network used is EdgeFace-Base, fetched by `scripts/fetch_checkpoints.sh`. Every number below comes from training the model with the listed launcher and then evaluating it.

| Paper result | Train | Evaluate |
|---|---|---|
| Synthetic cross-resolution macro-mean, both modalities | `train_edgeface_lr.run`, `train_edgeface_kd.run`, `train_edgeface_pdt.run` × {56 ↓c/↑a, 28 ↓a/↑c, 28 ↓c/↑a, 14 ↓a/↑c} | `evaluate_model.run` |
| Per-benchmark HR→LR and LR→LR | same models | `evaluate_model.run` |
| Synthetic vs real, same models | `train_edgeface_lr.run` | `evaluate_model.run` + `evaluate_model_tinyface.run` (DFA) |
| TinyFace under `aligned_pad` crops | LR / KD / PDT families | `align_tinyface.run <split> mtcnn 112`, then `evaluate_model_tinyface.run` |
| TinyFace under DFA alignment | all of the above, plus `train_sr_pretrain.run` + `train_pdt_up32.run` | `align_tinyface.run <split> dfa-resnet50 112`, then `evaluate_model_tinyface.run` |
| Comparison against EFaR 2023 | `train_edgeface_hr.run`, `train_edgeface_lr.run cubic area 56` | `evaluate_model_tinyface.run`; `python flops.py` for the complexity column |
| HR verification and IJB-C | all LR backbones and KD variants | `evaluate_model.run`, `evaluate_model_ijbc.run` |
| Effect of the SR front-end in isolation | `train_sr_pretrain.run` (stage 1 only) | `upsample_tinyface.run` + `evaluate_model_tinyface.run` |
| Fairness on RFW | HR, `56 ↓c/↑a`, `28 ↓c/↑a` backbones | see [`fairness/`](fairness/README.md) |

All reported accuracies use horizontal flip as test-time augmentation.

Two things worth repeating, because they determine whether a comparison means anything:

- **Fix the aligner.** Absolute TinyFace accuracy shifts substantially between
  the padded-crop and DFA pipelines. Cross-paper TinyFace numbers collected under
  different alignment can differ by tens of points.
- **Report the direct-feed baseline.** A restoration or translation pipeline
  should be compared against simply feeding the aligned LR image to the same
  backbone, before it is claimed to help.

---

## Repository layout

```
configs/            one config per experiment; paths.py resolves LRFR_DATA_ROOT
slurm/              launchers; env.sh.example holds all site configuration
scripts/            dataset synthesis, bin generation, weight and aligner fetch
backbones/          EdgeFace variants, iresnet, PDT wrapper and SR upsamplers
face_alignment/     MTCNN and DFA pipelines, TinyFace alignment and lists
eval/, eval_tinyface/  verification and Python ports of the TinyFace protocol
fairness/           RFW fairness protocol (scoring script pending)
LICENSES/           full license texts (REUSE-compliant layout)
docs/               inherited InsightFace documentation
```

Entry points:

| Script | Purpose |
|---|---|
| `train_v2.py` | standard training (HR baseline, interpolation LR, native 32 px stem) |
| `train_v2_kd.py` | knowledge distillation from a frozen HR teacher |
| `train_PDT.py` | PDT translator, optionally with a prepended upsampler |
| `train_sr_pretrain.py` | stage-1 identity-aware SR pretraining |
| `evaluate_model.py` | verification benchmarks, no distributed setup needed |
| `eval_ijbc.py` | IJB-C TAR@FAR |
| `eval_tinyface_identification.py` | TinyFace mAP and CMC |
| `upsample_tinyface.py` | apply a trained upsampler to aligned TinyFace crops |

### Changes from the original EdgeFace

| File | Change |
|---|---|
| `train_v2.py` | `--resume-override` flag; gradient clipping covers both backbone and `PartialFC_V2` |
| `train_v2_kd.py` | new — embedding and PartialFC-weight distillation from a frozen teacher |
| `train_PDT.py` | new — PK-sampled contrastive training of a PDT translator and optional upsampler |
| `train_sr_pretrain.py` | new — stage-1 SR pretraining with an identity-aware perceptual term |
| `backbones/PDT/upsamplers.py` | new — sub-pixel and RRDB 32→112 upsamplers |
| `dataset.py` | native input resolution threaded through the DALI pipeline |
| `eval/verification.py` | fixed CPU/CUDA device mismatch; fixed `acc1` hardcoded to 0 |
| `scripts/preprocess_rec.py` | `passthrough` and `esrgan` modes, independent `--interp-down` / `--interp-up`, `--shuffle-index-file` |
| `scripts/degradations.py` | vendored Real-ESRGAN high-order degradation utilities |
| `scripts/generate_lr_bin.py` | degraded verification bins in HR→LR and LR→LR modalities |
| `scripts/generate_shuffle_index.py` | global shuffle order shared across array tasks |
| `scripts/merge_rec_parts.py` | merge array-job output parts into one `.rec` |
| `evaluate_model.py` | standalone evaluation, no distributed setup |
| `face_alignment/dfa/` | CVLface DFA aligner integration |

---

## License and attribution

This repository is released under the BSD 3-Clause license, see [`LICENSE`](LICENSE),
and is [REUSE](https://reuse.software/)-compliant: every file carries its
copyright and license, either in an SPDX header, or through
[`REUSE.toml`](REUSE.toml) for files that cannot hold comments. Full license
texts are in [`LICENSES/`](LICENSES/). Verify with `reuse lint`, or via the
bundled [pre-commit](https://pre-commit.com/) configuration
(`pre-commit run --all-files`), which also runs basic repository checks.

### Models and datasets used at run time

This repository makes use of the following models and datasets at running time
but does **not** redistribute any of them — this release contains code only.
Each item must be obtained from its distributor under its own terms.

| Name | Type | Original distribution link | License / terms |
|---|---|---|---|
| EdgeFace-Base weights | model | [Idiap/EdgeFace-Base](https://huggingface.co/Idiap/EdgeFace-Base) | CC BY-NC-SA 4.0 |
| CVLface DFA aligners | model | [minchul/cvlface_DFA_resnet50](https://huggingface.co/minchul/cvlface_DFA_resnet50) | follows its training dataset, WIDER FACE: CC BY-NC-ND 4.0; used here for evaluation-time TinyFace alignment only |
| WebFace4M | dataset | [face-benchmark.org](https://www.face-benchmark.org/) | research use only (signed agreement) |
| LFW | dataset | [vis-www.cs.umass.edu/lfw](https://vis-www.cs.umass.edu/lfw/) | research use only |
| CFP-FP | dataset | [cfpw.io](http://www.cfpw.io/) | research use only (agreement) |
| AgeDB-30 | dataset | [ibug.doc.ic.ac.uk/resources/agedb](https://ibug.doc.ic.ac.uk/resources/agedb/) | research use only |
| CALFW | dataset | [whdeng.cn/CALFW](http://whdeng.cn/CALFW/index.html) | research use only (academic) |
| CPLFW | dataset | [whdeng.cn/CPLFW](http://whdeng.cn/CPLFW/index.html) | research use only (academic) |
| IJB-C | dataset | [nist.gov](https://www.nist.gov/programs-projects/face-challenges) | research use per NIST distribution agreement |
| TinyFace | dataset | [qmul-tinyface.github.io](https://qmul-tinyface.github.io/) | research use only (no explicit license) |
| RFW | dataset | [whdeng.cn/RFW](http://www.whdeng.cn/RFW/index.html) | research use only (academic, agreement) |

Links point to each item's original distribution page and may be temporarily
offline; obtain every dataset directly from its distributor under its own terms.

### Third-party code

It builds on work distributed under its own terms:

- [EdgeFace](https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface) — the
  backbone and the original training pipeline, BSD 3-Clause, Idiap Research
  Institute; see [`README_EDGEFACE.md`](README_EDGEFACE.md).
- [InsightFace](https://github.com/deepinsight/insightface) (MIT, © Jiankang
  Deng and Jia Guo) — the training and evaluation framework this repository is
  derived from, see [`README_INSIGHTFACE.md`](README_INSIGHTFACE.md) and
  [`LICENSES/MIT.txt`](LICENSES/MIT.txt). Derived files carry both copyright
  lines and a note describing the modifications.
- [mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch) (MIT, © Dan
  Antoshchenko) — vendored under `face_alignment/mtcnn_pytorch/`. The alignment
  helpers `align_trans.py` and `matlab_cp2tform.py` are by Yafei Zhao (2017),
  vendored via [AdaFace](https://github.com/mk-minchul/AdaFace) (MIT, © Minchul
  Kim), as are `face_alignment/align.py` and `mtcnn.py`.
- [BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0) /
  [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — the degradation model
  ported in `scripts/degradations.py`.
- [pytorch-insightface](https://github.com/nizhib/pytorch-insightface) (MIT,
  © Evgeny Nizhibitsky) — `backbones/iresnet.py`.
- [CVLface](https://huggingface.co/minchul/cvlface_DFA_resnet50) — the DFA
  aligners, fetched at setup time rather than redistributed. The model card
  declares no license of its own and defers to the license of its training
  dataset, [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) (CC BY-NC-ND 4.0).
- The [TinyFace](https://qmul-tinyface.github.io/) identification protocol —
  no explicit license; its MATLAB code and `.mat` pair files are **not**
  redistributed (they ship with the dataset). `eval_tinyface/` contains this
  repository's Python ports of the scoring code.

## Citation

```bibtex
@inproceedings{luevano2026lrfr,
  title     = {Improving Low-Resolution Face Recognition under Limited Data:
               How Synthetic Data Generation Can Close the Domain Gap},
  author    = {Luevano, Luis S. and {\"O}zt{\"u}rk, {\"U}nsal and
               Otroshi Shahreza, Hatef and George, Anjith and Marcel, S{\'e}bastien},
  booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
  year      = {2026},
  note      = {Accepted at the 2026 IJCB Focus Session "Generative AI for Fair and Secure Biometrics under Limited Data"}

}
```

## Acknowledgments

Funded by Frontex under the Frontex Research Grants Programme, Call
2024/CFP/INNOVATE/01, Grant Agreement No. 2025/280; the CarMen project,
HORIZON-CL3-2023-BM-01, no. 101168325; the PopEye project,
HORIZON-CL3-2023-BM-01, no. 101168317; and the CERTAIN project,
HORIZON-CL4-2024-DATA-01, no. 101189650.
