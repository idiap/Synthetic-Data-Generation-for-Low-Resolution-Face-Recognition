# Demographic fairness evaluation (RFW)

> **Status: to be released.** The code to reproduce the evaluation reported in the paper was will be added here.

## What is measured

Demographic fairness on [RFW](http://www.whdeng.cn/RFW/index.html), which splits
subjects into four groups: African, Asian, Caucasian and Indian.

Three EdgeFace-S backbones are compared — the HR-trained baseline and the
`56 ↓c/↑a` and `28 ↓c/↑a` LR-augmented variants — under the following protocol.

1. **Threshold calibration.** For each backbone, pick the score threshold that
   gives TMR@FMR = 10⁻² on IJB-C, using the same IJB-C protocol as the HR
   verification table (`slurm/evaluate_model_ijbc.run`). The threshold is fixed
   per model and reused unchanged on RFW.
2. **Probe degradation.** RFW probes are degraded with bicubic downsampling to
   the model's training resolution and bicubic upsampling back to 112 px.
   Results are reported at 56 and 28 px only: at 14 and 7 px the global FNMR
   exceeds 0.9 for every model, the fixed threshold no longer separates mated
   from non-mated scores, and the per-group comparison stops being informative.
3. **Per-group rates.** For each group *d*, compute FMR as a percentage and the
   ratio `r_d = FMR_d / g`, where `g` is the geometric mean of the FMR over the
   four groups. `r_d = 1` means the group sits exactly at the geometric mean.
   A group with no false matches has its FMR floored at `0.5/3000`, half the
   smallest non-zero rate observed.
4. **Aggregate disparity.** Report the Gini coefficient of the FMR over the four
   groups: 0 when all groups share the same FMR, growing with the spread.

## What the paper concludes

The Caucasian group has the lowest FMR ratio in every reported cell and the
African group the highest, matching the bias previously reported on RFW. LR-aware
training does not systematically reduce that spread: the FMR Gini of the
`56 ↓c/↑a` and `28 ↓c/↑a` backbones moves in both directions relative to the
HR-trained model depending on test resolution, and the largest single group
ratio anywhere in the table belongs to `28 ↓c/↑a`. Average-accuracy gains from
LR-aware synthesis therefore do not translate into reduced demographic
disparity.
