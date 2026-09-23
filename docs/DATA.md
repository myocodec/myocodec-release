# Pretraining corpus

Twelve public sEMG corpora, pooled: **248,493 recordings, 25,114 channel-hours**, from
which **9,593,476 windows** of 5.0 s are drawn (53.1% of the enumerable windows; the rest
is lost to per-corpus activity filtering and partial-window drop).

Channel-hours, not recording-hours, is the relevant unit: the codec is channel-independent
and trains on single-channel windows.

| Corpus | recordings | subjects | channel-hours | native rate |
|---|---:|---:|---:|---:|
| emg2qwerty | 1,135 | 108 | **11,074** | 2 kHz |
| emg2pose | 25,253 | — | **6,763** | 2 kHz |
| Hyser | 6,615 | 20 | 2,593 | 2048 Hz |
| putEMG | 712 | 44 | 1,045 | 5120 Hz |
| EMG-EPN-612 | 91,800 | 306 | 1,014 | 200 Hz |
| GRABMyo | 15,351 | 43 | 682 | 2048 Hz |
| MeganePro | 45 | 45 | 610 | 2 kHz |
| Ninapro | 82,053 | 119 | 588 | 2 kHz |
| emg2speech (ALS) | 10,257 | 2 | 304 | 5 kHz → 2 kHz |
| CSL-HDEMG | 1,450 | 5 | 232 | 2048 Hz |
| Gaddy | 12,382 | 26 | 156 | 1 kHz |
| CapgMyo | 1,440 | 18 | 51 | 1 kHz |
| **Total** | **248,493** | | **25,114** | |

> The corpus is dominated by two of the three benchmark domains: emg2qwerty and emg2pose
> together are 71% of all channel-hours. The front-end is not domain-agnostic — it is
> pretrained largely on two of the three domains it is then evaluated on, though never on
> their evaluation recordings.

**Ninapro is six sub-databases, not one**, spanning 10–16 channels. Worth stating because
the name reads as a single corpus and the channel count is not constant across it:

| sub-DB | subjects | nominal channels | train recordings | train channel-hours |
|---|---:|---:|---:|---:|
| DB1 | 27 | 10 | 11,548 | 5.8 |
| DB2 | 40 | 12 | 13,885 | 129.1 |
| DB4 | 10 | 12 | 6,101 | 49.6 |
| DB5 | 10 | 16 | 4,635 | 52.2 |
| DB6 | 10 | 14 | 13,747 | 115.1 |
| DB7 | 22 | 12 | 9,385 | 64.5 |
| **total** | **119** | | | **416.2** |

DB3 and DB8 are not present. Records showing fewer than the nominal channel count have had
dead channels dropped by `build_corpus.py`, which removes any channel with std < 1e-6. Note
the citation range: DB1–DB3 are Atzori et al., but DB4/DB5 are Pizzolato et al., DB6 is
Palermo et al. and DB7 is Krasoulis et al.

**emg2speech (ALS) is not redistributed here.** Dropping it costs 1.2% of the corpus's
channel-hours; the other eleven corpora are publicly downloadable, and the pipeline runs
unchanged without it — omit it from the `build_corpus` loop below.

## Splits

Splits are assigned in `build_corpus.py`, before windows are cut, and **only `train`-split
recordings enter pretraining**. The validation and test recordings of every downstream
benchmark are excluded at shard-extraction time, so the frozen front-end has never seen the
evaluation data, even unsupervised.

| Corpus | split rule |
|---|---|
| emg2pose | official, from `emg2pose_metadata.csv` |
| emg2qwerty | official, held-out **user** via `metadata.csv` |
| Gaddy | official, book/sentence list in `testset_largedev.json` |
| Ninapro | official, by repetition: rep 5 → test, rep 2 → val |
| EMG-EPN-612 | held-out subject (its own `trainingSamples`/`testingSamples` share subjects, so using them would leak) |
| emg2speech | per-utterance hash (single-speaker corpus) |
| all others | held-out subject, by hash of the subject id |

## Per-corpus preprocessing

Each corpus is conditioned with **its own published pipeline**, not one common chain. The
full table with provenance per entry is `PAPER_SPEC` in
[`streaming_emg_codec/data/conditioning.py`](../streaming_emg_codec/data/conditioning.py);
the summary:

| Corpus | applied here | why |
|---|---|---|
| Gaddy | `butter(3, 2 Hz)` high-pass + 60 Hz notch comb (harmonics 1–7) | their `read_emg.py` does this; we read the raw `.npy` and bypass it |
| Hyser | `butter(8, 10–500 Hz)` + 50 Hz notch comb | their paper's chain; we glob the **raw** files, which PhysioNet states are unfiltered |
| CSL-HDEMG | `butter(4, 20–400 Hz)` | their paper's chain; not present in what we read (26.6% sub-20 Hz power) |
| EMG-EPN-612 | 50 Hz notch, fundamental only | 200 Hz native, so no harmonic is representable |
| emg2speech | 50 Hz notch, fundamental only | near a no-op; already band-passed from ~60 Hz |
| emg2qwerty, emg2pose, putEMG, GRABMyo, Ninapro, MeganePro, CapgMyo | **nothing** | already carry their own paper's preprocessing |

Only **17.87%** of training windows are touched. Pushing an already-filtered corpus through
a second, different chain is destructive: an earlier version high-passed everything at
20 Hz — a limb-sEMG convention — and on facial/neck speech EMG, where articulation lives at
roughly 2–15 Hz, that cost 11.06 points of downstream word error rate while leaving
validation cross-entropy unchanged.

Two deliberate non-actions:

- **The upper band is not harmonized.** It is set by each corpus's acquisition rate and
  hardware anti-alias filter, and it is signal, not contamination. EMG-EPN is 200 Hz native
  so it holds nothing above 100 Hz, while emg2speech keeps 44–52% of its power in
  150–450 Hz. Low-passing everything to the common floor would destroy real signal from the
  high-rate corpora to match an artefact of the low-rate one.
- **Ninapro DB1 is gated out of a rebuild, but is present in the released corpus.** Its
  Otto Bock 13E200 output is a rectified, smoothed RMS envelope at 100 Hz, not a raw EMG
  waveform, and `readers.py` now drops it on native rate (`MIN_NATIVE_FS`, which catches any
  other low-rate source too). That gate was added *after* the released shards were
  extracted, so the released checkpoint did train on DB1: **5.8 of Ninapro's 416.2 training
  channel-hours, 1.4% of Ninapro and 0.02% of the corpus.** Rebuilding the corpus with the
  current code excludes it; we report what the released model actually saw.

All conditioning is **causal** (`sosfilt`, forward-only, initial conditions from the first
sample). The authors' own implementations mostly use zero-phase `filtfilt`; a zero-phase
front end would make the whole chain non-causal and invalidate the streaming latency claim.
The magnitude response is identical and only the phase differs;
`conditioning.group_delay_ms()` reports the cost per corpus.

## Expected layout

Point `$EMG_DATA_ROOT` at a directory holding the corpora under these names — the globs are
in `CONFIG` at the bottom of
[`streaming_emg_codec/data/readers.py`](../streaming_emg_codec/data/readers.py):

```
$EMG_DATA_ROOT/
  emg2qwerty/*.hdf5                        + metadata.csv
  emg2pose/emg2pose_data/*.hdf5            + emg2pose_metadata.csv
  hyser/**/*raw*.dat                       (WFDB)
  grabmyo/Session*/**/*.dat                (WFDB)
  ninapro/**/*.mat
  putemg/*.hdf5
  csl/subject*/session*/*.mat
  capgmyo/*.mat
  gaddy/**/*_emg.npy                       + testset_largedev.json
  emgepn/**/trainingJSON/user*/*.json
  meganepro/*.mat
  emg2speech/*_emg_2khz.h5                 (not public)
```

Both Meta corpora (emg2qwerty, emg2pose) are on the public
`fb-ctrl-oss` S3 bucket. Hyser and GRABMyo are on PhysioNet; the `get-zip` endpoint crawls,
so prefer the S3 mirror:

```bash
aws s3 sync --no-sign-request s3://physionet-open/hd-semg/2.0.0/ "$EMG_DATA_ROOT/hyser/"
```

## Building the shards

```bash
export EMG_DATA_ROOT=/path/to/raw
export EMG_CORPUS_ROOT=/path/to/corpus2k
export EMG_SHARD_ROOT=/path/to/shards

# 1. per-dataset 2 kHz HDF5 + manifest (run once per corpus; --worker/--nworkers to shard)
for ds in emg2qwerty emg2pose hyser grabmyo ninapro putemg csl capgmyo gaddy emgepn meganepro; do
  python -m streaming_emg_codec.data.build_corpus --dataset "$ds"
done
cat "$EMG_CORPUS_ROOT"/manifest_*.jsonl > "$EMG_CORPUS_ROOT/manifest.jsonl"

# 2. conditioned, normalized 5 s windows -> h5 shards (~191 GB for the train split)
python -m streaming_emg_codec.data.extract_shards --split train --target 9593476
EMG_SHARD_ROOT=/path/to/shards_val \
  python -m streaming_emg_codec.data.extract_shards --split val --target 200000
```

Step 2 is the only point where per-corpus conditioning can be applied: the shards are a
bare `windows` array with no dataset label, and the manifest row that carries `dataset` is
consumed here. It is also the correct point — filtering after Ninapro's (stimulus,
repetition) segmentation would put 14–35% of every 200–506 sample segment inside the filter
transient, whereas here the whole recording is filtered before it is cut into windows.

Normalization is **per-window and across-channel**: one mean and one standard deviation
shared over channels and time within a window, so relative amplitude between channels
survives. Per-channel normalization would erase it.
