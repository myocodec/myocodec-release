"""Per-dataset signal conditioning.

Each dataset is preprocessed with ITS OWN published pipeline rather than one common chain
invented here. The table below records, per corpus, what that pipeline is, where it was
read from, and whether the released files already carry it.

Why per-corpus and not one chain: an earlier version high-passed everything at 20 Hz,
which is a limb-sEMG convention. On facial/neck speech EMG, where articulation lives at
roughly 2-15 Hz, that removed signal rather than contamination -- it cost 11.06 points of
word error rate on the downstream speech task (82.56% vs 71.50% over matched steps, about
7 sd) while leaving validation cross-entropy and auxiliary phoneme accuracy unchanged. The
loss was specific to intelligibility. Pushing an already-filtered corpus through a second,
different filter chain is destructive, so corpora that ship their paper's recipe are left
alone: only 17.87% of the training windows are touched here.

Fields:
  hp     (order, corner Hz) or None
  bp     (order, low Hz, high Hz) or None
  notch  (fundamental Hz, max harmonic) or None; Q is NOTCH_Q
  rate   the native rate the paper specifies the filter at
  src    provenance: 'code' (read from the authors' implementation), 'paper' (stated in
         the dataset paper), 'requested', or a combination
  act    whether anything is actually applied

CAUSAL by construction. The codec is a streaming model, so a zero-phase filtfilt front end
would make the whole chain non-causal and quietly invalidate the latency claim. The
authors' own implementations mostly use filtfilt; we use sosfilt instead. The magnitude
response is identical and only the phase differs. Every stage runs forward-only with
initial conditions set from the first sample, and `group_delay_ms` reports the cost.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi

NOTCH_Q = 30.0          # ~2 Hz wide at 60 Hz: narrow enough to cost negligible signal
TRIM_MS = 150.0         # discard this much of the head after filtering. A pure step settles
                        # in 0 ms thanks to the zi initialisation, but a real recording
                        # starting at a large DC offset still shows a ~30 ms transient
                        # reaching 38x the signal's own std, because each cascade stage is
                        # initialised from the PREVIOUS stage's first output rather than its
                        # steady state. A window containing that spike would have its
                        # per-window std wrecked. 150 ms covers the measured worst case
                        # (impulse ringing dies in 120 ms with seven notches) and costs
                        # nothing on multi-second recordings.


PAPER_SPEC: dict[str, dict] = {
    # ---- NEEDS ACTION: the released data does NOT carry the paper's preprocessing ----

    # Verified in dgaddy/silent_speech read_emg.py:
    #   scipy.signal.iirnotch(freq, 30, fs) over 60 Hz harmonics 1..7
    #   scipy.signal.butter(3, 2, 'highpass', fs=fs)
    # We load the raw *_emg.npy and bypass all of it.
    'gaddy':      dict(hp=(3, 2.0), bp=None, notch=(60.0, 7), rate=1000,
                       src='code', act=True),

    # Hyser paper: 8th-order Butterworth 10-500 Hz + notch comb at 50 Hz to 400 Hz. That is
    # how their PREPROCESSED files are made; PhysioNet states the *raw* files have no
    # filtering at all, and our glob takes *raw*.dat. Alternative: read their preprocessed
    # files instead and apply nothing.
    'hyser':      dict(hp=None, bp=(8, 10.0, 500.0), notch=(50.0, 8), rate=2048,
                       src='paper', act=True),

    # CSL-HDEMG (Amma et al.): zero-lag 4th-order Butterworth 20-400 Hz. Not present in what
    # we read -- we measure 26.57% sub-20 Hz power and a 5.3x drift ratio. 20 Hz is THEIR
    # corner for this limb HD-EMG dataset, not a convention imported from elsewhere.
    'csl':        dict(hp=None, bp=(4, 20.0, 400.0), notch=None, rate=2048,
                       src='paper', act=True),

    # ---- ALREADY CARRIES ITS OWN PREPROCESSING: do nothing ----

    # emg2qwerty paper: hardware bandpass -3 dB at 20/850 Hz plus a 40 Hz digital high-pass
    # post-acquisition. Their transforms.py has no filtering at all -- only spectrogram and
    # augmentation. Our measured ~40 Hz corner corroborates the digital stage.
    'emg2qwerty': dict(hp=None, bp=None, notch=None, rate=2000, src='paper+code', act=False,
                       note='HW 20-850 Hz + 40 Hz digital HP upstream; repo applies no filtering'),

    # emg2pose paper: same CTRL-labs wristband, hardware bandpass 20-850 Hz. transforms.py
    # again has no filtering. No notch is documented, and none was applied -- the 33-39x
    # 60 Hz line we measure is genuinely in the released data. Matching the paper means
    # leaving it there.
    'emg2pose':   dict(hp=None, bp=None, notch=None, rate=2000, src='paper+code', act=False,
                       note='HW 20-850 Hz upstream; no notch documented, line left in place'),

    # putEMG paper: built-in ANALOGUE bandpass 3-900 Hz. Its 49.8% sub-20 Hz power is by
    # design at a 3 Hz corner, not contamination -- a 20 Hz high-pass would have destroyed
    # it, the same mistake that cost 11 WER points on gaddy.
    'putemg':     dict(hp=None, bp=None, notch=None, rate=5120, src='paper', act=False,
                       note='analogue 3-900 Hz upstream; sub-20 Hz content is intended'),

    # GRABMyo (PhysioNet): 4th-order Butterworth 10-500 Hz + 60 Hz notch, and the dataset is
    # released with that already applied.
    'grabmyo':    dict(hp=None, bp=None, notch=None, rate=2048, src='paper', act=False,
                       note='BP 10-500 Hz + 60 Hz notch already applied to released files'),

    # Ninapro: DB1 Otto Bock 13E200, 90-450 Hz; DB2/DB3 Delsys Trigno, 20-450 Hz. Powerline
    # is removed by the Ninapro pipeline with a 50 Hz HAMPEL filter, which interpolates the
    # spectrum only where it detects a peak rather than notching unconditionally -- already
    # in the released data.
    # CAVEAT: DB1's Otto Bock output is a rectified, smoothed RMS ENVELOPE at 100 Hz, not
    # raw EMG -- a different signal class. It is excluded from the corpus by the native-rate
    # gate in readers.py (MIN_NATIVE_FS), so no DB1 window reaches the codec.
    'ninapro':    dict(hp=None, bp=None, notch=None, rate=2000, src='paper', act=False,
                       note='DB1 90-450 Hz envelope, DB2/3 20-450 Hz + 50 Hz Hampel, upstream'),

    # MeganePro: Ninapro group, Delsys electrodes, same 50 Hz Hampel powerline stage.
    'meganepro':  dict(hp=None, bp=None, notch=None, rate=2000, src='paper', act=False,
                       note='Delsys + 50 Hz Hampel upstream'),

    # CapgMyo (Du et al. 2017): hardware bandpass 20-380 Hz at 1 kHz, values normalised to
    # [-1, 1]. Our measured 1.20% sub-20 Hz is consistent.
    'capgmyo':    dict(hp=None, bp=None, notch=None, rate=1000, src='paper', act=False,
                       note='HW 20-380 Hz upstream'),

    # ---- UNRESOLVED: left as no-ops on purpose ----

    # EMG-EPN-612: Myo Armband, 8 ch at 200 Hz, which emits already-conditioned data. The
    # BP 20-90 Hz + 50 Hz notch that appears in the literature is what USERS of the dataset
    # apply downstream, not what the dataset itself ships. Applying it would be importing a
    # convention, which is exactly the error this table exists to correct.
    # 50 Hz notch added on request to remove powerline. Fundamental only: at 200 Hz the
    # Nyquist is 100 Hz, so no harmonic is representable.
    # FLAG: EMG-EPN-612 was recorded in Quito, Ecuador, which is a 60 Hz mains country, and
    # we measure no significant line at EITHER frequency here (50 Hz 0.3-0.8x, 60 Hz
    # 0.3-1.2x). 60.0 is the physically motivated value if this is meant to catch real hum.
    'emgepn':     dict(hp=None, bp=None, notch=(50.0, 1), rate=200, src='requested', act=True),

    # Non-public 31-channel speech-EMG corpus, source 5 kHz stored at 2 kHz.
    # No external paper to match. Measured as already band-passed from ~60 Hz: 0.00% below
    # 20 Hz and 0.03% in 20-60 Hz.
    # 50 Hz notch added on request. Fundamental only, to remove as little signal as
    # possible. FLAG: this corpus measures 0.03% of its power in the whole 20-60 Hz band
    # and no line at 50 Hz (0.3x) -- it is already band-passed from ~60 Hz, so the notch
    # lands in an existing stopband and should be close to a no-op.
    'emg2speech': dict(hp=None, bp=None, notch=(50.0, 1), rate=2000, src='requested', act=True,
                       note='already band-passed from ~60 Hz, so this is near a no-op'),
}


def paper_sos(dataset: str, fs: float):
    """The filter chain a dataset's own paper specifies, as [(label, sos), ...]."""
    spec = PAPER_SPEC.get(dataset)
    if not spec:
        return []
    chain = []
    if spec.get('hp'):
        order, corner = spec['hp']
        chain.append(('hp%g' % corner,
                      butter(order, corner / (0.5 * fs), btype='high', output='sos')))
    if spec.get('bp'):
        order, lo, hi = spec['bp']
        hi = min(hi, 0.45 * fs)
        chain.append(('bp%g-%g' % (lo, hi),
                      butter(order, [lo / (0.5 * fs), hi / (0.5 * fs)],
                             btype='band', output='sos')))
    if spec.get('notch'):
        f0, kmax = spec['notch']
        for k in range(1, int(kmax) + 1):
            fk = f0 * k
            if fk >= 0.45 * fs:
                break
            b, a = iirnotch(fk, NOTCH_Q, fs)
            chain.append(('notch%g' % fk, tf2sos(b, a)))
    return chain


def condition_paper(x: np.ndarray, fs: float, dataset: str, trim_ms: float = 0.0):
    """Apply a dataset's OWN documented preprocessing. Causal, forward-only.

    The authors' implementations use zero-phase filtfilt; we use sosfilt so the pipeline
    stays streamable. Magnitude response is identical -- only phase differs -- and the
    in-band group delay of gaddy's 2 Hz high-pass is far below its 20 Hz predecessor's.
    """
    x = np.asarray(x, dtype=np.float64)
    chain = paper_sos(dataset, fs)
    y = x
    for _, sos in chain:
        y = _causal(sos, y)
    ntrim = int(round(trim_ms * fs / 1000.0))
    diag = {'dataset': dataset, 'stages': [lab for lab, _ in chain],
            'src': (PAPER_SPEC.get(dataset) or {}).get('src'), 'trimmed': 0}
    if ntrim and y.shape[-1] > 4 * ntrim:
        y = y[..., ntrim:]
        diag['trimmed'] = ntrim
    return y.astype(np.float32), diag


def _causal(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Forward-only filter, initialised from the first sample to kill the step transient."""
    zi = sosfilt_zi(sos)
    if x.ndim == 1:
        y, _ = sosfilt(sos, x, zi=zi * x[0])
        return y
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        out[c], _ = sosfilt(sos, x[c], zi=zi * x[c, 0])
    return out


def group_delay_ms(dataset: str, fs: float, at_hz=(20, 60, 100, 200, 400)):
    """Causal group delay of a dataset's chain in ms, at a few probe frequencies.

    Computed from the sos response directly: converting a Q=30 notch to transfer-function
    form and calling scipy's group_delay() warns about a near-singular denominator and
    returns garbage at low frequency, so the phase is differentiated numerically instead.
    A notch's delay is concentrated in its own narrow stopband; across the rest of the band
    only the high-pass or band-pass contributes. Inside a stopband the magnitude collapses
    and the phase derivative is meaningless, so those probes report None rather than a
    number in the thousands of ms.
    """
    from scipy.signal import sosfreqz
    total, grid, mag = None, None, None
    for _, sos in paper_sos(dataset, fs):
        w, h = sosfreqz(sos, worN=16384, fs=fs)
        omega = 2.0 * np.pi * w / fs
        gd = -np.gradient(np.unwrap(np.angle(h)), omega)      # samples
        total = gd if total is None else total + gd
        mag = np.abs(h) if mag is None else mag * np.abs(h)
        grid = w
    if total is None:
        return {}
    out = {}
    for f in at_hz:
        if f >= 0.45 * fs:
            continue
        if float(np.interp(f, grid, mag)) < 0.1:
            out[f] = None
        else:
            out[f] = float(1000.0 * np.interp(f, grid, total) / fs)
    return out
