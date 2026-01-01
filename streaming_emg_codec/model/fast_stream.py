"""Low-latency streaming inference.

At one frame per step the codec is not compute-bound -- it is *launch*-bound. A single
frame costs about 6 ms on an RTX PRO 6000, and 64 channels batched together cost 6.9 ms:
64x the arithmetic for 11% more time. What that 6 ms actually buys is several hundred tiny
CUDA kernels (16 transformer layers x ~30 launches, plus the quantizer), each a few
microseconds of launch overhead with the GPU idle in between, plus three `.item()` calls in
the quantizer that flush the pipeline on every frame.

`StreamingSession` removes all three costs without changing the model or its weights:

  * **no host synchronisation** -- the quantizer's variable-bitrate bookkeeping, which
    calls `.item()` three times per frame, is replaced by a fixed all-codebooks path;
  * **static KV ring buffers** -- the reference cache grows and periodically compacts with
    a `clone()`, so its tensors move. A ring of exactly `window_size + 1` entries never
    moves and never shifts, which is also what makes capture possible;
  * **one CUDA graph** -- the whole encode+decode step is captured once and replayed, so
    the per-frame host cost is a single graph launch instead of several hundred.

The ring is exact, not an approximation. A query at stream position *p* attends to keys
*p-W..p*: that is `W+1` keys, every one of which satisfies both the causal and the
sliding-window mask, and softmax attention is permutation-invariant in the key axis. So
once `W+1` frames have been seen, "the last `W+1` entries, in whatever order the ring holds
them" is precisely the computation the reference performs. Before that the key sets genuinely
differ, so the first `W+1` frames run on the reference path verbatim and the ring is then
seeded from its cache. The ring write position is a device tensor advanced inside the graph,
so one capture serves every slot.

`tools/bench_streaming.py --verify` checks token-for-token equality against the reference
path over a real signal.

    session = StreamingSession(model, batch_size=1, device="cuda")
    for frame in stream:                      # frame: [B, C, frame_size]
        recon, codes = session.step(frame)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from streaming_emg_codec.model.attention import FLASH_ATTN_AVAILABLE
from streaming_emg_codec.model.codec import StreamingEMGCodec, StreamingState

if FLASH_ATTN_AVAILABLE:
    from flash_attn import flash_attn_with_kvcache
else:                                            # pragma: no cover - depends on the wheel
    flash_attn_with_kvcache = None


class StreamingSession:
    """Frame-synchronous encode+decode with a captured CUDA graph.

    Args:
        model: a trained `StreamingEMGCodec` in eval mode.
        batch_size: independent streams times channels per stream.
        device: CUDA device; on CPU the session runs without capture.
        cache_dtype: KV cache dtype. Defaults to what the reference path allocates
            (bfloat16 on CUDA), so the fast path is token-exact against it.
        use_cuda_graph: False keeps the other optimisations but skips capture, which is
            the useful configuration when profiling or when capture is unavailable.
        compile: run the step through `torch.compile` before capture, fusing the
            elementwise chains. Costs a one-off compile pause.
        capacity: flash backend only -- frames of KV the cache holds before its tail is
            compacted to the front. 512 is ~9 s of stream at 50 Hz, so compaction costs
            one small copy roughly every 450 frames; larger values trade memory for
            fewer compactions and change nothing else.
        backend: "flash" uses `flash_attn_with_kvcache`, the same kernel the reference
            dispatches to on a machine with flash-attn, so the fast path is token-exact
            against it. "sdpa" uses torch SDPA over a ring buffer and is the portable
            fallback. "auto" picks flash when the wheel is importable.
        exact_mask: sdpa backend only -- pass the reference's (all-true) attention mask. It costs a little
            speed and pins SDPA to the same kernel the reference dispatches to. With
            False, SDPA may choose a fused kernel whose reduction order differs in the
            last bits, which can flip a token sitting on a quantizer boundary.
    """

    def __init__(
        self,
        model: StreamingEMGCodec,
        batch_size: int = 1,
        device: torch.device | str = "cuda",
        cache_dtype: torch.dtype | None = None,
        use_cuda_graph: bool = True,
        exact_mask: bool = True,
        compile: bool = False,
        backend: str = "auto",
        capacity: int = 512,
    ):
        self.model = model.eval()
        self.device = torch.device(device)
        self.batch = int(batch_size)
        self.frame_size = model.frame_size
        self.exact_mask = exact_mask
        if cache_dtype is None:
            cache_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.cache_dtype = cache_dtype

        enc_cfg, dec_cfg = model.config.encoder, model.config.decoder
        if enc_cfg.window_size <= 0 or dec_cfg.window_size <= 0:
            raise ValueError("StreamingSession requires a bounded attention window")
        self.enc_window = enc_cfg.window_size
        self.dec_window = dec_cfg.window_size
        self.n_codebooks = model.config.rvq.num_codebooks
        self.embedding_dim = model.config.rvq.embedding_dim

        if backend == "auto":
            backend = "flash" if (FLASH_ATTN_AVAILABLE and self.device.type == "cuda") else "sdpa"
        if backend == "flash" and not (FLASH_ATTN_AVAILABLE and self.device.type == "cuda"):
            raise ValueError("backend='flash' requires flash-attn and CUDA")
        if backend not in ("flash", "sdpa"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend

        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self.capacity = int(capacity)
        # Inductor fuses the elementwise chains (RoPE, residuals, SiLU, casts), which are
        # ~40% of the captured step's kernels. Off by default: it adds a compile pause on
        # the first steady-state frame and its kernels are not the reference's.
        self._compile = bool(compile)
        self._compute_fn = self._compute
        self.reset()

    # ------------------------------------------------------------------ public API

    def reset(self) -> None:
        """Start a new stream: drop all context and re-enter the warmup phase."""
        self._n = 0
        self._ref_state = StreamingState()
        self._graph = None
        self._ready = False
        self._warm_frames = max(self.enc_window, self.dec_window) + 1

    @property
    def warmup_frames(self) -> int:
        """Frames served by the reference path before the fast path takes over."""
        return self._warm_frames

    @torch.no_grad()
    def step(self, frame: torch.Tensor):
        """One frame in, one frame out.

        Args:
            frame: [batch, channels, frame_size]; batch*channels must equal `batch_size`.
        Returns:
            (reconstruction [B, C, frame_size], indices [B, C, 1, n_codebooks])
        """
        if frame.shape[-1] != self.frame_size:
            raise ValueError(f"expected {self.frame_size} samples per frame, got {frame.shape[-1]}")
        b, c = frame.shape[0], frame.shape[1]
        if b * c != self.batch:
            raise ValueError(f"batch*channels ({b * c}) does not match session batch {self.batch}")

        if not self._ready:
            recon, idx, self._ref_state = self.model.streaming_step(
                frame, state=self._ref_state, n_codebooks=None
            )
            self._n += 1
            if self._n == self._warm_frames:
                self._seed_from_reference()
            return recon, idx

        if self.backend == "flash" and self._seq_host >= self.capacity - 1:
            self._compact_flash()
        self._in.copy_(frame.reshape(self.batch, 1, self.frame_size))
        if self.use_cuda_graph:
            if self._graph is None:
                self._capture()
            self._graph.replay()
        else:
            self._forward()
        self._n += 1
        if self.backend == "flash":
            self._seq_host += 1
        return (self._recon.reshape(b, c, self.frame_size),
                self._idx.reshape(b, c, 1, self.n_codebooks))

    # ------------------------------------------------------- warmup -> ring handoff

    def _seed_from_reference(self) -> None:
        """Copy the reference caches into static buffers and allocate the step buffers."""
        if self.backend == "flash":
            return self._seed_flash()
        dev, dt = self.device, self.cache_dtype
        self._rings, self._slot = {}, {}
        for tag, caches, window in (
            ("enc", self._ref_state.encoder_cache, self.enc_window),
            ("dec", self._ref_state.decoder_cache, self.dec_window),
        ):
            size = window + 1
            rings = []
            for cache in caches:
                ring = torch.zeros(self.batch, size, 2, cache.n_heads, cache.head_dim,
                                   device=dev, dtype=dt)
                # the reference holds `filled` entries; the last `size` of them are exactly
                # the ones a steady-state query may attend to
                ring.copy_(cache.kv_cache[:, cache.filled - size:cache.filled].to(dt))
                rings.append(ring)
            self._rings[tag] = rings
            # the next write evicts the oldest entry, which now sits at slot 0.
            # device-resident so the graph can advance it between replays
            self._slot[tag] = torch.zeros(1, device=dev, dtype=torch.long)

        # absolute stream position for RoPE, advanced inside the graph
        self._rope_pos = torch.tensor(float(self._n), device=dev, dtype=torch.float32)
        if self._compile:
            self._compute_fn = torch.compile(self._compute, mode="max-autotune-no-cudagraphs",
                                             dynamic=False)
        self._in = torch.zeros(self.batch, 1, self.frame_size, device=dev, dtype=torch.float32)
        if self.exact_mask:
            self._enc_mask = torch.ones(1, 1, 1, self.enc_window + 1, device=dev, dtype=torch.bool)
            self._dec_mask = torch.ones(1, 1, 1, self.dec_window + 1, device=dev, dtype=torch.bool)
        else:
            self._enc_mask = self._dec_mask = None
        self._ready = True

    def _seed_flash(self) -> None:
        """Seed flash-attn's own KV cache from the reference caches.

        flash_attn_with_kvcache appends the new key and value at `cache_seqlens` and
        attends in one kernel, so there is no ring: the cache is indexed by absolute
        stream position and `cache_seqlens` is a device tensor the graph advances. The
        cost is that the cache is finite -- every `capacity - (W+1)` frames the tail is
        compacted to the front, which is one copy roughly once a minute at 50 Hz and
        leaves the captured graph untouched because the buffers never move.
        """
        dev, dt = self.device, self.cache_dtype
        self._kc, self._vc = {}, {}
        for tag, caches in (("enc", self._ref_state.encoder_cache),
                            ("dec", self._ref_state.decoder_cache)):
            ks, vs = [], []
            for cache in caches:
                k = torch.zeros(self.batch, self.capacity, cache.n_heads, cache.head_dim,
                                device=dev, dtype=dt)
                v = torch.zeros_like(k)
                n = cache.filled
                k[:, :n].copy_(cache.kv_cache[:, :n, 0].to(dt))
                v[:, :n].copy_(cache.kv_cache[:, :n, 1].to(dt))
                ks.append(k); vs.append(v)
            self._kc[tag], self._vc[tag] = ks, vs
        filled = self._ref_state.encoder_cache[0].filled
        self._seqlens = torch.full((self.batch,), filled, device=dev, dtype=torch.int32)
        self._seq_host = filled          # host mirror; avoids a per-frame device read
        self._rope_pos = torch.tensor(float(self._n), device=dev, dtype=torch.float32)
        self._in = torch.zeros(self.batch, 1, self.frame_size, device=dev, dtype=torch.float32)
        self._enc_mask = self._dec_mask = None
        if self._compile:
            self._compute_fn = torch.compile(self._compute, mode="max-autotune-no-cudagraphs",
                                             dynamic=False)
        self._ready = True

    def _compact_flash(self) -> None:
        """Slide the last W+1 entries to the front so the finite cache never overruns."""
        keep = max(self.enc_window, self.dec_window) + 1
        n = self._seq_host
        for tag in ("enc", "dec"):
            for k, v in zip(self._kc[tag], self._vc[tag]):
                k[:, :keep] = k[:, n - keep:n].clone()
                v[:, :keep] = v[:, n - keep:n].clone()
                k[:, keep:].zero_(); v[:, keep:].zero_()
        self._seqlens.fill_(keep)
        self._seq_host = keep

    def _attend_flash(self, layer, x, k_cache, v_cache, window):
        attn = layer.attn
        qkv = attn.qkv(x).view(self.batch, 1, 3, attn.n_heads, attn.head_dim)
        qkv = self._rope(qkv, attn.rotary.inv_freq).to(self.cache_dtype)
        ctx = flash_attn_with_kvcache(
            qkv[:, :, 0].contiguous(), k_cache, v_cache,
            k=qkv[:, :, 1].contiguous(), v=qkv[:, :, 2].contiguous(),
            cache_seqlens=self._seqlens,
            softmax_scale=attn.softmax_scale, causal=True,
            window_size=(window, 0),
        )
        ctx = ctx.contiguous().view(self.batch, 1, attn.n_embd).to(x.dtype)
        return attn.out(ctx)

    # -------------------------------------------------------------- the fast kernel

    def _rope(self, qkv: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        # reference: outer(arange(offset, offset+1), inv_freq).view(1, 1, 1, 1, -1)
        freqs = (self._rope_pos * inv_freq).view(1, 1, 1, 1, -1)
        cos, sin = freqs.cos().to(qkv.dtype), freqs.sin().to(qkv.dtype)
        first, second = qkv[..., 0::2], qkv[..., 1::2]
        return torch.stack((first * cos - second * sin,
                            first * sin + second * cos), dim=-1).flatten(-2)

    def _attend(self, layer, x, ring, slot, mask):
        attn = layer.attn
        qkv = attn.qkv(x).view(self.batch, 1, 3, attn.n_heads, attn.head_dim)
        qkv = self._rope(qkv, attn.rotary.inv_freq).to(self.cache_dtype)
        ring.index_copy_(1, slot, qkv[:, :, 1:3])         # evict oldest, store newest
        q = qkv[:, :, 0].transpose(1, 2)                   # [B, H, 1, D]
        k = ring[:, :, 0].transpose(1, 2)                  # [B, H, W+1, D]
        v = ring[:, :, 1].transpose(1, 2)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0,
                                             scale=attn.softmax_scale)
        ctx = ctx.transpose(1, 2).contiguous().view(self.batch, 1, attn.n_embd).to(x.dtype)
        return attn.out(ctx)

    def _stack(self, stack, x, tag, mask):
        if self.backend == "flash":
            window = self.enc_window if tag == "enc" else self.dec_window
            for layer, k, v in zip(stack.layers, self._kc[tag], self._vc[tag]):
                x = x + self._attend_flash(layer, layer.norm1(x), k, v, window)
                x = x + layer.ffn(layer.norm2(x))
            return stack.norm(x)
        slot = self._slot[tag]
        for layer, ring in zip(stack.layers, self._rings[tag]):
            x = x + self._attend(layer, layer.norm1(x), ring, slot, mask)
            x = x + layer.ffn(layer.norm2(x))
        return stack.norm(x)

    def _rvq_encode(self, x: torch.Tensor) -> torch.Tensor:
        """All codebooks, no host sync. Same arithmetic as ResidualVectorQuantizer.encode."""
        residual = x.reshape(-1, self.embedding_dim)
        out = []
        for vq in self.model.rvq.vqs:
            cb = vq.codebook.weight
            d = (residual.pow(2).sum(dim=1, keepdim=True)
                 - 2 * residual @ cb.t()
                 + cb.pow(2).sum(dim=1).unsqueeze(0))
            i = d.argmin(dim=1)
            residual = residual - vq.codebook(i)
            out.append(i)
        return torch.stack(out, dim=-1).view(self.batch, 1, self.n_codebooks)

    def _rvq_decode(self, indices: torch.Tensor) -> torch.Tensor:
        vqs = self.model.rvq.vqs
        out = vqs[0].codebook(indices[:, :, 0])
        for idx in range(1, self.n_codebooks):
            out = out + vqs[idx].codebook(indices[:, :, idx])
        return out

    def _compute(self):
        """Encode+decode one frame. Pure apart from the ring writes, so it can be compiled."""
        m = self.model
        h = self._stack(m.encoder, m.linear_in(self._in), "enc", self._enc_mask)
        idx = self._rvq_encode(m.rvq_in(h))
        h = self._stack(m.decoder, m.rvq_out(self._rvq_decode(idx)), "dec", self._dec_mask)
        return m.linear_out(h).reshape(self.batch, 1, self.frame_size), idx

    def _advance(self) -> None:
        """Move the stream on by one frame, in place, so a graph replay is self-advancing."""
        self._rope_pos += 1.0
        if self.backend == "flash":
            self._seqlens.add_(1)
            return
        for tag, window in (("enc", self.enc_window), ("dec", self.dec_window)):
            self._slot[tag].add_(1).remainder_(window + 1)

    def _forward(self) -> None:
        self._recon, self._idx = self._compute_fn()
        self._advance()

    # ------------------------------------------------------------- graph capture

    def _state_snapshot(self):
        if self.backend == "flash":
            return ("flash",
                    {t: [k.to("cpu", copy=True) for k in ks] for t, ks in self._kc.items()},
                    {t: [v.to("cpu", copy=True) for v in vs] for t, vs in self._vc.items()},
                    self._seqlens.cpu(), self._rope_pos.cpu())
        return ("sdpa",
                {t: [r.to("cpu", copy=True) for r in rs] for t, rs in self._rings.items()},
                {t: s.cpu() for t, s in self._slot.items()},
                self._rope_pos.cpu())

    def _state_restore(self, snap) -> None:
        if snap[0] == "flash":
            _, ks, vs, seq, pos = snap
            for tag in ks:
                for dst, src in zip(self._kc[tag], ks[tag]):
                    dst.copy_(src)
                for dst, src in zip(self._vc[tag], vs[tag]):
                    dst.copy_(src)
            self._seqlens.copy_(seq)
            self._rope_pos.copy_(pos)
            return
        _, rings, slots, pos = snap
        for tag, rs in rings.items():
            for dst, src in zip(self._rings[tag], rs):
                dst.copy_(src)
        for tag, s in slots.items():
            self._slot[tag].copy_(s)
        self._rope_pos.copy_(pos)

    def _capture(self) -> None:
        """Capture one steady-state step.

        Both the side-stream warmup and the capture itself write to the session's static
        buffers, so the stream position is snapshotted first and restored afterwards --
        otherwise the first replay would compute a frame several positions ahead.
        """
        snap = self._state_snapshot()
        torch.cuda.synchronize()

        # warm the allocator and cuBLAS workspaces on a side stream before capturing
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._forward()
        torch.cuda.synchronize()

        self._state_restore(snap)
        self._graph = graph
