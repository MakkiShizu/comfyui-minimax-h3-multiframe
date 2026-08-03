<!-- README-I18N:START -->

[汉语](./README.md) | **English**

<!-- README-I18N:END -->

# ComfyUI MiniMax H3 Multi-Frame Keyframes

> Add **first-N / last-N frame** conditioning to MiniMax H3 (FL2VA), removing the built-in
> `MiniMaxH3ImageToVideo` limit of one first frame + one last frame.

**Status: verified working in practice.** The two end anchors (frame 0, last frame) fall inside the
training distribution and match well; multi-frame continuous segments (especially non-anchor frames
inside a run) are experimental usage outside the training distribution — actual results depend on testing.

---

## Conclusion up front

**Feasible at the code level.** The DiT has no structural limit on the number or position of keyframes;
the only hard block is one line in `PackedLayout` inside `comfy/ldm/minimax/model.py`:

```python
raise ValueError("only first/last keyframe anchors are supported")
```

This is a policy check, not an architectural limit (derivation below). This repo generalizes it to
arbitrary frame indices with a minimal patch.

**Model level: outside the training distribution.** Official FL2VA was only trained on three keyframe
index sets: `[0]`, `[-1]`, `[0, -1]`:

- SGLang server docs: *"For fl2va, provide one or two image conditions with role keyframe.
  The supported frame-index sets are `[0]`, `[-1]`, and `[0, -1]`."*
- Diffusers `MiniMaxH3Blocks`: keyframes have only two entry points, `image` / `last_image`
- MiniMax official: FL2VA accepts 0, 1, or 2 images

So "first N / last N frames" has no corresponding training signal in the weights and is experimental
usage. The node has been verified in practice to **work normally** (head/tail segment conditions are
injected correctly and the video is generated), and the two end anchors match well; however, multi-frame
continuous segments (especially motion coherence in video-continuation scenarios) are not guaranteed —
results depend on actual testing.

---

## Installation

Place this whole repo into ComfyUI's `custom_nodes/` directory (folder name arbitrary), restart ComfyUI,
and the node **`MiniMax H3 Multi-Frame to Video`** appears under the `model/conditioning/minimax` category.

This node reuses ComfyUI's built-in minimax runtime and dependencies — **no additional third-party
packages need to be installed**.

---

## Node and inputs

Node: `MiniMax H3 Multi-Frame to Video` (`model/conditioning/minimax`)
Outputs: `positive` (conditioning) + `latent` (empty video/audio latent with the correct timeline length)

### Input overview

| Input | Description |
| --- | --- |
| `first_frames` | Consecutive frames starting from frame 0, in order. Excess frames are trimmed from the tail down to a legal length |
| `last_frames` | Consecutive frames ending on the last frame, in order. Excess frames are trimmed from the head |
| `length` | Total output frame count (including the head/tail segments occupied by conditioning), 24 fps, rounded up to `17k+5` |
| `prompt_frames` | Which two frames Qwen sees as `<Picture i>`. See section below |
| `cond_noise_aug` | Timestep the keyframe rows are pinned at, default `0.999` (cleaner = stronger constraint) |

---

### `prompt_frames`: text-side visual reference

This parameter controls **which frames Qwen's text encoder sees as `<Picture i>` visual references
inside the prompt**. It does **not** affect the actual keyframe conditioning fed to the DiT
(keyframe cond rows) — those are always the full head/tail segment split into latent tokens,
independent of this parameter.

| Option | head segment frame | tail segment frame | Meaning |
| --- | --- | --- | --- |
| **`anchor`** (default) | `head[:1]` → frame 0 (first frame) | `tail[-1:]` → last frame | The **two ends** of the segment as reference, matching the official FL2VA first/last frame presentation and inside the training distribution |
| **`boundary`** | `head[-1:]` → last frame of head segment (right before the generated segment) | `tail[:1]` → first frame of tail segment (right after the generated segment) | Takes the two frames on the **boundaries of the generated (denoised) region** |
| **`none`** | — | — | Plain text, no `<Picture>` inserted, Qwen sees no image |

Points:

- **Only changes the text-side visual hint, not the DiT-side condition.** Regardless of the choice, the
  never-denoised cond rows the DiT receives are the full head/tail segment (split per token);
  `prompt_frames` purely decides which image is embedded in the prompt text.
- **At most 2 images**: both head and tail given → at most two; only one end → one; `none` → zero —
  aligned with the official tokenizer's non-ref-mode cap (Qwen sees at most 2 `<Picture>`).
- **Default `anchor` is intentional**: FL2VA was trained on `[0]`/`[-1]`, and the text side also sees the
  first + last frame; `boundary` / `none` deviate the text visual reference from training, exploratory usage.
- **Watch the snapping**: input frame count is rounded down to a legal segment length
  `1, 5, 22, 39, ...`. For example, with 10 frames each on both ends only 5 are actually used; choosing
  `boundary` then references the **5th frame + 5th-from-last** (not the 10th / 10th-from-last).

---

### How conditioning takes effect: not "first N frames pinned"

Common misconception: the output is `input-head + generated-middle + input-tail` spliced together, with
head/tail pinned verbatim.

Reality (from `comfy/ldm/minimax/model.py`):

- Packed-sequence segment layout is `[text] → [cond keyframes…] → [audio] → [video]`.
- head/tail inputs are VAE-encoded into **separate `cond` segments** with `img_update=False` →
  **never updated** in the denoising loop, fixed context.
- The `video` segment = the **full timeline** (`n_video = latent_t * frame_rows`). `final_layer` only
  takes the `video` / `audio` segment outputs; the `cond` segment is **excluded from the output**.
- Therefore **the decoded video comes only from the `video` segment — every frame of the whole clip is
  generated by the model, including frame 0 and the last frame**.
- The keyframe cond rows are placed at the RoPE `t` coordinate of their corresponding frame (sharing the
  same `t` as the generated frame at that position in the `video` segment); the generated frame "sees"
  this context through attention and is thus guided to match.

So:

- **Not a hard copy**: the output frames are re-denoised by the model; head/tail are **soft conditions
  (context)**, not pixel copies of the input pasted into the output.
- **But strongly guided**: at `cond_noise_aug = 0.999` (default) the cond rows are `99.9%` clean keyframe
  latents (`r = aug*r + (1-aug)*noise`), a strong constraint; the two end anchors are in-distribution and
  will visually closely resemble the input.
- **Middle is freely generated**: between head and tail there is no per-frame condition, only text +
  temporal coherence.
- **Pixel-exact equality is not guaranteed**: architecturally these are generated frames + soft
  constraints. Interior frames of a multi-frame run (e.g. head segment frames 1–4) are condition injections
  outside the training distribution, and their fidelity is unverified.

`cond_noise_aug` direction: lower value → cleaner cond → tighter grip (still not a hard copy);
higher value (toward 1) → blurrier cond → looser.

---

## Why it works in code

### 1. The keyframe rope time coordinate is already a general formula

The video VAE compresses `17k+5` pixel frames into `5k+2` latent tokens, each token covering frames
cycling through `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`. The DiT's rope `t` axis is this partition times
`FRAME_RESCALE = 5/3`.

Core hardcodes the two anchors:

```python
if pixel_index == 0:
    cond_t = float(text_len)
elif pixel_index == frame_count - 1:
    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
```

The sum of spans of a full clip equals exactly `FRAME_RESCALE * frame_count`
(every 5 tokens cover `1+4+4+4+4 = 17` frames, the trailing 2 tokens add 5 more frames).
Substituting merges both branches into one formula:

```
cond_t = text_len + FRAME_RESCALE * pixel_index
```

Verified bit-for-bit against core values for `frame_count ∈ {5, 22, 39, 124, 141, 362}`.
In other words, the coordinate for any frame index is already there; core just didn't expose it.

### 2. The packed sequence has no cap on the number of cond rows

`PackedLayout` appends a `cond` segment per keyframe; in DiT `_forward`:

- `mod_segments` loops per segment, arbitrary count
- `all_video_rows[~img_update] = cond_video_rows` fills in order, arbitrary count
- `has_vis_cond` is just `any(...)`
- attention has no mask; the final output layer takes only the single `video` / `audio` segment

Nothing assumes the cond segment count is 1 or 2. The `ref2va` branch already packs multiple latent-frame
video condition rows with exactly the same mechanism, only the reference block uses its own cursor space
and is not aligned to the target timeline.

### 3. The VAE chunking rule restricts N to specific values

`MiniMaxH3VideoVAE.encode_temporal`: pads up to a multiple of `clip_length = 17` (repeating the last
frame), emits 5 tokens per 17 frames, then uniformly drops `token_drop = 3` tokens. Only when the frame
count is `17k+5` do the 3 dropped tokens happen to be exactly the padded padding; other frame counts
**silently drop real content**. A single frame takes the `x.shape[2] == 1` still-image branch and yields
1 token.

So legal condition segment lengths are `1, 5, 22, 39, 56, ...`, and the node rounds down to the nearest
legal value internally.

### 4. Tail-segment alignment is natural

On the target timeline, the token covering "1 frame" starts at `17k`. When the tail length is `17j+5`,
its start is `frame_count - (17j+5) = 17(m-j)`, landing exactly on such a boundary, so each cond token's
rope position of the tail segment coincides exactly with the target token position it describes.
`N = 1` degenerates to the core's last-frame behavior (lands on `frame_count-1`, not a boundary,
consistent with the built-in node).

---

## Implementation

### `layout.py` — the only core patch

Instead of copying `PackedLayout.__init__`:

1. Stub all anchors to index 0 and call the core constructor (core generates one `cond` segment per
   keyframe, with correct structure, `img_pos`, `img_update`, `seq_len`)
2. Afterwards only rewrite the `t` column of these segments in `position_ids`

The patch replaces the module attribute `comfy.ldm.minimax.model.PackedLayout`; both `model_base.py` and
DiT `_forward` resolve it at call time, so both take effect. **When all anchors are first/last frames it
passes straight through to the core**, so the built-in node and existing workflows keep their exact numbers.

### `nodes.py` — the node

- The head segment is encoded by the VAE once as a whole, then split per latent token into
  `[1,24,1,h,w]` entries used one-by-one as keyframe items. Splitting per token and re-concatenating is
  row-order equivalent to patching the whole segment (`patchify_video` is t-major).
- `resolved_frame_index` holds the starting pixel frame number of that token.
- The head uses stretch (`disabled`), the tail uses center crop (`center`), consistent with the built-in
  node's strategy.
- `cond_noise_aug` exposes the core-supported-but-node-unset `minimax_visual_cond_noise_aug`.
- `prompt_frames` only affects which frames Qwen takes as `<Picture>` (see above), and does not change
  the DiT keyframe condition.

### Equivalence with the built-in node

With `first_frames` / `last_frames` each given one image + `cond_noise_aug = 0.999`, this node produces
conditioning identical to `MiniMaxH3ImageToVideo` (taking the pass-through branch).

---

## Cost

Each cond latent frame adds `(H/32) * (W/32)` token rows. At 1344×768 that is 1008 rows/frame; the target
video body is 37×1008 rows. Attaching a 22-frame head segment (7 latent tokens) grows the sequence by
~19%; attention is O(S²), so leave headroom for VRAM and time.

---

## Known limitations

- Head + tail total frame count must be ≤ `length`, otherwise it errors.
- Qwen presentation always uses 0–2 `<Picture>`; the rest of the segment's frames go only into the DiT
  condition rows, not the text encoder. This is a deliberate choice to stay consistent with training.
- Input frame count is snapped to a legal segment length `1, 5, 22, 39, ...`; the excess is silently
  dropped.
- Multi-frame continuous segment conditioning (especially non-anchor interior frames of a run) is
  outside the FL2VA training distribution; its visual quality and motion coherence are experimental —
  results depend on actual testing.
