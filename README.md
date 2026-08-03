# ComfyUI MiniMax H3 Multi-Frame Keyframes

> 给 MiniMax H3（FL2VA）加上**首 N 帧 / 尾 N 帧**条件节点，突破内置
> `MiniMaxH3ImageToVideo` 只能给一张首帧 + 一张尾帧的限制。

<!-- README-I18N:START -->

**汉语** | [English](./README.en.md)

<!-- README-I18N:END -->

**状态：已实机验证可运行。** 两端锚点（frame 0、最后一帧）落在训练分布内、匹配良好；
多帧连续段（尤其 run 内部非锚点帧）属超出训练分布的实验性用法，实际效果以测试为准。

---

## 结论先讲

**代码层面：可行。** DiT 对关键帧数量和位置没有结构性限制，唯一的硬拦截是
`comfy/ldm/minimax/model.py` 里 `PackedLayout` 的一句

```python
raise ValueError("only first/last keyframe anchors are supported")
```

这是一条策略检查，不是架构限制（推导见下）。本仓库用一个最小补丁把它推广到任意帧索引。

**模型层面：超出训练分布。** 官方 FL2VA 只在 `[0]`、`[-1]`、`[0, -1]` 三种关键帧索引集合上训练：

- SGLang 服务端文档：*"For fl2va, provide one or two image conditions with role keyframe.
  The supported frame-index sets are `[0]`, `[-1]`, and `[0, -1]`."*
- Diffusers `MiniMaxH3Blocks`：关键帧只有 `image` / `last_image` 两个入口
- MiniMax 官方说明：FL2VA accepts 0, 1, or 2 images

所以「首 N 帧 / 尾 N 帧」在权重上没有对应的训练信号，属于实验性用法。节点已通过实机运行验证
**可正常工作**（首/尾段条件能正确注入并生成视频），两端锚点匹配良好；但多帧连续段（尤其视频
续写场景下的运动连贯性）不保证，效果以实际测试为准。

---

## 安装

把本仓库整体放入 ComfyUI 的 `custom_nodes/` 目录（文件夹名任意），重启 ComfyUI 即可在
`model/conditioning/minimax` 分类下看到节点 **`MiniMax H3 Multi-Frame to Video`**。

本节点复用 ComfyUI 自带的 minimax 运行时与依赖，**无需额外安装任何第三方包**。

---

## 节点与输入

节点：`MiniMax H3 Multi-Frame to Video`（`model/conditioning/minimax`）
输出：`positive`（conditioning）+ `latent`（空视频/音频 latent，含正确的时间轴长度）

### 输入一览

| 输入 | 说明 |
| --- | --- |
| `first_frames` | 从第 0 帧开始的连续帧，按顺序。多余帧从尾部截掉到合法长度 |
| `last_frames` | 以最后一帧结尾的连续帧，按顺序。多余帧从头部截掉 |
| `length` | 输出总帧数（含被条件占用的首尾段），24 fps，向上对齐到 `17k+5` |
| `prompt_frames` | Qwen 侧看到哪两帧作 `<Picture i>`。详见下节 |
| `cond_noise_aug` | 关键帧行被钉住的时间步，默认 `0.999`（越干净约束越强） |

---

### `prompt_frames`：文本侧视觉参考

这个参数控制 **Qwen 文本编码器在 prompt 里看到哪几帧作为 `<Picture i>` 视觉参考**。
它**不影响**真正喂给 DiT 的关键帧条件（keyframe cond rows）——后者永远是整个 head/tail 段
切成的 latent token，与此参数无关。

| 选项 | head 段取帧 | tail 段取帧 | 含义 |
| --- | --- | --- | --- |
| **`anchor`**（默认） | `head[:1]` → 第 0 帧（首帧） | `tail[-1:]` → 最后一帧 | 片段**两端**作参考，与官方 FL2VA 首/尾帧呈现一致，落在训练分布内 |
| **`boundary`** | `head[-1:]` → head 段末帧（紧贴生成段之前） | `tail[:1]` → tail 段首帧（紧贴生成段之后） | 取**生成（去噪）区间两道边界**的帧 |
| **`none`** | — | — | 纯文本，不插入 `<Picture>`，Qwen 看不到任何图 |

要点：

- **只改文本侧视觉提示，不改 DiT 侧条件。** 无论选哪个，DiT 拿到的 never-denoised cond rows
  都是完整的 head/tail 段（按 token 切），`prompt_frames` 纯粹决定 prompt 文本里夹带哪张图。
- **图片数 ≤ 2**：head/tail 都给时最多两张，单独一端一张，`none` 时零张——对齐官方 tokenizer
  的非 ref 模式上限（Qwen 侧最多见 2 张 `<Picture>`）。
- **默认 `anchor` 是有意的**：FL2VA 在 `[0]`/`[-1]` 上训练，文本侧看到的也是首帧 + 尾帧；
  `boundary` / `none` 会让文本视觉参考偏离训练设定，属探索性用法。
- **注意 snapping**：输入帧数会被向下取整到合法段长 `1, 5, 22, 39, ...`。例如各输入 10 帧时
  实际只取 5 帧，选 `boundary` 引用的是**第 5 帧 + 倒数第 5 帧**（而非第 10 / 倒数第 10）。

---

### 条件如何生效：不是「前 N 帧被原样钉死」

常见误解：以为输出是 `input-head + 生成-middle + input-tail` 三段拼接，head/tail 被原样钉死。

实际（来自 `comfy/ldm/minimax/model.py`）：

- 打包序列段布局为 `[text] → [cond 关键帧…] → [audio] → [video]`。
- head/tail 输入经 VAE 编码成**独立的 `cond` 段**，`img_update=False` → 去噪循环里**从不被更新**，
  是固定上下文。
- `video` 段 = **完整时间线**（`n_video = latent_t * frame_rows`）。`final_layer` 只取 `video` /
  `audio` 段输出，`cond` 段被**排除在输出之外**。
- 因此**解码出的视频只来自 `video` 段——整段每一帧都是模型生成的，包括 frame 0 与最后一帧**。
- 关键帧 cond 行被放在其对应帧的 RoPE `t` 坐标上（与 `video` 段里同位置的生成帧共享同一 `t`），
  生成帧通过注意力「看到」这个上下文，从而被引导去匹配。

所以：

- **不是硬拷贝**：输出帧是模型重新去噪生成的，head/tail 是**软条件（context）**，不是把输入像素
  复制进输出。
- **但强烈引导**：cond 行在 `cond_noise_aug = 0.999`（默认）下是 `99.9%` 干净的关键帧 latent
  （`r = aug*r + (1-aug)*noise`），约束很强；两端锚点在分布内，视觉上会高度接近输入。
- **中段自由生成**：head/tail 之间没有逐帧条件，只靠文本 + 时间连贯性生成。
- **逐像素一致不被保证**：架构上这些是生成帧 + 软约束。多帧 run 的内部帧（如 head 段 frame 1–4）
  属于超出训练分布的条件注入，贴合程度未经验证。

`cond_noise_aug` 的方向：值越低 → cond 越干净 → 抓得越紧（仍非硬拷贝）；
值越高（趋近 1）→ 条件越糊 → 越松。

---

## 为什么代码上成立

### 1. 关键帧的 rope 时间坐标本来就是一个通用公式

视频 VAE 把 `17k+5` 个像素帧压成 `5k+2` 个 latent token，
token 覆盖的帧数按 `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)` 循环。
DiT 的 rope `t` 轴就是这个划分乘以 `FRAME_RESCALE = 5/3`。

核心代码对两种锚点分别写死：

```python
if pixel_index == 0:
    cond_t = float(text_len)
elif pixel_index == frame_count - 1:
    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
```

而一个完整片段的 span 之和恰好等于 `FRAME_RESCALE * frame_count`
（每 5 个 token 覆盖 `1+4+4+4+4 = 17` 帧，尾部 2 个 token 再补 5 帧）。
代入后两个分支合并成同一个式子：

```
cond_t = text_len + FRAME_RESCALE * pixel_index
```

已在 `frame_count ∈ {5, 22, 39, 124, 141, 362}` 上逐一核对，与核心值逐位一致。
也就是说，任意帧索引的坐标都是现成的，核心只是没开放。

### 2. 打包序列本身对 cond 行数没有上限

`PackedLayout` 按 keyframe 列表逐个追加 `cond` 段；DiT `_forward` 里

- `mod_segments` 按段循环，段数任意
- `all_video_rows[~img_update] = cond_video_rows` 按序填充，条数任意
- `has_vis_cond` 只是 `any(...)`
- 注意力无 mask，最终输出层只取唯一的 `video` / `audio` 段

没有任何地方假设 cond 段是 1 个或 2 个。`ref2va` 分支已经在打包多 latent 帧的视频条件行，
机制完全一样，只是参考块走的是自己的 cursor 空间、不与目标时间轴对齐。

### 3. VAE 的分块规则决定了 N 只能取特定值

`MiniMaxH3VideoVAE.encode_temporal`：按 `clip_length = 17` 向上补帧（重复末帧），
每 17 帧出 5 个 token，最后统一丢掉 `token_drop = 3` 个 token。
只有当帧数为 `17k+5` 时，丢掉的 3 个 token 恰好是补出来的 padding；
其他帧数会**静默丢掉真实内容**。单帧走 `x.shape[2] == 1` 的静态图分支，出 1 个 token。

所以条件段长度合法值是 `1, 5, 22, 39, 56, ...`，节点内部会向下取整到最近的合法值。

### 4. 尾段的对齐是天然的

目标时间轴上「覆盖 1 帧」的 token 起点在 `17k`。
尾段长度取 `17j+5` 时起点为 `frame_count - (17j+5) = 17(m-j)`，正好落在这种边界上，
因此尾段每个 cond token 的 rope 位置与它所描述的目标 token 位置完全重合。
`N = 1` 时退化为核心的最后一帧行为（落在 `frame_count-1`，非边界，与内置节点一致）。

---

## 实现方式

### `layout.py` — 唯一的核心补丁

不复制 `PackedLayout.__init__`，而是：

1. 把所有锚点 stub 成 index 0 调用核心构造（核心会为每个 keyframe 生成一个 `cond` 段，
   结构、`img_pos`、`img_update`、`seq_len` 全部正确）
2. 事后只改写这些段在 `position_ids` 上的 `t` 列

补丁替换的是模块属性 `comfy.ldm.minimax.model.PackedLayout`，
`model_base.py` 与 DiT `_forward` 都在调用时解析，两处同时生效。
**当所有锚点都是首/尾帧时直接透传给核心**，所以内置节点与已有工作流的数值不变。

### `nodes.py` — 节点

- 首段整段送 VAE 编码一次，再按 latent token 切成 `[1,24,1,h,w]` 逐个作为 keyframe 条目。
  逐 token 切片再拼接与整段 patchify 的行顺序完全等价（`patchify_video` 是 t-major）。
- `resolved_frame_index` 填该 token 的起始像素帧号。
- 首段用拉伸（`disabled`），尾段用居中裁剪（`center`），与内置节点的策略一致。
- `cond_noise_aug` 暴露了核心已支持但没有节点设置的 `minimax_visual_cond_noise_aug`。
- `prompt_frames` 只影响 Qwen 侧 `<Picture>` 的取帧（详见上文），不改变 DiT 关键帧条件。

### 与内置节点的等价性

`first_frames` / `last_frames` 各接一张图 + `cond_noise_aug = 0.999` 时，
本节点产出的 conditioning 与 `MiniMaxH3ImageToVideo` 完全相同（走透传分支）。

---

## 代价

每个 cond latent 帧会额外增加 `(H/32) * (W/32)` 行 token。1344×768 下是 1008 行/帧，
目标视频本体是 37×1008 行；接一个 22 帧的首段（7 个 latent token）会让序列长约 19%，
注意力是 O(S²)，显存和耗时都要留余量。

---

## 已知限制

- 首段 + 尾段总帧数必须 ≤ `length`，否则报错。
- Qwen 呈现始终只放 0–2 张 `<Picture>`，段内其余帧只进 DiT 条件行，不进文本编码器。
  这是刻意保持与训练一致的选择。
- 输入帧数会被 snap 到合法段长 `1, 5, 22, 39, ...`，超出部分静默丢弃。
- 多帧连续段条件（尤其 run 内部非锚点帧）超出 FL2VA 训练分布，其视觉质量与运动连贯性属
  实验性，以实际测试为准。
