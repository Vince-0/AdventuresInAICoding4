# Adventures In AI Coding 4
## FreeToken

**I ran Mixture-of-Experts models bigger than my 10 GB graphics card** by keeping the specialists in system RAM using FreeToken. Here is what worked, how fast, what passed a small coding quiz, and what I would actually leave running day to day.

**Series:** [AI_Coding](https://github.com/Vince-0/AI_Coding) -> [AdventuresInAICoding](https://github.com/Vince-0/AdventuresInAICoding) -> [AdventuresInAICoding2](https://github.com/Vince-0/AdventuresInAICoding2) -> [AdventuresInAICoding3](https://github.com/Vince-0/AdventuresInAICoding3) -> **#4 FreeToken**

**Upstream:** [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) · [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md) · [paper](https://arxiv.org/abs/2608.16157)

**Primer (how LLMs / MoE / KV work in plain language):** [AI Theory](https://github.com/Vince-0/AI_Theory)

---

## Key concepts

Read this first if the jargon below is new:  **[AI Theory](https://github.com/Vince-0/AI_Theory#key-concepts)**.

| Term | Full name / meaning | On this adventure |
|------|---------------------|-------------------|
| **Host RAM (WSL)** | **Host** random-access memory = system RAM inside the **Windows Subsystem for Linux** guest | **MoE fit gate** (~27 GiB). Check with `free -h` in WSL. |
| **VRAM** | **Video** random-access memory = memory on the graphics card (GPU) | 10 GB for attention + MoE cache + KV - **not** the full expert pool. |
| **Expert pool** | The full set of MoE **specialist** weight tensors | Stays in **host RAM** with FreeToken offload. |
| **MoE cache (LRU)** | **Mixture-of-Experts** GPU **cache** using **least-recently-used** eviction | Hot specialists on the card; miss -> fetch over PCIe (`offload`) or CPU. |
| **`offload` vs `hybrid`** | FreeToken **miss backends** (how a cache miss is filled) | This box prefers **`offload`** after `ft bench bw`. |
| **PCIe H2D** | **Peripheral Component Interconnect Express**, **host-to-device** copy (CPU RAM -> GPU) | Steady miss path once weights are resident (not the HDD download). |
| **KV reserve / rebuild** | **Key-Value** attention cache: **reserve** = how many pages are set aside; **rebuild** = resize live via CLI | `ft ctl cache rebuild --kv N --moe M` (trade chat length vs MoE cache). |
| **NVFP4 / MXFP4** | **NVIDIA 4-bit floating point** / **microscaling 4-bit floating point** weight formats | Why these MoEs fit ~27 GiB host RAM. |
| **`fused` (dense on FT)** | FreeToken **fused** path = **dense** model (no routed experts) | Wrong demo for host-RAM MoE offload. |
| **`ft serve` / `ft launch` / `ft ctl`** | FreeToken command-line tools: **serve** HTTP, **launch** agent stack, **ctl** control/cache | Day-to-day ops after install. |
| **MTP (vs #3)** | **Multi-token prediction** (draft several tokens ahead) | Fast **fitted** GGUFs in Adventure #3 - not mixed with FreeToken MoE-offload here. |

---

## Why

Cloud AI is someone else's computer, and credits add up. In [Adventures #3](https://github.com/Vince-0/AdventuresInAICoding3) I pushed **models that already fit** a 10 GB card to go **faster** (MTP on small GGUFs).

This round asks a different question: can I run **Mixture-of-Experts (MoE)** models whose full weight file is **larger than the graphics card**, at a speed that still feels usable?

Usual local rule: everything important must fit in **VRAM** (GPU memory). MoE only *activates* a few specialists per token, but you still **store** the whole specialist library somewhere. That library is the **expert pool** - often far bigger than 10 GB.

**FreeToken's idea:** keep that library in **host RAM** (system memory), keep a small **hot cache** of specialists on the GPU, and on a miss copy what you need over the PCIe link (`offload`). Success on a gaming PC means: "MoE larger than VRAM runs at interactive speed" - not "buy a 48 GB card."

Background: [AI Theory - sparse compute vs storage](https://github.com/Vince-0/AI_Theory#sparse-compute-vs-storage).

### Why not just llama.cpp / buy VRAM?

| Approach | Idea | Gap for this question |
|----------|------|------------------------|
| Quantize until it fits (llama.cpp / Ollama) | Shrink weights into VRAM | Huge expert pools still awkward; oversized files thrash when they do not fit RAM either |
| Dense "fit the card" servers (e.g. vLLM) | Fast serving for models that fit | Wrong tool for "35B-class MoE on 10 GB VRAM" |
| Buy more VRAM | Bigger card holds more | Expensive; ignores MoE + host RAM as the design |
| **FreeToken** | Host-RAM experts + GPU cache + bandwidth-aware misses | Built so **total MoE size >> VRAM** is normal |

This write-up is **filter + measure** on my ~27 GiB WSL RAM box - not a claim that DeepSeek-class MoEs run on a 3080.

---

## How

**PC:** Windows 11, WSL Ubuntu 24.04, RTX **3080 10 GB**, WSL `memory=28GB` (~**27 GiB** inside), FreeToken **0.1.2**, CUDA toolkit **13.2** for FreeToken (kept **12.8** for llama.cpp), spinning HDD.

1. Install FreeToken from source; point `CUDA_HOME` at the 13.2 toolkit so JIT can find `nvcc`.
2. Run `ft bench bw` to compare CPU vs PCIe bandwidth -> this box prefers **`--moe-backend offload`**.
3. Download only the weight shards I need (example: gpt-oss without `metal/*` and `original/*`).
4. Serve one model at a time: `ft serve ... --moe-backend offload --moe-cache-auto`.
5. Smoke-test with Hermes (`ft launch`), then measure **speed** and a **coding quiz** (below).
6. Compare against #3's snappy llama.cpp MTP host and a same-family Q8 GGUF baseline.

```bash
ft --version
ft bench bw --dtype nvfp4,bf16
# This box: CPU STREAM ~49.5 GB/s · PCIe H2D ~26.8 GB/s -> prefer --moe-backend offload

ft serve --model ~/LLM/models/hf/gpt-oss-20b --served-model-name gpt-oss-20b \
  --host 127.0.0.1 --port 1919 --moe-backend offload --moe-cache-auto \
  --kv-reserve-tokens 4096 --memory-ratio 0.85 --max-running-requests 1
```

**Sizing rule:** host RAM holds the expert pool (prefer roughly under ~20-23 GB NVFP4 here); VRAM holds attention + MoE cache + conversation memory (KV).

---

## Measuring

Two different scores - do not mix them:

| Name | Plain meaning | How to read it |
|------|---------------|----------------|
| **Speed (Layer B)** | How fast the model *writes* after warmup | **Tokens per second (tok/s)** on short decode. Higher = snappier typing feel. Cold load from HDD is excluded. |
| **Coding quiz (C0)** | Can it write small correct Python? | Fixed prompts -> extract code -> run checks -> **pass/fail**. Reported as **passed/total** (e.g. 19/24). |

**C0 (Python coding quiz)** - same *spirit* as Adventures #3:

- Ask the model a fixed list of short Python tasks (functions, classes, file I/O, etc.).
- Parse the code out of the reply.
- Run it against expected checks.
- Count passes. Prompts must **pin required names** (function/class names) or the checker and the model disagree and the score collapses (I saw 5/24 before pinning, then **19/24** on gpt-oss).

The full quiz scripts live in my local lab (`opencode/benchmark_python_*`) and are **not** copied into this GitHub repo. The **numbers below** are from those runs; the method is what matters for reading the tables.

**Speed (Layer B)** - short decode after the model is warm, using a small local bench script against the FreeToken OpenAI-compatible API.

**vs #3:** same idea as Hermes/python quizzes and tok/s there - so you can compare "fast small GGUF" vs "capable MoE on FreeToken."

---

## Results

### FreeToken MoE / offload (this adventure)

| Role | Model | On-disk | Speed (tok/s) | Coding quiz | Takeaway |
|------|--------|---------|---------------|-------------|----------|
| Preferred FreeToken daily driver | `openai/gpt-oss-20b` (MXFP4) | ~13 GB | **~45.5** | **19/24** | Best FT balance here; Hermes one-shot `PHASE7_OK`. Slower than #3 MTP (~135 tok/s) but much stronger on the quiz (~19/24 vs ~6-8/28). |
| Mid MoE | `nvidia/Gemma-4-26B-A4B-NVFP4` | ~18.8 GB | **~24.0** | **8/24** | Runs; weaker quiz; not my Hermes pick. |
| Headline "bigger than VRAM" MoE | `nvidia/Qwen3.6-35B-A3B-NVFP4` | ~23.5 GB | **~35.7** | deferred | Largest FT MoE I ran here; proof of the premise. Not the snappy agent host. |

### Compared to Adventures #3 (llama.cpp)

| Model | Speed / quiz | Role vs FreeToken |
|-------|--------------|-------------------|
| `Qwen3.5-4B-MTP` Q4 | ~130-148 tok/s; ~5-8/28 | **Snappy** Hermes/OpenCode feel; weaker coding quiz than gpt-oss |
| `Qwen3.5-9B` Q4 / Sushi Q4 | ~50-90 tok/s; ~2-7/28 | Mid llama.cpp path |
| `Qwen3.6-35B-A3B` Q8 GGUF | cold ~0.2 / warm ~17 tok/s | Same family as FT headline; **loses** to NVFP4 offload (~35.7) for UX on this box |

**What I would run day to day**

| Need | Pick |
|------|------|
| Snappy interactive coding/chat | `Qwen3.5-4B-MTP` (llama.cpp) - #3 |
| Stronger FreeToken / Hermes on `:1919` | `gpt-oss-20b` |
| Demo MoE larger than VRAM | `Qwen3.6-35B-A3B-NVFP4` |

FreeToken **does not mix** MTP + MoE-offload on these checkpoints. Treat as **two configs**, not one winner.

### What FreeToken claimed vs this PC

| Claim | Here |
|-------|------|
| MoE larger than VRAM via host-RAM experts | **Yes** - gpt-oss / Gemma / Qwen NVFP4 |
| Interactive speed on RTX 30-class | **Yes** - roughly 24-46 tok/s on my speed tests |
| Bandwidth-aware backend | Calibrated; **`offload` won** |
| Live KV <-> MoE cache trade | **Yes** - `ft ctl cache rebuild --kv/--moe` |
| Drop-in agent API | **Yes** - Hermes via `ft launch` |
| Huge frontier MoEs (DeepSeek / GLM / Flash-Next class) | **Out of scope** - need far more host RAM |

### Models I tried / rejected (judgment, not only success)

**Shortlist:** warmup gpt-oss; mid Gemma; Muse same size band as Qwen but **rejected**; headline Qwen NVFP4; llama.cpp Q8 as same-family baseline.

| Rejected | Why (plain) |
|----------|-------------|
| Muse-Glimmer-30B-NVFP4 | Labeled like a MoE candidate; engine treated it as **dense** (`fused`) - will not demo host-RAM experts on 10 GB |
| DeepSeek / GLM / Flash-Next class | Expert pools far beyond ~27 GiB RAM |
| Dense "27B" as an MoE demo | Wrong product for this story |
| Kimi / Moonshot class | Not on FreeToken known-good list + huge |

**Lesson:** a name on a support list is not the same as MoE architecture.

### Longer chat memory slows decode (KV vs MoE cache)

On the GPU, conversation memory (**KV**) and the **MoE expert cache** share a limited budget. Asking for a huge context without shrinking the expert cache fails. Rebuild **both**:

```bash
ft ctl cache rebuild --kv N --moe M --wait 300
```

Short-decode speed on gpt-oss as KV grows (MoE cache shrinks):

| KV target | MoE slots | Mean tok/s |
|-----------|-----------|------------|
| 4096 | 316 | **44.4** |
| 32768 | 200 | **31.4** |
| 65536 | 120 | **22.9** |

About **half** the short-decode speed from "speed config" to long context on this test. I treat **two presets**: speed (small KV, big MoE cache) vs long-chat (large KV, smaller cache).

---

## Agents (Hermes / OpenCode)

Same idea as #3: drive a local OpenAI-compatible server from an agent harness. FreeToken speaks that API; `ft launch` wires Hermes/OpenCode.

```bash
ft launch hermes --dry-run
ft launch hermes -y -- chat -q "Reply with exactly the string PHASE7_OK and nothing else." -Q --max-turns 1
# -> PHASE7_OK
```

That one-shot proves the plug-in path. Coding **quality** is the C0 quiz above; **feel** is still snappier on #3's MTP 4B host (~135 tok/s) than on gpt-oss (~45 tok/s).

```bash
ft launch opencode --dry-run
```

Less provider config pain than #3's llama.cpp `auth.json` tinkering. I did not rematch Tetris.html on FreeToken.

---

## Issues

| Symptom | What I did / learned |
|---------|----------------------|
| Minutes of cold load on HDD | Warm up before quoting tok/s |
| Bandwidth bench picks offload | Use `--moe-backend offload` here |
| HF "13 GB" model downloads ~41 GB | Exclude unused variants (`metal/*`, `original/*`) |
| Empty replies on short generations | gpt-oss reasoning budget - lower reasoning effort for smoke tests |
| C0 looked like 5/24 then 19/24 | Pin required function/class names in prompts |
| Muse "MoE" failed the story | Dense/`fused` - drop from MoE matrix |
| Huge Q8 GGUF with `-ngl 99` | Abort; even with fit tricks, NVFP4 offload won for UX |
| Rebuild KV alone -> errors | Always pass `--moe` down with `--kv` |
| mlock limits on big MoEs | Experts may be pageable - expect jitter or raise limits / add RAM |

---

## Lessons

1. **RAM** gates whether the MoE fits; **VRAM** gates cache + chat memory.
2. PCIe miss traffic is not the same problem as a slow disk.
3. Measure WSL RAM inside the guest (`free -h`).
4. Support matrix ≠ architecture (Muse).
5. Same-family Q8 GGUF can lose to NVFP4 + FreeToken offload when the file is bigger than RAM.
6. Rank **quiz score and speed together** - not tok/s alone.
7. Dual hosts beat one fake winner: snappy MTP vs capable MoE.

**Maybe later:** SSD for less waiting; more RAM for bigger MoEs; more VRAM for long chat + big expert cache together. None of those alone unlocks the largest frontier MoEs on FreeToken.

---

### How FreeToken differs from llama.cpp

llama.cpp **does** support MoE. The common path (see [DocShotgun's MoE offload guide](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)) is: put attention / dense / shared experts / KV on the GPU, park **routed experts** in host RAM with `--cpu-moe` / `--n-cpu-moe` / `-ot exps=CPU`, and optionally pin some expert layers back on GPU if VRAM remains. Placement is mostly **fixed at load time**.

FreeToken asks a narrower question: MoE **total size >> VRAM**, with a **dynamic GPU expert cache** and an explicit miss policy.

| | **llama.cpp (typical MoE offload)** | **FreeToken** |
|---|-------------------------------------|---------------|
| Expert home | Host RAM (and optionally some layers fixed on GPU) | Host RAM holds the **full expert pool** (source of truth) |
| Leftover VRAM | Often **whole expert layers** chosen at launch | A **shared LRU MoE cache** of recently routed experts |
| Placement | **Static** at load (`-ot` / `--n-cpu-moe`) | **Dynamic** - cache follows token routing |
| Miss handling | Mostly compute on CPU (or pay PCIe if you copy) | Explicit backends: **`offload`** (PCIe fill), **`cpu`**, **`hybrid`** (split by measured bandwidth) |
| Tuning | Manual `-ot` / layer counts / batch sizes | `ft bench bw` -> prefer offload vs hybrid on *this* box |
| Formats | **GGUF** ecosystem | Native **NVFP4 / MXFP4** paths (plus its own packing) |
| Elasticity | Restart / retune for big KV vs weight tradeoffs | Live `ft ctl cache rebuild` (KV vs MoE cache) |
| Maturity | Very mature server / tooling / ecosystem | Young specialized MoE server |

### Why not just use llama.cpp?

Often you **should**. Prefer llama.cpp (or Ollama / ik_llama forks) when:

- The model **fits** (or nearly fits) and you want max ecosystem maturity
- You care about **GGUF**, MTP, tooling, and long battle-testing
- You are fine with **static** expert placement and CPU-heavy decode
- FreeToken does not support the checkpoint / quant you want

Reach for FreeToken when the question is specifically: **MoE total size >> VRAM**, host RAM holds the pool, and you want interactive decode from a **dynamic GPU expert cache** (and preferably NVFP4-class weights).

On this box that showed up empirically: same-family **Qwen Q8 GGUF** on llama.cpp lost UX to **NVFP4 + FreeToken offload** (warm ~17 vs ~35 tok/s). That is not "llama.cpp cannot MoE" - it is "this format + this miss policy won on 10 GB / ~27 GiB WSL."

Dense "fit the card" servers (e.g. vLLM / SGLang) are usually the wrong default for "35B-class MoE on a 3080." They shine when the model (or shards) fit; they are not primarily "host-RAM expert pool + small consumer VRAM."

### Practical takeaway

- **llama.cpp MoE is real** - attention/KV on GPU, routed experts mostly in RAM.
- **FreeToken is not "MoE support"** - it is a **different serving architecture** for oversized MoE (cache + miss policy + elastic VRAM split).
- Default stack here: **llama.cpp for fitted / general work**; **FreeToken when you deliberately want MoE larger than the card** and can live with a younger stack and narrower model matrix.

---

## Reflections

FreeToken's consumer MoE story holds on a 3080 + ~27 GiB WSL box for gpt-oss / Gemma / Qwen NVFP4.

I still have a lot to learn. I do know which wall I hit (host RAM) and which two setups I would leave running.
