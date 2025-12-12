# SpQR: A Sparse‑Quantized Representation for Near‑Lossless LLM Weight Compression

## 0. Metadata
- **Full title**: SpQR: A Sparse‑Quantized Representation for Near‑Lossless LLM Weight Compression
- **Authors**: Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, Dan Alistarh
- **Venue / year**: ICLR 2024
- **Links**:
  - ArXiv: https://arxiv.org/abs/2306.03078
  - OpenReview: https://openreview.net/forum?id=Q1u25ahSuy
  - Code: https://github.com/Vahe1994/SpQR (mirrored under `context/refcode/SpQR/`)
- **Keywords**: LLMs, post‑training quantization, outliers, sparse‑quantized formats, GPTQ

## 1. TL;DR
- **Problem**: 3–4 bit PTQ gives big memory wins but hurts accuracy, especially for smaller deployable LLMs.
- **Idea**: Keep almost all weights in low‑bit grouped quantization, but store a tiny set of highly sensitive weights (“outliers”) in fp16 sparsely.
- **Result**: Near‑lossless perplexity (<1% gap vs fp16) at ~4.6–4.7 bits/parameter, plus 20–30% faster token generation on GPU.

> "Quantization down to 3-4 bits per parameter usually leads to moderate-to-high accuracy losses... To address this accuracy issue, we introduce... SpQR... [which] works by identifying and isolating outlier weights... and storing them in higher precision..." (paper-source/primary/tex/main.tex:126-132)  
> "SpQR... yields faster inference than 16-bit baselines... 20-30% faster for LLM generation..." (paper-source/primary/tex/main.tex:175-176)

## 2. Motivation: why near‑lossless low‑bit matters
- **Edge/consumer deployment needs <4–5 bpp** to fit 7–65B LLMs on single GPUs / laptops / phones.
- **Generative setting is fragile**: small quantization errors can compound over autoregressive decoding.

> "By compressing such LLMs via quantization to 3-4 bits per parameter, they can fit into memory-limited devices..." (paper-source/primary/tex/main.tex:126-128)  
> "Since LLM generation is sequential... small relative errors can accumulate and lead to severely corrupted outputs." (paper-source/primary/tex/main.tex:150-151)

## 3. Related PTQ methods & their pitfalls

### 3.1 RTN / rounding‑based PTQ
- **What it does**: direct round‑to‑nearest weights, often with groupwise scales.
- **Pitfall**: needs large groups to save metadata, but then errors from sensitive weights spread; accuracy drops remain large.

> "Early work... used direct rounding of weights to the nearest quantization level, while customizing the quantization granularity (i.e., group size) to trade off space for increased accuracy." (paper-source/primary/tex/main.tex:221-223)  
> "4-bit quantization is an optimal point for round-to-nearest-based methods..." (paper-source/primary/tex/main.tex:227-229)

### 3.2 GPTQ
- **What it does**: Hessian‑aware greedy solver to minimize layer output error.
- **Pitfall**: still non‑trivial loss at 3–4 bits unless groups are very small (metadata heavy); doesn’t explicitly isolate weight outliers.

> "GPTQ... proposed a higher-accuracy approach... works via an approximate large-scale solver..." (paper-source/primary/tex/main.tex:224-225)  
> "One common drawback of existing methods is that the accuracy loss relative to the original model is still significant... especially... 7-13B..." (paper-source/primary/tex/main.tex:231-232)

### 3.3 LLM.int8 / activation outlier handling
- **What it does**: isolates *activation / feature* outliers to higher precision.
- **Pitfall**: addresses input‑feature outliers, not the scattered outlier **weights** that dominate low‑bit weight error.

> "LLM.int8() suggested isolating 'outlier features' which would be quantized separately to higher bit-width." (paper-source/primary/tex/main.tex:222)  
> "We ... demonstrate that similar outliers occur in the weights, for particular output hidden dimensions." (paper-source/primary/tex/main.tex:173)

### 3.4 SparseGPT / sparsity + quantization
- **What it does**: joint sparsification and quantization of remaining weights.
- **Pitfall**: targets medium sparsities, still suffers accuracy loss vs fp16 at similar bit budgets.

> "SparseGPT presented an approach to jointly sparsify LLM weights... together with quantization..." (paper-source/primary/tex/main.tex:230-231)

## 4. Observations that led to SpQR
- **Outlier weights are structured + unstructured**: sensitivity maps show rows, columns, heads, rotary patterns, plus scattered singles.
- **Tiny fraction causes most error**: ~1% weights can contribute >75% quantization error.

> "Sensitive weights... are not random but have particular structures... row outliers... column outliers... sensitive attention heads... rotary embedding pattern... unstructured outliers." (paper-source/primary/tex/method.tex:3,16-31)  
> "In some cases, 1% of the weights account for over 75% of the total quantization error." (paper-source/primary/tex/method.tex:61-63)

## 5. SpQR method & representation

### 5.1 High‑level idea
- **Hybrid format**:
  - **Base**: low‑bit (3–4 bit) grouped weights.
  - **Residual**: sparse fp16 outlier weights.
  - **Metadata**: scales/zeros, also quantized (bilevel).

> "SpQR works by identifying and isolating outlier weights... while compressing all other weights to 3-4 bits..." (paper-source/primary/tex/main.tex:129-130)  
> "We implement... very small group size... and quantize the quantization scales themselves to a 3-bit representation." (paper-source/primary/tex/main.tex:158-160)

### 5.2 Outlier identification
- **Sensitivity criterion**: for each small group, estimate leave‑one‑out error; mark weights whose exemption reduces error by ≥τ.
- **Budget**: τ tuned so outliers stay under ~1% model‑wide.

> "Outlier detection... isolate weights whose direct quantization has outsize impact... pick a threshold τ... usually around 1% of weights." (paper-source/primary/tex/method.tex:67-74)  
> "A weight can be chosen to be an outlier... if GPTQ can employ this weight to compensate errors from many other weights." (paper-source/primary/tex/method.tex:79)

- **Reference code**: per‑group leave‑one‑out error and mask.  
> "group_weight = weight[:, column_index : column_index + groupsize]" (context/refcode/SpQR/spqr_engine.py:141-145)  
> "loo_quantization_error_sq = get_leave_one_out_error(...); likely_unstructured_outlier_mask = (loo_quantization_error_sq > unstructured_outlier_threshold)" (context/refcode/SpQR/spqr_engine.py:153-161)

### 5.3 Bilevel / grouped quantization
- **First‑order groups (β₁)**: extremely small groups of consecutive weights share one scale/zero.  
> "groupwise quantization with extremely small groups, typically of β₁=8-32 weights... separate quantization scale and zero-point." (paper-source/primary/tex/method.tex:53-55)

- **Second‑order groups (β₂)**: quantize the *scales/zeros* themselves in blocks of β₂ consecutive entries to reduce metadata cost.  
> "We group groupwise statistics from β₂=16 consecutive values and quantize them together..." (paper-source/primary/tex/method.tex:58-59)  
> "fit_statistics... fit_quantizer(s, β₂)... quantize(s, s_s, s_z)" (paper-source/primary/tex/method.tex:172-175)

- **Implementation geometry (important for mental model)**:
  - **Weights** are grouped as full‑height vertical strips of width `groupsize` along the input/column axis.  
    > "group_weight = weight[:, column_index : column_index + groupsize]" (context/refcode/SpQR/spqr_engine.py:141-145)
  - **Stats** are grouped by reshaping consecutive scale/zero entries into `qq_groupsize` blocks.  
    > "scale_groups = self.scale.reshape(-1, self.qq_groupsize)... zero_groups = self.zero.reshape(-1, self.qq_groupsize)" (context/refcode/SpQR/quant_groups.py:103-114)
  - In the optimized kernel these correspond to tiles of β₁ output rows × β₂ input columns.  
    > "beta1: Tile width; beta2: Tile height" (context/refcode/SpQR/inference_lib/src/spqr_quant/inference.py:77-83)

### 5.4 Storage format
- **Stored per layer**:
  1. Quantized non‑outlier weights (b_w bits).
  2. Quantized 1st‑order scales/zeros + 2nd‑order stats.
  3. CSR‑like sparse outliers.

> "Representation consists of (1) quantized weights, (2) first level... second level quantization statistics, and (3) the CSR outlier indices and values." (paper-source/primary/tex/method.tex:196-197)

- **Outlier CSR layout**:
  - fp16 value + u16 column index per outlier.
  - u32 row‑prefix counts (CSR row_ptr).  
> "store two scalars: the 16-bit weight value and the 16-bit column index... for each row... a 32-bit number..." (paper-source/primary/tex/method.tex:218-222)

### 5.5 Inference
- **Dense + sparse decomposition**:  
  - Dense low‑bit matmul over base weights.  
  - Sparse matmul over outliers; sum results.
- **Custom sparse algorithm** uses CSR ordering and GPU shared memory for load‑balanced, mostly contiguous access.

> "combine this sparse algorithm together with a dense-quantized matrix multiplication for 3-4 bit weights." (paper-source/primary/tex/main.tex:175-176)  
> "divide the matrix into blocks... load outliers into shared memory... determine if outliers are part of the segment... load weights... perform matmul." (paper-source/primary/tex/method.tex:229-237)

## 6. Experimental results

### 6.1 Perplexity vs bit budget (LLaMA)
- Two SpQR configurations per model size:
  - **Near‑lossless (~4.6–4.7 bpp)**: higher budget, ≤1% perplexity gap vs fp16.
  - **Size‑matched (~3.9–4.0 bpp)**: tuned to ~4 bpp for fair comparison to RTN‑4bit / GPTQ‑4bit.

  **Key slice (LLaMA‑7B)**:

  | Method | Avg bits | Wiki2 | C4 | PTB |
  | --- | ---: | ---: | ---: | ---: |
  | fp16 baseline | 16.00 | 5.68 | 7.08 | 8.80 |
  | RTN‑4bit | 4.00 | 6.43 | 7.93 | 10.30 |
  | GPTQ‑4bit | 4.00 | 6.13 | 7.43 | 9.27 |
  | SpQR (~4 bpp) | 3.94 | 5.87 | 7.28 | 9.07 |
  | SpQR (near‑lossless) | 4.63 | 5.73 | 7.13 | 8.88 |

> Table 1 shows both SpQR rows per size and their Avg bits (paper-source/primary/tex/experiments.tex:61-65,67-71,80-90).

### 6.2 Inference speed (A100, batch=1)
- Optimized SpQR kernel beats fp16 and is ~2× faster than dense‑quant + PyTorch sparse baseline.

  **Key slice (tokens/s)**:
  - **13B scratch**: fp16 `37±0.8` → SpQR optimized `44±0.5`; SpQR(PyTorch) `24±1.2`.  
  - **30B scratch**: fp16 `19±1.1` → SpQR optimized `22±0.9`; SpQR(PyTorch) `8.8±0.4`.  
  - **13B prefix‑1024**: fp16 `31±0.9` → SpQR optimized `37±0.8`.  
  - **30B prefix‑1024**: fp16 `17±0.8` → SpQR optimized `22±1.3`.  

> "Inference speed comparison (tokens/s)... optimized SpQR... faster than the 16-bit baseline and almost 2.0x faster than ... PyTorch sparse..." (paper-source/primary/tex/experiments.tex:204-221)

### 6.3 Ablations (what matters)
- **Bilevel stats + small groups** materially improve loss at the same ~3.6 bpp.  
> "3-bit SpQR with group size 16... using 3-bit bilevel quantization vs ... 16-bit statistics... quantized statistics significantly improves language modeling loss." (paper-source/primary/tex/experiments.tex:148-150)

- **Unstructured outliers are most efficient per bit** vs row/column outliers.  
> "Overall, unstructured outliers reduce perplexity significantly faster..." (paper-source/primary/tex/experiments.tex:152-153)

- **Minor heuristics**: rounding zero‑points and act‑order have smaller effects.  
> "'Round zero'... reduces footprint but increases perplexity... act order... slightly improves loss..." (paper-source/primary/tex/experiments.tex:155)

## 7. Conclusion & future work
- **Takeaway**: A tiny fp16 sparse residual + aggressively compressed metadata unlocks near‑lossless 3–4 bit PTQ for LLMs.

> "SpQR... achieve near-lossless 16-bit accuracy with less than 4.75 bits per parameter on average." (paper-source/primary/tex/main.tex:342)

- **Future directions**:
  - Reduce outlier overhead by grouping/compressing them further.  
    > "This could be reduced significantly by grouping outliers, which we leave as future work." (paper-source/primary/tex/method.tex:222-223)
  - Evaluate human‑judged generation quality and fuse dense+sparse kernels more tightly.  
    > "We do not evaluate the generative quality... another limitation is that we do not fuse sparse matrix multiplication with regular quantized matrix multiplication... leave... to future work." (paper-source/primary/tex/main.tex:343-344)
