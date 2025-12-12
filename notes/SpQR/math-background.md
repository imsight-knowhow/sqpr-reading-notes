# Math Background for GPTQ / SpQR

This note collects the minimal matrix‑calculus facts needed to follow the GPTQ/OBQ equations used by SpQR. It is **not** a full derivation of GPTQ, just the parts that explain where the quadratic objective, the Hessian, and the greedy score come from.

## 1. Setup and notation

For a single linear layer:

- Weight matrix: \(W\in\mathbb{R}^{m\times n}\) (output dim \(m\), input dim \(n\)).
- Calibration activations at the layer input: \(X\in\mathbb{R}^{n\times N}\), where each column is one calibration sample (or one token position).
- Full‑precision output on calibration data: \(Y = WX\).
- Quantized weights: \(\widehat W\) (constrained to a low‑bit grid for most entries).

GPTQ/SpQR perform **post‑training** reconstruction: they do not use gradients of the original training loss. Instead they approximate the local error in **layer outputs** on calibration data.

## 2. Frobenius norm and trace

The Frobenius norm of a matrix \(A\) is
\[
\|A\|_F^2 = \sum_{i,j} A_{ij}^2.
\]

It can be written using the **trace**:
\[
\|A\|_F^2 = \mathrm{tr}(AA^\top).
\]

### What is \(\mathrm{tr}(\cdot)\)?

For a square matrix \(B\), \(\mathrm{tr}(B)=\sum_i B_{ii}\) (sum of diagonal entries).

### Why use trace?

Trace lets us rewrite sums compactly and take derivatives cleanly. Two identities used repeatedly:

- **Cyclic property**: \(\mathrm{tr}(ABC)=\mathrm{tr}(CAB)=\mathrm{tr}(BCA)\) as long as products are defined.
- **Derivative rule**: if \(L=\mathrm{tr}(U^\top A U)\) with symmetric \(A\), then \(\nabla_U L = 2AU\).

## 3. Layer‑wise reconstruction objective

GPTQ starts from the layer reconstruction loss
\[
L(\widehat W) \;=\; \|WX-\widehat W X\|_F^2
\;=\; \|(W-\widehat W)X\|_F^2.
\]

Let the error matrix be \(E = W-\widehat W\). Using trace:
\[
L(\widehat W)=\mathrm{tr}\!\left(EXX^\top E^\top\right).
\]

So the loss is a **quadratic form** in the weights, with data matrix \(XX^\top\) as curvature.

## 4. Gradient and Hessian

It is easiest to view the loss row‑wise (because it separates), but the resulting Hessian is the same for every row.

Take one output row \(w^\top\in\mathbb{R}^{n}\) and its quantized version \(\widehat w^\top\).
The row loss is
\[
L(\widehat w) = \|(w-\widehat w)X\|_2^2
= (w-\widehat w)^\top (XX^\top)(w-\widehat w).
\]

### Gradient

\[
\nabla_{\widehat w} L = -2(XX^\top)(w-\widehat w).
\]

### Hessian

Differentiate again:
\[
H \;\equiv\; \nabla_{\widehat w}^2 L
\;=\; 2XX^\top.
\]

In practice GPTQ adds **damping** for stability/invertibility:
\[
H \approx 2XX^\top + \lambda I.
\]

**Interpretation**: \(H\) measures how output error grows when we perturb weights. Large curvature directions mean **sensitive** weights/features.

## 5. Why \(\Delta L\) depends on \(H\)

For the reconstruction loss we use, the second‑order (quadratic) form is not an approximation — it follows directly.

Start from the row loss (Section 4):
\[
L(\widehat w)=\|(w-\widehat w)X\|_2^2 .
\]
Expand the squared norm:
\[
\|(w-\widehat w)X\|_2^2
= (w-\widehat w)X\,X^\top (w-\widehat w)^\top
= (w-\widehat w)^\top (XX^\top)(w-\widehat w).
\]
Define the perturbation \(\delta w=\widehat w-w\) (so \(w-\widehat w=-\delta w\)):
\[
L(\widehat w)=\delta w^\top (XX^\top)\,\delta w.
\]
Using \(H=2XX^\top\), this becomes
\[
L(\widehat w)=\tfrac12\,\delta w^\top H\,\delta w.
\]
Since the full‑precision weights give \(L(w)=0\), the loss increase from quantization is
\(\Delta L = L(\widehat w)-L(w)=L(\widehat w)\),
so the quadratic form above is exactly how \(H\) controls sensitivity to quantization errors.

## 6. Optimal Brain Quantization (OBQ) score

OBQ is the second‑order greedy solver that GPTQ scales up.

### Problem: quantize one coordinate but re‑optimize the rest

Suppose we force one coordinate \(q\) to a quantized value:

- Let the quantization error on that coordinate be
  \(\varepsilon = w_q - \widehat w_q\).
- The remaining coordinates \(F=\{1,\dots,n\}\setminus\{q\}\) are still free to move to minimize the quadratic loss.

We solve
\[
\min_{\widehat w_F}\;\; \tfrac12\,\delta w^\top H\,\delta w
\quad \text{s.t. } \delta w_q = -\varepsilon.
\]

### Solution (Lagrange multiplier)

The optimizer for the free coordinates is
\[
\delta w_F^*
= -\frac{\varepsilon}{[H^{-1}]_{qq}}\,(H^{-1})_{F,q}.
\]

Plugging this back gives the **minimal achievable loss increase**:
\[
\Delta L^*(w_q)
= \tfrac12\,\frac{\varepsilon^2}{[H^{-1}]_{qq}}
= \tfrac12\,\frac{(\mathrm{quant}(w_q)-w_q)^2}{[H^{-1}]_{qq}}.
\]

These are exactly the OBQ/GPTQ key equations.

### Why does \([H^{-1}]_{qq}\) appear?

If weights were independent (diagonal \(H\)), then
\([H^{-1}]_{qq} = 1/H_{qq}\) and
\(\Delta L^* = \tfrac12\, H_{qq}\varepsilon^2\).

When weights are **coupled** (off‑diagonal \(H\neq 0\)), allowing the other coordinates to re‑optimize changes the effective curvature along \(q\). The Schur‑complement algebra yields the reciprocal of the diagonal of \(H^{-1}\). Intuitively:

- \(H^{-1}\) behaves like a covariance of “how much the rest can move to absorb error in \(q\)”.
- Small \([H^{-1}]_{qq}\) ⇒ little slack to compensate ⇒ large loss increase ⇒ **high sensitivity**.

## 7. How GPTQ makes OBQ practical

OBQ as written is too expensive for LLM layers because it would repeatedly update full \(H^{-1}\) per scalar weight (cubic in \(n\)).

GPTQ keeps the same quadratic score but:

- Estimates \(H\) once from calibration activations \(X\).
- Uses Cholesky factors instead of explicit inversion.
- Quantizes **columns/blocks** in a fixed order for efficiency, applying the OBQ correction in block form.

SpQR extends this GPTQ pass with outlier detection and bilevel statistics, but the above quadratic machinery is the core reason Hessian‑weighted scores identify sensitive weights.
