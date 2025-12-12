now the main-note.md contains enough information, we are going to create a `notes/SpQR/present-note.md`for presentation (a markdown doc with essential information for presentation, style is similar to `main-note.md`, just less and more focused information). It will contain the following info:

- title and metadata, tldr
- existing quantization methods related to SpQR, but with pitfalls and limitations of each method
- key observations that led to SpQR
- about SpQR quantization representation:
    - how outliers are identified
    - how quantized weights and outliers are stored
    - how quantized weights and outliers are used during inference
- experimental results (mainly focusing on the comparison with existing quantization methods, and ablation studies on key components of SpQR)
- conclusion and future work    

structure the `present-note.md` in a way that is easy to follow during a presentation, using bullet points, subheadings, and concise explanations.

important to note that:
- DO NOT refer to `main-note.md` in `present-note.md`; it should be a standalone document.
- use blockquotes and line numbers to reference specific parts of the source paper and source code to ground your explanations and allow for verifiability.