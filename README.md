# AI4Vis Project

## Task

We investigate **Mixed Boolean-Arithmetic (MBA) deobfuscation**, which simplifies obfuscated expressions that mix arithmetic and bitwise operations into an equivalent but more concise form.

For example:

```text
(x | y) - (~x & y)  →  x
```

We use the **NeuReduce** dataset of one million generated MBA expressions paired with their simplified forms.

We will explore the model through the analysis of its inputs and outputs to identify systematic patterns related to:

- number of variables,
- expression length,
- operator composition.

Based on this, we will investigate what the model has learned and how this relates to the final output of its deobfuscation attempts.

**Paper:**  
https://aclanthology.org/2020.findings-emnlp.56.pdf

**Dataset:**  
Included in the NeuReduce repository.

## Model

We will use the **Transformer variant of NeuReduce**, a sequence-to-sequence model for MBA simplification, with the pretrained checkpoint released with the code.

Depending on our findings, we may also compare it with other model architectures evaluated in the NeuReduce paper to investigate differences in their behavior and learned representations.

**NeuReduce repository:**  
https://github.com/fvrmatteo/NeuReduce  
(fork of `nhpcc502/NeuReduce`)

**Model checkpoint:**

```text
PyTorch/best-transformer.pt
```