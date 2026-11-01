**Task**
We investigate Mixed Boolean-Arithmetic (MBA) deobfuscation, which simplifies obfuscated expressions that mix arithmetic and bitwise operations into an equivalent but more concise form (e.g. (x | y) − (~x & y) simplifies to x). We use the NeuReduce dataset of one million generated MBA expressions paired with their simplified forms.

We will explore the model through the analysis of its inputs and outputs to identify systematic patterns related to the number of variables, expression length or operator composition. Based on this we will investigate what the model has learned and how this relates to the final output of its deobfuscation attempts. 

Paper: https://aclanthology.org/2020.findings-emnlp.56.pdf
Dataset: included in the repository

Model
We will use the Transformer variant of NeuReduce, a sequence-to-sequence model for MBA simplification, with the pretrained checkpoint released with the code.

Later in our analysis we may also compare it with one of NeuReduce’s recurrent variants (e.g. GRU) retrained on the same data which would allow us to further analyse representations and model output. 

Repository: https://github.com/fvrmatteo/NeuReduce (fork of nhpcc502/NeuReduce)
Model checkpoint: model.h5 (Transformer) in the repository
