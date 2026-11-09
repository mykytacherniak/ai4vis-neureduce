]633;E;cat /tmp/record.md;cecf62e9-be68-41ad-b9fd-889b610424b2]633;C# AI interaction record - Phase I (AI-aided pair)

AI model used: Claude (chat with code sandbox). Period: 2-7 November 2026.
Model analysed: NeuReduce Transformer (best-transformer.pt).

## How AI broadened the analysis

- Initial notebooks including histograms and data analysis. 
- Controlled input changes (metamorphic probes) on all test inputs (notebook 05).
- Interactive projection explorer with linked selection (notebook 06).
- Projection of wrong outputs by kind of mistake (notebook 06).
- Check of term order in training and test data (notebook 08).
- Presentation notebook and slide deck (notebook 07).

## Important prompts and their effect

1. "How do we set up NeuReduce to use best-transformer.pt? There are many notebooks, how do we start?"
   Effect: AI rebuilt the preprocessing without the removed torchtext API and found that the original notebook's accuracy (0.826) was miscomputed (0.899).
   Decision: accepted after checking against the notebook's printed outputs.

2. Teammate's plan: load dataset.csv, make an 80/20 split, run the model, write one shared CSV with input/output attributes and correctness.
   Effect: AI kept the plan (presented in context) but used the original seeded split instead of a new one.
   Decision: modified. A new split would have put about 80% training data into the test set.

3. "We  want to see what happens when the model gets non-linear or unexpected input. How could we best implement this? We need permutations, name-changes, etc. How can we best achieve this?"
   Effect: AI proposed controlled rewrites with known correct answers. Became notebook 05 and our main finding.
   Decision: accepted.

4. "Do the overview for both the training and the test data."
   Effect: showed the gap between seen and unseen examples from the start.
   Decision: our extension of the AI's plan.

5. "This is difficult to read. What are we looking at? Add simpler versions, don't remove code."
   Effect: simpler plots added + set the readability standard for later figures.
   Decision: modified on our request.

6. "Check accuracy against how often an answer appeared in training."
   Effect: AI expected this to be the main finding. Data showed unseen answers fail 2-3x more often but cause only about 10% of errors.
   Decision: AI's framing rejected, result kept as a secondary finding.

7. "Write the error taxonomy notebook and give a rundown."
   Effect: most mistakes are near-misses, two-variable AND/OR answers get collapsed to one variable. After our review, AI admitted its confidence box plot exaggerated the separation and replaced it.
   Decision: accepted with correction.

8. "For our current projections, the clusters are huge and hard to understand. How can we constrain this and reuse earlier results?"
    Effect: fewer named regions, colouring by mistake type and robustness. Showed that fragility is present in every kind of input.
    Decision: modified. First coarse clustering put more than 9,000 of 10,885 inputs into one region and was replaced.

9. "The data is spread out, clusters are not clean cut."
    Effect: reframed as a finding: errors are not concentrated by input type, so we tested the written form. Used as the bridge slide to the rewrite experiment.
    Decision: accepted.

## Failure patterns

- Wrong or overconfident hypotheses such as memorization being the main cause and a separate "no constant in target" hotspot that turned out to be the AND/OR group.
- Code that worked in the AI's sandbox but not elsewhere.
- Plot errors such as wrong line order, hidden outliers and overlapping labels.
- Overstated effort and requirements such as how long an analysis might take, in turn discouraging certain approaches but relenting when probed further.
- Verbose first versions, complicated figures which were hard to understand first try.

## Best practices

- Ask for verification against raw outputs or a second method for every surprising number.
- Ask what a plot shows before using it and push back on unreadable figures.
- Challenge effort and requirement claims.
- Start with the idea for analysis and work from there to ground any AI recommendations.

## Other notes

1. Precise error types instead of accuracy

MBA has two properties you can check exactly. You can test whether an output is semantically equivalent to the input, and whether it is simpler than it. So I would sort the model’s outputs on the test set into at least these categories:

Syntactically invalid: the output doesn’t parse, e.g., unbalanced parentheses or a truncated token. This is the clearest form of seq2seq “hallucination.”
Exact match with the ground truth.
Equivalent, different form: correct, just not exactly the target expression.
Equivalent but not simplified: correct, but barely shorter than the input.
“Almost right”: one coefficient, sign or constant is off, or one term is missing or extra.
Completely wrong.

Checking equivalence is cheap for linear MBA. It’s enough to evaluate on all {0,1}ⁿ inputs (this is the fundamental theorem of MBA), and it’s worth also checking on random 64-bit values or with z3. I think the “almost right” category is the most interesting, because the structure of those errors reveals a lot. For it I’d do a term-level diff: split both the output and the ground truth into sets of (coefficient, bitwise term) pairs.

2. Features for every input

number of variables
input length in characters and in tokens
ground-truth length and the ratio between the two (compression ratio)
count of each operator (+ - * & | ^ ~) and their proportions
nesting depth
number of terms
size and digit count of the coefficients
the truth table / signature vector

The last one matters most. For linear MBA it uniquely determines the simplified form. That lets you place each expression in two “spaces”: the syntactic space (the surface the model sees) and the semantic space (what it should learn). If the error patterns differ between the two, that is a strong finding by itself.

3. Concrete hypotheses worth testing

Length or complexity? Accuracy probably drops with length. The deeper question is whether input length causes it (attention, max sequence length, truncation) or the complexity of the target. Stratifying separates the two: vary input length while holding target length fixed, and vice versa.
Number handling: a character- or token-level model often doesn’t “compute” a coefficient. Instead it copies or makes up frequent numbers. Check whether errors concentrate in the coefficients, and whether the error rate rises with the number of digits.
Memorization and pull toward frequent targets: the distribution of target expressions in the training set is probably long-tailed. I’d check whether wrong outputs are disproportionately targets that appear often in the training data. That would be a prior bias, a “hallucination toward the frequent answer.”
Train/test overlap: check how many test targets appear verbatim in the training set, and whether the model does better on those. This is common in generated datasets and could explain a lot.
Role of variables: does the model degrade at 3–4 variables, and is it worse on some variable names (e.g., t, z) than on x, y?
Confidence: does the log-probability or beam score predict correctness? How many equivalent candidates are in the top-k beams? If the 2nd beam is often correct, the model “knows” the answer but ranks it wrong.

4. Synthetic, metamorphic tests

This is what shows a pattern is systematic rather than cherry-picked. Take an expression, apply a semantics-preserving transformation, and see whether the output changes:

swapping operands (x & y → y & x), shuffling term order
renaming variables (x↔y)
changing constants, using larger coefficients
adding a sub-expression that equals zero, e.g. (x ^ y) - (x | y) + (x & y)

A model that really “understood” the task is invariant to these. If it isn’t, it’s fitting surface patterns. Longer, more-variable, or nonlinear expressions serve as out-of-distribution tests on top of this.

Visualization ideas
Heatmap: number of variables × input length, colored by error type (not just accuracy). It shows far more than a table.
2D projection (UMAP) of the feature vectors, with linked views: clicking a cluster shows its operator composition, coefficient distribution and a few examples. This is exactly the drill-down the assignment asks for: what distinguishes the cluster, and does it have sub-clusters?
Term-level confusion matrix: which ground-truth bitwise term the model replaces with which, or drops. This shows which terms are “hard.”
Metamorphic stability plot: how much the output changes under each transformation.
Confidence vs. correctness: a calibration curve broken down by error type.

Suggested order
Run the checkpoint on the test set (a few tens of thousands of examples is enough). Save the outputs and the beam scores.
Write the equivalence check and the error categorization.
Compute the features and look at the training-set distributions. That’s where your hypotheses come from.
Pick 2–3 hypotheses and go deep on them with synthetic tests. You don’t need to do everything: fewer patterns, understood thoroughly, are worth more.

I think the strongest story will probably come from hypotheses 2 and 3. If the model systematically gets the coefficients wrong, or drifts toward frequent targets, that explains well what it actually learned: surface templates, not arithmetic.
