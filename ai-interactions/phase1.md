# AI interaction record - Phase I (AI-aided pair)

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