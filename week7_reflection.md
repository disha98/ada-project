# Week 7 Reflection

## What I have achieved so far

- Compared **Bag-of-Words vs TF-IDF** feature representations using the same Logistic Regression model to understand whether TF-IDF's frequency weighting helps
- Tested **context window sizes of 3, 4, and 5 words** to see whether more context improves predictions with bag-of-words features
- Trained a **Random Forest** classifier on a 50,000-sample subset and documented its limitations for this problem structure
- Trained a **simple MLP** (512-256-128 hidden layers) as a neural baseline, using the same TF-IDF features for fair comparison
- Compared all models on Top-1, Top-3, and Top-5 accuracy with a grouped bar chart
- Ran qualitative prediction examples through all models to show how their outputs differ

## Challenges

The main challenge this time was deciding what to keep and what to cut. My original plan included Random Forest, MLP, feature experiments, and window size experiments, all of which are meaningful but each adds complexity. I had to think carefully about how to scope each experiment so the notebook tells a coherent story without becoming an unfocused collection of results.

Random Forest was particularly tricky to scope. I also knew from the earlier TA's feedback that it might not work well, but I still wanted to try it rather than just assert it was infeasible. Training on a 50k subset let me include real results and a grounded discussion of why tree-based models struggle here.

I also initially thought fitting the LabelEncoder on all targets (train + val) before splitting might be a form of data leakage, but after looking into it I confirmed it is not. LabelEncoder is just a string-to-integer mapping that carries no statistical information, and the models only ever see training labels during fitting.

One known sort of problem is that sklearn's default tokenizer strips the angle brackets from `<UNK>`, so the token is silently dropped from TF-IDF and BoW feature vectors. I chose not to fix this because `<UNK>` is a catch-all for many unrelated rare words and its presence in a context window carries very little discriminative signal, and a well-regularized model would likely give it near-zero weight anyway. All models are affected equally, so it does not change any comparisons.

## What I would specifically like feedback on

- Any suggestions for the final project phase - I plan to follow my original README commitment (hyperparameter tuning, test set evaluation, error analysis), but I am open to adjusting the scope or any feedback around what I can add/expand on

## Plans for the final project deadline

Following my original execution plan:

- **Hyperparameter tuning:** Grid search or targeted tuning for the MLP (learning rate, layer sizes, regularization) and Logistic Regression (regularization strength)
- **Test set evaluation:** Hold out a separate set of books as a final test set and report Top-1, Top-3, and Top-5 accuracy for all models
- **Error analysis:** Investigate which words are most frequently mispredicted, whether certain types of context are harder than others, and how vocabulary size affects accuracy
- **Final writeup and demo:** Summarize findings comparing classical ML vs MLP, discuss strengths and limitations of classical approaches for next-word prediction
