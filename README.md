# Next Word Predictor

The goal of this project is to predict the most likely word following a given context. While modern approaches rely on Transformers, this project will explore using **classical ML models** (Random Forests, Logistic Regression) and simple neural networks (MLP) for this task.  

I will use a collection of **50–100 books from [Project Gutenberg](https://www.gutenberg.org/)** for training and testing.

The goals are to:  

- Formulate the task as a multi-class classification problem, where the input is a context window of 3–5 preceding words and the output is the predicted next word.
- Compare classical ML models (Random Forest, Logistic Regression) with a simple MLP baseline.  
- Evaluate performance using **Top-1 and Top-k accuracy**, and optionally **cross-entropy / perplexity**.  
- Explore how preprocessing (tokenization, limiting vocabulary, TF-IDF features, word embeddings) affects performance.  

### Execution Plan

#### By Week 5 Homework Deadline

- Collect and explore dataset:
  - Choose 50 random books from Project Gutenberg
  - Clean text (remove headers/footers, lowercase, remove punctuation)  
- Preprocess data:
  - Tokenize text into words  
  - Generate training samples by creating sliding windows of 3–5 words as input contexts, with the subsequent word as the target label.
- Feature engineering:
  - Convert context words into numerical features (TF-IDF, one-hot encoding, etc)  
- Build a baseline classifier using Logistic Regression or Random Forest  
- Evaluate initial metrics:
  - Top-1 accuracy on validation set  
  - Report key dataset characteristics, including vocabulary size and context window length

#### By Week 7 Homework Deadline

- Train full models:
  - Random Forest, Logistic Regression on full training set  
  - Simple MLP with 2-3 hidden layers
- Evaluate models:
  - Top-1 and Top-3 accuracy  
  - Optionally compute cross-entropy / perplexity  
- Experiment with feature representations:
  - Bag-of-words vs TF-IDF  
  - Effect of context window size (3 vs 5 words)  
- Compare classical ML vs MLP baseline performance  

#### By Final Project Due Date

- Fine-tune models:
  - Hyperparameter tuning for Random Forest
  - Regularization for Logistic Regression / MLP  
- Evaluate fully on test set:
  - Top-1, Top-3, and Top-5 accuracy  
  - Cross-entropy or perplexity
- Analyze errors:
  - Which words are most frequently mispredicted?  
  - Does context length or vocabulary size affect accuracy?  
- Summarize findings:
  - Compare classical ML vs MLP  
  - Discuss strengths and limitations of classical approaches

## Results & Findings

All numbers below are evaluated on the **held-out test books** (book-level 70/15/15 split, ~2,940 evaluated samples after dropping targets unseen in training).

| Model | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- |
| Bigram baseline | **11.70%** | **23.13%** | **28.06%** |
| MLP (BoW) | 9.97% | 17.38% | 22.01% |
| Logistic Regression (BoW) | 9.90% | 19.56% | 25.48% |
| Random Forest (TF-IDF) | 7.31% | 14.66% | 19.59% |
| Unigram baseline | 5.88% | 12.55% | 17.86% |
| Uniform random | ~0.02% | — | — |

Key findings:

- **The trained models did not beat the bigram baseline.** A trivial order-aware bigram (predict the most likely word given only the previous word) beats Logistic Regression, the MLP, and Random Forest on every metric. The trained models comfortably beat the unigram and uniform-random references, but that is a low bar.
- **Word order is the bottleneck.** Bag-of-words and TF-IDF features discard word order, so the models cannot use the immediate-adjacency signal that the bigram exploits. This is the main reason the bigram wins and is the core limitation of the classical approach for this task.
- **MLP ≈ Logistic Regression.** The MLP's Top-1 edge over LR (9.97% vs 9.90%) is within overlapping bootstrap confidence intervals, and LR is clearly better on Top-3/Top-5. Practically a tie — the extra model complexity did not pay off.
- **Shorter context worked better.** In the context-length ablation, a 3-word window (10.48%) outperformed 4 (9.74%) and 5 (9.16%) on validation. With order-blind features and a fixed sample budget, more context words add sparsity/noise rather than signal.
- **Bigger vocabulary helped monotonically** (2k → 7.24%, 5k → 7.66%, 10k → 9.72%), and **BoW slightly outperformed TF-IDF** for these models.
- **Errors are dominated by common function words** (`the`, `a`, `and`, `in`, `to`), and the most common confusion is predicting `the` in place of other function words — expected behavior for a sparse-feature model with no word order.

## Error Analysis

Looking at where the best model got things wrong, almost all the mistakes involve very common words. The most frequently mispredicted true words are function words like `the`, `a`, `and`, `in`, and `to`, and the most common single error is predicting `the` when the real next word was a different function word.

This makes sense: function words show up everywhere, so they are both the most common targets and the safest guesses for a model that can't see word order. The model leans on overall word frequency, which means it does fine on generic filler words but struggles with the more specific, content-carrying words that actually depend on context.

## Choices, Tradeoffs & Limitations

Most of the limitations here come from running everything locally on a laptop and from deliberate scoping decisions to keep the pipeline finishing in a reasonable time, rather than from the method itself.

- **Compute budget capped the dataset size.** Everything runs locally, so I capped the sample budget (and the shared test subset at ~3,000 rows) so the full notebook — tuning three model families plus ablations — finishes end-to-end without an overnight run. A bigger test set would tighten the confidence intervals, but it wouldn't change the headline that the bigram wins.
- **I chose to keep window=4 for the final models even though the ablation liked window=3.** Window=3 was only marginally better on validation (10.48% vs 9.74%), and keeping window=4 makes the final results directly comparable to the Week 5 and Week 7 checkpoints. I treated comparability across the term as more valuable than chasing a fraction of a percent.
- **The label space grows between the validation and final stages**, because the final models are refit on train+val (so they see more target words). This is the intended behavior of a proper train/val/test workflow, but it does mean perplexity isn't directly comparable across the two stages, since the number of classes changes. Random Forest perplexity is also high mostly because tree probability estimates are poorly calibrated, not because the model is far worse than the others.
- **Classical sparse features are the real ceiling.** Bag-of-words throws away word order, which is exactly the signal the bigram uses — so this is a limitation of the chosen approach, not something more tuning would have fixed.

## Summary & Reflection

This week the project was extended from the Week 5 and Week 7 checkpoints into a full final pipeline: stronger baselines (unigram + bigram), regularization tuning for LR / MLP / Random Forest, context-length and vocabulary-size ablations, a single held-out test evaluation with bootstrap confidence intervals, and error analysis.

The methodology is the part I'm most satisfied with: a strict book-level split (generalizing to unseen books, not just unseen windows), a label space fit on training targets only with explicit tracking of unseen targets, all model families evaluated on the same shared test subset, and confidence intervals so the comparisons are honest about noise.

The honest conclusion is that **classical sparse-feature models do learn real signal but fail to beat a simple bigram baseline** for next-word prediction. The deciding factor is feature representation — bag-of-words throws away word order, which is exactly the information a bigram keeps and the single most useful cue for predicting the next word. This is a satisfying negative result: it concretely demonstrates *why* production next-word prediction moved to sequence-aware models (RNNs, Transformers) rather than classical sparse-feature classifiers. If I continued the project, the natural next step would be order-aware features or embeddings rather than further tuning of the current models.
