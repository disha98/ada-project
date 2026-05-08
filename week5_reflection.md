# Week 5 Reflection

## What I have achieved so far

- Collected 50 books from Project Gutenberg
- Cleaned the text and tokenized it into words
- Built a dataset of sliding-window context and target pairs
- Created a smaller vocabulary with an `<UNK>` token for out-of-vocabulary words
- Saved processed outputs such as the vocabulary mapping and sample dataset
- Built a baseline model using SGD-based logistic regression with TF-IDF features
- Generated exploratory plots and summary statistics for the dataset
- Reported key information such as token counts, vocabulary size, and context window length
- Improved the evaluation setup by changing from a random split to a book-level validation split
- Updated the preprocessing so `<UNK>` stays inside context windows instead of being removed

## What I am happy with from my project work so far

I am happy that I now have a complete end-to-end baseline pipeline. The notebook goes from raw text collection all the way through cleaning, feature engineering, model training, evaluation, and qualitative prediction examples. I also think the project has a sensible baseline for comparison with later models such as Random Forest and MLP.

I am also happy that I caught an important evaluation issue early. My earlier version used a random validation split over highly overlapping windows, which could overestimate performance. Fixing that to a book-level split makes the project more credible, even if it may lower the reported accuracy. (I actually learnt more about this in one of the presentation videos so that was great!)

## Challenges

The main challenge I am facing is that next-word prediction is a very difficult multi-class problem, especially with a large vocabulary and relatively simple models. Even when the pipeline is working, the accuracy is still modest, so I need to think carefully about what counts as meaningful progress and how to compare models fairly.

Another challenge is balancing ambition with computational simplicity. I want to compare classical ML methods and a simple neural baseline, but some choices that look straightforward in theory become expensive in practice because of the dataset size and number of classes. I also need to decide how much to experiment with features, such as TF-IDF versus bag-of-words or different context window sizes, without making the project too broad.

## I would specifically like feedback or advice on

I would especially appreciate feedback on whether my project framing is strong enough for the course goals now that I have a Week 5 baseline. In particular, I would like advice on:

- whether the current evaluation setup and metrics are appropriate for this kind of task
- whether I'll be able to achieve testing with Random Forest considering the data and sample size
- whether there are any obvious risks in my current direction that I should address before I move into the Week 7 stage

### Some thoughts

**Why is Top-1 accuracy low?**  
Next-word prediction is fundamentally ambiguous. Given any 4-word context, dozens of words are plausible continuations. A 10,000-class random baseline would achieve only ~0.01% accuracy, so even a few percent Top-1 accuracy represents meaningful signal well above chance.

**What does Top-3 accuracy mean here?**  
If the model places the correct word within its top 3 predictions, that is meaningful for applications like autocomplete, where presenting a short list of suggestions to the user is acceptable.

**Limitations of this approach:**

- TF-IDF features treat context words as a *bag* — word **order** within the context window is ignored. The model cannot distinguish "the cat sat" from "sat cat the".
- No semantic representation — words are discrete symbols with no notion of similarity. "dog" and "cat" look completely unrelated in TF-IDF space.

**What's really going on with the example predictions?:**

- Right now, the model often gives very similar top predictions for many different examples.
- Since TF-IDF does not understand word order or meaning very well, the model falls back to common next words like "the", "of", "and", or "to".
- This means the model is learning some general patterns from frequent words, but it is not using the context deeply enough yet.
- So the current results are still useful, because they show why next-word prediction is hard and why I need stronger features or a better model next.

**Planned improvements for Week 7:**

- Keep the current model as my baseline, since it gives me something simple and clear to compare against.
- Try better input features, especially **word embeddings** such as GloVe or Word2Vec, so the model can use word meaning better.
- Train a simple **MLP** on top of those embedding features and compare it against the baseline.
- Test a few context window sizes, such as 3, 4, and 5 words, to see whether a little more context helps.
- If possible, I may still test **Random Forest** on a smaller version of the data, but I might have to re-evaluate feasibility because it may be too slow or not a good fit for this problem.
