# Project 2: Next Word Predictor

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
