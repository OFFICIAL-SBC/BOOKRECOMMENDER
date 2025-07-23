# Gloria 📚

![Mom's iamge](/assets/mom.png)

## A Personal Semantic Book Recommender

**Gloria** is an intelligent book recommendation system that leverages semantic understanding to suggest books based on the meaning and intent of a user’s input prompt — not just keywords. Simply enter a phrase, and Gloria returns a curated list of books whose descriptions align semantically with your prompt.

---

## 🛠 Development Phases

### 1. Data Exploration 🔍

#### 📊 Statistical Analysis

1. We selected the dataset **[`dylanjcastillo/7k-books-with-metadata`](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)** available on Kaggle, which contains metadata for 7,000 books.
2. A thorough statistical inspection of the dataset was performed:

   - **Missing Data Analysis**:
     - The column with the most missing values was `Subtitle`.
   - **Numerical Data Analysis**:
     - We analyzed the distributions, ranges, and basic statistics of numerical columns such as page count, year, and ratings.
   - **Categorical Data Analysis**:
     - We examined columns like `Category`, assessing:
       - The number of unique values
       - Most frequent value
       - Frequency of the most common category
     - We found **567 unique categories**, with **"Fiction"** being by far the most common.

   #### 📈 **Top 10 Most Frequent Categories**

   | Category                  | Count |
   | ------------------------- | ----- |
   | Fiction                   | 2,588 |
   | Juvenile Fiction          | 538   |
   | Biography & Autobiography | 401   |
   | History                   | 266   |
   | Literary Criticism        | 166   |
   | Philosophy                | 162   |
   | Comics & Graphic Novels   | 159   |
   | Religion                  | 137   |
   | Drama                     | 132   |
   | Juvenile Nonfiction       | 116   |

   > ⚠️ **Note:** "Fiction" alone accounts for nearly **50%** of all records in the dataset.

3. All rows with a `null` value in the `Description` field were removed.

---

### 2. Data Cleaning & Preprocessing 🧹

#### ⚠️ Problem: Too Many Sparse Categories

> 🛠 **Solution:**  
> All categories were reclassified into two broader groups: **Fiction** and **Nonfiction**.

This simplification reduces noise and helps the model generalize better during semantic matching.

---

#### ✂️ Short Descriptions Are Not Useful

> 🛠 **Insight:**  
> Descriptions need to be **meaningful** and **informative** to enable accurate semantic similarity analysis. Very short descriptions are not suitable.

- We applied a threshold-based filtering mechanism:
  - Rows with descriptions shorter than a certain number of characters were removed.
  - Several thresholds were tested: **5, 14, 24, 34**
  - After trial and error, a **final threshold of 25 characters** was chosen to balance data quantity and quality.

# Vector Search 🔍

This phase builds upon the cleaned dataset obtained during the data exploration phase.

---

## 🔢 Goal

Transform textual book descriptions into numerical vectors (embeddings) that can be stored and searched semantically using a vector database.

---

## ⚙️ Embedding Strategy

- Raw text needs to be converted into a format suitable for vector search. This involves:

  1. **Tokenization** – Breaking text into smaller components (words or subwords).
  2. **Embedding Generation** – Converting tokens into high-dimensional vectors using pre-trained models.

- We use **encoder models** (not decoders) for this task:

  - They produce **document embeddings** by understanding the full context of the input.
  - Encoder models like BERT, MPNet, or similar are ideal since they capture semantic relationships between words.

- Why use pre-trained models?
  - Large Language Models (LLMs) are expensive to train from scratch.
  - Pre-trained models generalize well and can be fine-tuned or directly used for tasks like vectorization.

---

## 📁 Vector Database

- Once the document embeddings are generated, they are stored in a **vector database**.
- Key benefits:

  - Enables **vector search** out of the box.
  - Uses **cosine similarity** (or other similarity metrics) to compare queries to existing vectors.
  - Allows efficient retrieval through **approximate nearest neighbor (ANN)** algorithms, avoiding brute-force linear comparisons.
  - Each vector can be associated with a unique ID (in our case, the book's **ISBN-13**).

- This ID helps us link back the embedding to the actual metadata in the relational dataset.

---

## 🌐 Tools

- We use **LangChain** to manage vector search operations, including:
  - Embedding generation via API or local models
  - Storing vectors in a vector database
  - Executing semantic similarity queries

```python
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
```

---

## ✅ Summary

- Text is embedded using encoder models
- Embeddings are stored in a vector database with ISBN-13 as identifiers
- Search is based on semantic similarity, not keywords
- LangChain powers the embedding and retrieval workflow
- Vector databases optimize search through ANN indexing and scalable query performance

# Text Classification 🔖

In the previous stage, we noticed a major challenge: our dataset contained **567 unique categories** — too many to manage manually. Although some categories were manually grouped (e.g., "History" → "Nonfiction"), the sheer volume made full manual classification unfeasible.

---

## 🌐 The Problem

- The `category` field was messy and overly granular.
- This impacts recommendation quality and interpretability.
- We needed a scalable solution to group categories into broader, meaningful classes (e.g., Fiction vs Nonfiction).

---

## ✨ Solution: Text Classification with LLMs

- Text classification allows us to automatically categorize textual data.
- While not new (it predates LLMs), **LLMs have proven to be particularly effective** at this task due to their deep understanding of language.
- We chose to implement **Zero-Shot Text Classification**, which allows classification without additional training.

### What is Zero-Shot Classification?

- A method where a **pre-trained model** predicts categories it hasn’t seen during training.
- We only need to provide:
  - The text to be classified
  - A prompt that describes the classification task
  - Optionally, the list of possible target categories

---

## 🤔 Why Do LLMs Work Without Fine-Tuning?

- **Model used** `model="facebook/bart-large-mnli"`
- **Large Language Models** (typically with 100M+ parameters) have been exposed to a vast amount of language data.
- Their architecture (Transformers) enables them to understand semantic relationships by learning which words and topics tend to appear together.
- As a result, they can **infer** which topics or phrases are more aligned with general labels like "Fiction" or "Nonfiction" without explicit prior examples.

---

## 📈 Benefits of Zero-Shot Classification

- **No training data required** – suitable when labeled data is scarce
- **Flexible and scalable** – just change the prompt or labels to adapt the task
- **Efficient** – instant categorization without building and training a model

---

## ✅ Application in Gloria

- Applied Zero-Shot Classification to simplify the 567 categories into high-level groups
- Improved the coherence of recommendations and search filters
- Enabled a more robust semantic layer over raw category data

> This step was crucial for reducing noise and improving the quality of book recommendations by standardizing the category labels using a modern and scalable technique

# Sentiment Analysis 😊😡😭

In this phase, we implemented **sentiment analysis** to detect the emotional tone of each book description. This allows users to filter books based on emotional sentiment — whether they're looking for joyful, sad, or surprising content.

---

## 🌐 Objective

Transform sentiment analysis into a **text classification task** by predicting one of seven possible emotional categories for each book description.

### 🤖 Model Used

We use a **fine-tuned DistilRoBERTa model** specifically trained to classify English text into **7 emotional categories**:

```python
emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
```

- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Tool:** HuggingFace `pipeline()` with `text-classification` task
- **Setup:**

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
```

---

## 📊 Practical Implementation

- Each book description was passed through the classifier.
- The model returned a **probability score** for each of the seven emotions.
- These scores were saved in **seven additional columns** in the dataset.

This allowed the application to:

- Enable **emotion-based filtering** in the recommendation system.
- Example: If a user wants to read a "sad" or "joyful" book, we simply sort the corresponding sentiment column from **highest to lowest** and return the top results.

---

## ✅ Summary

- Sentiment analysis was modeled as a **multi-class classification** problem.
- A pre-trained model with strong performance was used to map descriptions to emotions.
- This enhances the recommender system by providing users with the option to explore books based on emotional tone.
