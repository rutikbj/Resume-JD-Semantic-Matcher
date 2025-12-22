# JD-Resume- Matcher

A modular, professional-grade Python web app for evaluating resume–job description fit using advanced Natural Language Processing (NLP) techniques—including BERT-based semantic similarity.  
Easily upload a resume and a job description (JD) as PDFs, select the similarity model, and instantly visualize how well they match, including detailed keyword analysis with data stored in csv file.

---

## Features

- **PDF Upload**: Accepts both resume and JD in PDF format; text is extracted automatically.
- **Text Extraction & Cleaning**: Reads multi-page PDFs and removes headers, footers, and non-informative text (as much as possible).
- **Advanced Preprocessing**:
    - Lowercasing, lemmatization, tokenization (using spaCy)
    - Removal of built-in and customizable stopwords (to ignore buzzwords like “opportunity” or “passion”)
    - Filter out punctuation and numbers
- **Multiple Similarity Models**:
    - **TF-IDF (Weighted Keyword Overlap)**: Classic NLP approach.
    - **Bag of Words**: Simple word overlap.
    - **Jaccard Similarity**: Unique word overlap (vocabulary match).
    - **BERT/SentenceTransformer**: State-of-the-art semantic similarity—captures meaning, not just words!
- **Keyword Analysis**:
    - **Matched Keywords**: Key skills/phrases present in both JD and resume.
    - **Unmatched Keywords**: Key skills/phrases missing from the resume, but present in JD.
- **Intuitive Web UI (Gradio)**: Analyze results, try different models, and test new files—all in your browser.
- **Highly Modular Design**: Easily extendable with new similarity models or preprocessing techniques.



---

## How It Works: End-to-End Flow

1. **User uploads two PDFs** (JD & resume).
2. **Text Extraction**: Reads and combines all pages.
3. **Preprocessing**:
    - Lowercase, tokenize, lemmatize
    - Remove common and domain-specific “filler” words
    - Outputs a cleaned, focused set of keywords
4. **Similarity Calculation**:
    - Choose a model (TF-IDF, BoW, BERT/SBERT, Jaccard)
    - The app calculates a similarity score (0 to 1)
5. **Keyword Analysis**:
    - Displays which keywords matched and which did not
    - Allows user to update their resume or JD for better fit!
6. **Results Displayed** in a clean Gradio interface.

---

## Setup & Usage

### **1.Set Up a Virtual Environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

*BERT/SentenceTransformer models are downloaded automatically on first use.*

### **3. Run the App**
```bash
python app.py
```
- Open the provided local URL in your browser.

---

## Customization & Extensibility

- **Add/Remove Filler Words:**  
  Edit `CUSTOM_STOPWORDS` in `Similarity_matcher/nlp_utils.py` to change which buzzwords are ignored.
- **Try Stronger Models:**  
  In `Similarity_matcher/similarity.py`, switch to different bigger models [SentenceTransformer model](https://www.sbert.net/docs/pretrained_models.html) for even better semantic matching.
- **Add New Similarity Functions:**  
  Just add a function to `Similarity_matcher/similarity.py` and update the `SIMILARITY_MODELS` dictionary in `app.py`.
- **Fine-Tune on Your Data:**  
  With labeled (resume, JD, label) pairs, fine-tune BERT for even higher accuracy ([SBERT training guide](https://www.sbert.net/docs/training/overview.html)).

---

## Example Use Cases

- **Recruiters:** Instantly score and rank resumes for open roles.
- **Job Seekers:** Optimize your resume for a specific job and see what skills to add.
- **HR & Analytics:** Rapidly screen large applicant pools and visualize gaps.
- **Career Services:** Advise students or clients on resume-JD fit.

---

## Model Comparison Table

| Model                    | Strengths                                | Weaknesses                          | Use Case            |
|--------------------------|------------------------------------------|-------------------------------------|---------------------|
| TF-IDF                   | Fast, classic, keyword focus             | Misses paraphrases, synonyms        | Quick keyword match |
| Bag of Words             | Simple, transparent                      | Ignores word importance/meaning     | Vocabulary overlap  |
| Jaccard                  | Unique word ratio                        | Very sensitive to vocab differences | Quick scan          |
| BERT/SBERT               | Captures context, synonyms, paraphrase   | Needs more compute, best when fine-tuned | True semantic match  |
| BERT/SBERT Advanced      | Captures context, synonyms, paraphrase   | More compute required then base BERT | True semantic match  |

---

## Example: Custom Stopwords

To filter out “filler” words, update in `Similarity_matcher/nlp_utils.py`:
```python
CUSTOM_STOPWORDS = {
    "offer", "opportunity", "outcome", "passion",
    "drive", "energetic", "dynamic", "vision", "future",
    "exciting", "enthusiastic", "fast-paced", "synergy", "values",
    
```
*This helps the app focus on skills, technologies, and experience keywords.*

---

## Why Use This App?

- **Not just keyword matching**—get *meaningful* similarity using state-of-the-art AI.
- **Instant insight:** See not only the similarity score, but what’s missing.
- **No more guesswork:** Know exactly how to tailor a resume or JD.
