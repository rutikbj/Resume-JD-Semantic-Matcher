import csv
import os
from datetime import datetime
import gradio as gr
from Similarity_matcher.pdf_utils import extract_text_from_pdf
from Similarity_matcher.nlp_utils import preprocess, match_keywords
from Similarity_matcher.similarity import (
    tfidf_similarity, bow_similarity, bert_similarity,bert_similarity_advanced,
     jaccard_similarity
)

Results_File = "JD_Resume_Similarity.csv"

SIMILARITY_MODELS = {
    "TF-IDF": tfidf_similarity,
    "Bag of Words": bow_similarity,
    "BERT (SentenceTransformer)": bert_similarity,
    "BERT ADVANCED (SentenceTransformer)":bert_similarity_advanced,
    "Jaccard (token overlap)": jaccard_similarity,
}

def analyze_pdfs(job_pdf, resume_pdf, model_choice):
    job_text = extract_text_from_pdf(job_pdf)
    resume_text = extract_text_from_pdf(resume_pdf)
    job_processed = preprocess(job_text)
    resume_processed = preprocess(resume_text)
    sim_func = SIMILARITY_MODELS[model_choice]
    similarity_score = sim_func(job_processed, resume_processed)
    matched, unmatched = match_keywords(job_processed, resume_processed)

    jd_name = getattr(job_pdf, "name", "JD.pdf")      # Gradio's file object has .name
    resume_name = getattr(resume_pdf, "name", "Resume.pdf")
    save_result(jd_name, resume_name, model_choice, f"{similarity_score:.2f}", matched, unmatched)
    

    return (
        f"{similarity_score:.2f}",
        ', '.join(sorted(matched)),
        ', '.join(sorted(unmatched))
    )

def save_result(jd_name, resume_name, model, score, matched, unmatched):
    # Check if the CSV file already exists
    file_exists = os.path.isfile(Results_File)
    # Open the file in append mode (creates file if doesn't exist)
    with open(Results_File, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # If the file is new, write the header row first
        if not file_exists:
            writer.writerow([
                "timestamp", "jd_file", "resume_file", "model",
                "similarity_score", "matched_keywords", "unmatched_keywords"
            ])
        # Write the new row with all the data
        writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            jd_name, resume_name, model,
            score,
            ';'.join(sorted(matched)),
            ';'.join(sorted(unmatched))
        ])


with gr.Blocks() as demo:
    gr.Markdown("# Resume & JD Similarity Matcher\nUpload a job description PDF and a resume PDF. Choose the similarity model you want to use. The app will output the similarity score, matched keywords, and unmatched keywords (from the job description).")
    with gr.Row():
        job_pdf = gr.File(label="Job Description (PDF)")
        resume_pdf = gr.File(label="Resume (PDF)")
    model_choice = gr.Dropdown(
        choices=list(SIMILARITY_MODELS.keys()),
        label="Select Similarity Model",
        value="BERT (SentenceTransformer)"
    )
    btn = gr.Button("Analyze")
    similarity_score = gr.Textbox(label="Similarity Score")
    matched_keywords = gr.Textbox(label="Matched Keywords")
    unmatched_keywords = gr.Textbox(label="Unmatched (from JD)")
    btn.click(analyze_pdfs, inputs=[job_pdf, resume_pdf, model_choice],
              outputs=[similarity_score, matched_keywords, unmatched_keywords])
demo.launch()