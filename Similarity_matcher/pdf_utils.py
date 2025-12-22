import fitz

def extract_text_from_pdf(pdf_file):
    with fitz.open(pdf_file.name) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text