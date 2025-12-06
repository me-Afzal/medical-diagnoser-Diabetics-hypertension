import streamlit as st
import fitz
import numpy as np
import easyocr
from docx import Document
import re

def extract_text_from_file(upload):
    ext = upload.name.lower()

    # TXT
    if ext.endswith(".txt"):
        return upload.read().decode("utf-8", errors="ignore")

    # DOCX
    if ext.endswith(".docx"):
        doc = Document(upload)
        return "\n".join([p.text for p in doc.paragraphs])

    # PDF
    if ext.endswith(".pdf"):
        upload.seek(0)
        pdf_data = upload.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")

        text = ""
        for page in doc:
            t = page.get_text()
            if t:
                text += t

        if len(text.strip()) < 100:  # Fallback to OCR if scanned PDF
            st.info("Scanning PDF using OCR...")
            reader = easyocr.Reader(['en'], gpu=False)
            ocr_text = []
            for page in doc:
                pix = page.get_pixmap()
                img = np.frombuffer(
                    pix.samples, dtype=np.uint8
                ).reshape(pix.height, pix.width, 3)
                result = reader.readtext(img, detail=0, paragraph=True)
                ocr_text.extend(result)
            text = "\n".join(ocr_text)

        return text

    return None


def clean_text(text: str):
    """Remove noisy characters and unnecessary tokens."""
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-English chars
    text = text.replace(":", ": ").strip()
    return text
