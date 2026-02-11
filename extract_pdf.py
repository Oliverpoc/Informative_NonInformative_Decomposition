
import pypdf
import os

pdf_path = "informative-and-non-informative-decomposition-of-turbulent-flow-fields.pdf"
try:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open("pdf_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Text extracted to pdf_text.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")
