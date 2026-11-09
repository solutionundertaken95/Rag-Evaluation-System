import pymupdf


def extract_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)

    full_text = ""

    for page in doc:
        full_text += page.get_text()

    return full_text