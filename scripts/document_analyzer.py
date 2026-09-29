from pathlib import Path
import json
import re

import fitz  # PyMuPDF
import pdfplumber
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "factory_rag2"/"harrier_docs"
CONFIG_FILE = PROJECT_ROOT / "config" / "harrier_docs.yaml"
OUTPUT_DIR = PROJECT_ROOT / "analysis"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def looks_like_heading(text, font_size, avg_font_size):
    """
    Heuristic heading detection.

    We deliberately don't hardcode specific heading names.
    """

    text = clean_text(text)

    if not text:
        return False

    if len(text) > 150:
        return False

    # Very large relative to normal body text
    if font_size >= avg_font_size * 1.25:
        return True

    # Short all-uppercase text
    if (
        len(text) <= 100
        and text.upper() == text
        and re.search(r"[A-Z]", text)
    ):
        return True

    # Numbered sections such as:
    # 1 Introduction
    # 1.1 Overview
    # 2.3.4 Settings
    if re.match(
        r"^\d+(\.\d+)*[\s\-\.:)]",
        text
    ):
        return True

    return False


# ---------------------------------------------------------
# Analyze one PDF
# ---------------------------------------------------------

def analyze_pdf(pdf_path):

    print(f"\nAnalyzing: {pdf_path.name}")

    result = {
        "file": pdf_path.name,
        "path": str(pdf_path),
        "page_count": 0,
        "total_characters": 0,
        "pages": [],
        "headings": [],
        "tables": [],
        "warnings": [],
        "sections": []
    }

    # -----------------------------------------------------
    # PyMuPDF analysis
    # -----------------------------------------------------

    doc = fitz.open(pdf_path)

    result["page_count"] = len(doc)

    # Collect font sizes first
    font_sizes = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:

            if "lines" not in block:
                continue

            for line in block["lines"]:

                for span in line["spans"]:

                    text = span.get("text", "").strip()

                    if text:
                        font_sizes.append(span["size"])

    avg_font_size = (
        sum(font_sizes) / len(font_sizes)
        if font_sizes
        else 10
    )

    # -----------------------------------------------------
    # Page-level analysis
    # -----------------------------------------------------

    for page_number, page in enumerate(doc, start=1):

        page_text = page.get_text("text")
        page_text_clean = clean_text(page_text)

        result["total_characters"] += len(page_text_clean)

        page_info = {
            "page": page_number,
            "characters": len(page_text_clean),
            "text_preview": page_text_clean[:300]
        }

        result["pages"].append(page_info)

        # -------------------------------------------------
        # Detect headings
        # -------------------------------------------------

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:

            if "lines" not in block:
                continue

            for line in block["lines"]:

                line_text_parts = []
                line_font_sizes = []

                for span in line["spans"]:

                    text = span.get("text", "").strip()

                    if text:
                        line_text_parts.append(text)
                        line_font_sizes.append(span["size"])

                if not line_text_parts:
                    continue

                line_text = clean_text(
                    " ".join(line_text_parts)
                )

                font_size = (
                    max(line_font_sizes)
                    if line_font_sizes
                    else avg_font_size
                )

                if looks_like_heading(
                    line_text,
                    font_size,
                    avg_font_size
                ):

                    heading = {
                        "page": page_number,
                        "text": line_text,
                        "font_size": round(font_size, 2)
                    }

                    result["headings"].append(heading)

                    # Keep unique section names
                    if line_text not in result["sections"]:
                        result["sections"].append(line_text)

        # -------------------------------------------------
        # Detect obvious warning/caution text
        # -------------------------------------------------

        lower_text = page_text_clean.lower()

        warning_keywords = [
            "warning",
            "caution",
            "danger",
            "important",
            "note"
        ]

        for keyword in warning_keywords:

            if re.search(
                rf"\b{re.escape(keyword)}\b",
                lower_text
            ):

                result["warnings"].append({
                    "page": page_number,
                    "keyword": keyword
                })

    doc.close()

    # -----------------------------------------------------
    # Table analysis using pdfplumber
    # -----------------------------------------------------

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                try:
                    tables = page.find_tables()

                    for table_index, table in enumerate(
                        tables
                    ):

                        bbox = table.bbox

                        result["tables"].append({
                            "page": page_number,
                            "table_index": table_index,
                            "bbox": [
                                round(x, 2)
                                for x in bbox
                            ],
                            "rows": len(table.rows),
                            "columns": (
                                max(
                                    len(row.cells)
                                    for row in table.rows
                                )
                                if table.rows
                                else 0
                            )
                        })

                except Exception as e:

                    print(
                        f"  Table detection failed "
                        f"on page {page_number}: {e}"
                    )

    except Exception as e:

        print(
            f"  Could not open with pdfplumber: {e}"
        )

    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("DOCUMENT STRUCTURAL ANALYZER")
    print("=" * 70)

    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            f"Missing config file: {CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    documents = config.get("documents", [])

    if not documents:

        raise ValueError(
            "No documents found in documents.yaml"
        )

    all_results = []

    for document in documents:

        doc_id = document["id"]
        filename = document["file"]

        pdf_path = SOURCE_DIR / filename

        if not pdf_path.exists():

            print(
                f"\nWARNING: File not found:"
                f"\n  {pdf_path}"
            )

            continue

        analysis = analyze_pdf(pdf_path)

        analysis["document_id"] = doc_id

        all_results.append(analysis)

        # Individual report
        output_file = (
            OUTPUT_DIR /
            f"{doc_id}_structure.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                analysis,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"  Pages: {analysis['page_count']}"
        )

        print(
            f"  Headings: {len(analysis['headings'])}"
        )

        print(
            f"  Tables: {len(analysis['tables'])}"
        )

        print(
            f"  Warnings/notes: "
            f"{len(analysis['warnings'])}"
        )

    # -----------------------------------------------------
    # Combined report
    # -----------------------------------------------------

    combined_file = (
        OUTPUT_DIR /
        "document_structure_report.json"
    )

    with open(
        combined_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nCombined report:"
        f"\n{combined_file}"
    )


if __name__ == "__main__":
    main()