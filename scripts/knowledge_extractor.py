from pathlib import Path
import json
import re
import yaml
import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "factory_rag2"/"harrier_docs"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
OUTPUT_DIR = PROJECT_ROOT / "extracted_knowledge"

CONFIG_FILE = PROJECT_ROOT / "config" / "harrier_docs.yaml"
EXTRACTION_CONFIG = PROJECT_ROOT / "config" / "extraction.yaml"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_index_page(text):
    """
    Conservative detection of index / table-of-contents pages.
    """

    text = clean_text(text).lower()

    if not text:
        return False

    indicators = [
        "table of contents",
        "contents",
        "index"
    ]

    indicator_count = sum(
        1 for item in indicators
        if item in text
    )

    # Lots of dotted page references is another strong signal.
    dotted_entries = len(
        re.findall(r"\.{3,}\s*\d+", text)
    )

    return (
        indicator_count > 0
        or dotted_entries >= 5
    )


def classify_text(text):
    """
    Initial knowledge-unit classification.

    This is deliberately conservative.
    """

    text_lower = text.lower()

    if re.search(r"\bwarning\b", text_lower):
        return "warning"

    if re.search(r"\bcaution\b", text_lower):
        return "warning"

    if re.search(r"\bnote\b", text_lower):
        return "note"

    # Procedure indicators
    procedure_patterns = [
        r"\bpress\b",
        r"\bturn\b",
        r"\bremove\b",
        r"\binsert\b",
        r"\bconnect\b",
        r"\bdisconnect\b",
        r"\badjust\b",
        r"\bselect\b",
        r"\bopen\b",
        r"\bclose\b",
        r"\breplace\b",
        r"\bcheck\b",
        r"\bset\b"
    ]

    procedure_matches = sum(
        bool(re.search(pattern, text_lower))
        for pattern in procedure_patterns
    )

    if procedure_matches >= 2:
        return "procedure"

    # Specification indicators
    specification_patterns = [
        r"\b\d+\s*(mm|cm|m|kg|g|l|litre|liter|v|a|kw|hp|ps|nm|rpm|bar|psi|°c)\b",
        r"\b\d+\s*x\s*\d+\b",
        r"\b\d+\s*cc\b"
    ]

    if any(
        re.search(pattern, text_lower)
        for pattern in specification_patterns
    ):
        return "specification"

    return "concept"


def extract_page_blocks(page):

    blocks = []

    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):

        if "lines" not in block:
            continue

        text_parts = []

        for line in block["lines"]:

            for span in line.get("spans", []):

                text = span.get("text", "")

                if text.strip():
                    text_parts.append(text)

        text = clean_text(" ".join(text_parts))

        if not text:
            continue

        blocks.append({
            "text": text,
            "bbox": block.get("bbox")
        })

    return blocks


# ---------------------------------------------------------
# Extract one document
# ---------------------------------------------------------

def extract_document(document_id, filename):

    pdf_path = SOURCE_DIR / filename

    print(f"\nProcessing: {filename}")

    if not pdf_path.exists():

        print(f"WARNING: File not found: {pdf_path}")

        return None

    doc = fitz.open(pdf_path)

    knowledge_units = []

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        page_text = clean_text(
            page.get_text("text")
        )

        # ---------------------------------------------
        # Skip index / TOC pages
        # ---------------------------------------------

        if is_index_page(page_text):

            knowledge_units.append({
                "id": f"{document_id}_page_{page_number}",
                "type": "navigation",
                "document_id": document_id,
                "source_file": filename,
                "page_start": page_number,
                "page_end": page_number,
                "text": page_text
            })

            continue

        blocks = extract_page_blocks(page)

        for block_index, block in enumerate(blocks):

            text = block["text"]

            if len(text) < 20:
                continue

            unit_type = classify_text(text)

            unit = {
                "id": (
                    f"{document_id}_"
                    f"p{page_number}_"
                    f"b{block_index}"
                ),

                "type": unit_type,

                "document_id": document_id,

                "source_file": filename,

                "page_start": page_number,

                "page_end": page_number,

                "source_block": block_index,

                "text": text
            }

            knowledge_units.append(unit)

    doc.close()

    return {
        "document_id": document_id,
        "source_file": filename,
        "knowledge_units": knowledge_units
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("KNOWLEDGE UNIT EXTRACTION")
    print("=" * 70)

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    documents = config.get(
        "documents",
        []
    )

    all_documents = []

    for document in documents:

        result = extract_document(
            document["id"],
            document["file"]
        )

        if result is None:
            continue

        all_documents.append(result)

        output_file = (
            OUTPUT_DIR /
            f"{document['id']}_knowledge.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

        units = result["knowledge_units"]

        type_counts = {}

        for unit in units:

            unit_type = unit["type"]

            type_counts[unit_type] = (
                type_counts.get(unit_type, 0) + 1
            )

        print(
            f"  Knowledge units: {len(units)}"
        )

        for unit_type, count in sorted(
            type_counts.items()
        ):

            print(
                f"    {unit_type}: {count}"
            )

    combined_file = (
        OUTPUT_DIR /
        "knowledge_units.json"
    )

    with open(
        combined_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_documents,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput:"
        f"\n{combined_file}"
    )


if __name__ == "__main__":
    main()