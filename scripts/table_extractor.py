from pathlib import Path
import json
import re

import fitz
import pdfplumber
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "factory_rag2" / "harrier_docs"
OUTPUT_DIR = PROJECT_ROOT / "extracted_knowledge" / "tables"

CONFIG_FILE = PROJECT_ROOT / "config" / "harrier_docs.yaml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def clean_cell(value):
    if value is None:
        return ""

    value = str(value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_row(row):
    return [
        clean_cell(cell)
        for cell in row
    ]


def row_signature(row):
    """
    Used later for detecting repeated table headers.
    """

    return tuple(
        cell.lower()
        for cell in row
    )


# ---------------------------------------------------------
# Extract tables from one PDF
# ---------------------------------------------------------

def extract_tables(pdf_path, document_id):

    print(f"\nProcessing tables: {pdf_path.name}")

    tables = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            try:

                page_tables = page.extract_tables()

            except Exception as exc:

                print(
                    f"  Table extraction failed "
                    f"on page {page_number}: {exc}"
                )

                continue

            for table_index, raw_table in enumerate(
                page_tables
            ):

                if not raw_table:
                    continue

                rows = []

                for raw_row in raw_table:

                    row = normalize_row(raw_row)

                    # Ignore completely empty rows
                    if not any(row):
                        continue

                    rows.append(row)

                if not rows:
                    continue

                max_columns = max(
                    len(row)
                    for row in rows
                )

                # Normalize row lengths
                for row in rows:

                    while len(row) < max_columns:
                        row.append("")

                table_id = (
                    f"{document_id}_"
                    f"p{page_number}_"
                    f"table{table_index}"
                )

                table = {
                    "table_id": table_id,
                    "document_id": document_id,
                    "source_file": pdf_path.name,
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_table_index": table_index,
                    "columns": max_columns,
                    "rows": len(rows),
                    "data": rows
                }

                tables.append(table)

    return tables


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("TABLE EXTRACTION")
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

    all_tables = []

    for document in documents:

        document_id = document["id"]
        filename = document["file"]

        pdf_path = SOURCE_DIR / filename

        if not pdf_path.exists():

            print(
                f"\nWARNING: File not found:"
                f"\n{pdf_path}"
            )

            continue

        tables = extract_tables(
            pdf_path,
            document_id
        )

        all_tables.extend(tables)

        output_file = (
            OUTPUT_DIR /
            f"{document_id}_tables.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                tables,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"  Tables found: {len(tables)}"
        )

    combined_file = (
        OUTPUT_DIR /
        "all_tables.json"
    )

    with open(
        combined_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_tables,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("TABLE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal tables:"
        f" {len(all_tables)}"
    )

    print(
        f"\nOutput:"
        f"\n{combined_file}"
    )


if __name__ == "__main__":
    main()