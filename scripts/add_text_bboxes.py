#!/usr/bin/env python3

import json
from pathlib import Path

import fitz


# ============================================================
# PATHS
# ============================================================

KNOWLEDGE_UNITS_PATH = Path(
    "extracted_knowledge/knowledge_units.json"
)

DOCUMENTS_CONFIG_PATH = Path(
    "config/harrier_docs.yaml"
)

OUTPUT_PATH = Path(
    "extracted_knowledge/knowledge_units.json"
)


# ============================================================
# LOAD DOCUMENT CONFIG
# ============================================================

def load_document_config():

    import yaml

    with open(
        DOCUMENTS_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    documents = config.get(
        "documents",
        []
    )

    result = {}

    for document in documents:

        document_id = document.get(
            "id"
        )

        source_file = document.get(
            "file"
        )

        if document_id and source_file:

            result[document_id] = source_file

    return result


# ============================================================
# LOAD KNOWLEDGE UNITS
# ============================================================

def load_knowledge_units():

    with open(
        KNOWLEDGE_UNITS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(
            "knowledge_units.json must contain a list."
        )

    return data


# ============================================================
# EXTRACT TEXT BLOCKS
# ============================================================

def get_page_blocks(page):

    blocks = page.get_text(
        "blocks"
    )

    result = []

    for block_index, block in enumerate(
        blocks
    ):

        if len(block) < 5:
            continue

        x0 = block[0]
        y0 = block[1]
        x1 = block[2]
        y1 = block[3]

        text = block[4]

        if not text:
            continue

        result.append({
            "block_index": block_index,

            "bbox": [
                round(float(x0), 2),
                round(float(y0), 2),
                round(float(x1), 2),
                round(float(y1), 2),
            ],

            "text": " ".join(
                str(text).split()
            ).strip(),
        })

    return result


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    return " ".join(
        str(text).split()
    ).strip().lower()


# ============================================================
# FIND MATCHING BLOCK
# ============================================================

def find_matching_block(
    unit,
    blocks
):

    unit_text = normalize_text(
        unit.get("text", "")
    )

    if not unit_text:
        return None

    # --------------------------------------------------------
    # Exact normalized match
    # --------------------------------------------------------

    for block in blocks:

        block_text = normalize_text(
            block["text"]
        )

        if block_text == unit_text:

            return block

    # --------------------------------------------------------
    # Containment match
    #
    # Useful when extraction has slightly different spacing
    # or line wrapping.
    # --------------------------------------------------------

    best_match = None
    best_score = 0

    for block in blocks:

        block_text = normalize_text(
            block["text"]
        )

        if not block_text:
            continue

        if (
            unit_text in block_text
            or block_text in unit_text
        ):

            shorter = min(
                len(unit_text),
                len(block_text)
            )

            longer = max(
                len(unit_text),
                len(block_text)
            )

            if longer == 0:
                continue

            score = shorter / longer

            if score > best_score:

                best_score = score
                best_match = block

    # Don't accept extremely weak matches.
    if best_score >= 0.70:
        return best_match

    return None


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(
    document_record,
    source_file
):

    document_id = document_record.get(
        "document_id"
    )

    units = document_record.get(
        "knowledge_units",
        []
    )

    pdf_path = (
        Path("factory_rag2/harrier_docs")
        / source_file
    )

    if not pdf_path.exists():

        print(
            f"WARNING: PDF not found: "
            f"{pdf_path}"
        )

        return 0, 0

    print(
        f"\nProcessing {document_id}: "
        f"{source_file}"
    )

    pdf = fitz.open(
        pdf_path
    )

    # --------------------------------------------------------
    # Build page -> text blocks cache
    # --------------------------------------------------------

    page_blocks = {}

    for page_index in range(
        len(pdf)
    ):

        page_number = page_index + 1

        page = pdf[
            page_index
        ]

        page_blocks[
            page_number
        ] = get_page_blocks(
            page
        )

    matched = 0
    unmatched = 0

    # --------------------------------------------------------
    # Match knowledge units to PDF blocks
    # --------------------------------------------------------

    for unit in units:

        unit_type = unit.get(
            "type"
        )

        # Navigation records are retained but we don't need
        # them for image/knowledge spatial linking.
        if unit_type == "navigation":

            unit["bbox"] = None
            unit["block_match"] = None

            continue

        page_start = unit.get(
            "page_start"
        )

        page_end = unit.get(
            "page_end"
        )

        if page_start is None:

            unit["bbox"] = None
            unit["block_match"] = None

            unmatched += 1

            continue

        if page_end is None:
            page_end = page_start

        matched_block = None
        matched_page = None

        # ----------------------------------------------------
        # Most knowledge units are page-local.
        #
        # Search from page_start to page_end.
        # ----------------------------------------------------

        for page_number in range(
            page_start,
            page_end + 1
        ):

            blocks = page_blocks.get(
                page_number,
                []
            )

            match = find_matching_block(
                unit,
                blocks
            )

            if match:

                matched_block = match
                matched_page = page_number

                break

        # ----------------------------------------------------
        # Store spatial metadata
        # ----------------------------------------------------

        if matched_block:

            unit["bbox"] = matched_block[
                "bbox"
            ]

            unit["block_match"] = {
                "page": matched_page,

                "block_index":
                    matched_block[
                        "block_index"
                    ],

                "match_text":
                    matched_block[
                        "text"
                    ],
            }

            matched += 1

        else:

            unit["bbox"] = None
            unit["block_match"] = None

            unmatched += 1

    pdf.close()

    print(
        f"  Units: {len(units)}"
    )

    print(
        f"  Matched: {matched}"
    )

    print(
        f"  Unmatched: {unmatched}"
    )

    return matched, unmatched


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "ADD TEXT SPATIAL METADATA"
    )

    print(
        "======================================"
    )

    document_files = (
        load_document_config()
    )

    data = load_knowledge_units()

    total_units = 0
    total_matched = 0
    total_unmatched = 0

    for document_record in data:

        document_id = (
            document_record.get(
                "document_id"
            )
        )

        source_file = (
            document_files.get(
                document_id
            )
        )

        if not source_file:

            print(
                f"WARNING: No source file "
                f"for {document_id}"
            )

            continue

        matched, unmatched = (
            process_document(
                document_record,
                source_file
            )
        )

        total_units += (
            matched + unmatched
        )

        total_matched += matched
        total_unmatched += unmatched

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "SPATIAL METADATA COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Total units processed : "
        f"{total_units}"
    )

    print(
        f"Matched to PDF block : "
        f"{total_matched}"
    )

    print(
        f"Unmatched            : "
        f"{total_unmatched}"
    )

    if total_units:

        percentage = (
            total_matched
            / total_units
            * 100
        )

        print(
            f"Match rate           : "
            f"{percentage:.2f}%"
        )

    print(
        f"\nOutput:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()