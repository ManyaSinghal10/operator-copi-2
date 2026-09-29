#!/usr/bin/env python3

import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RECONSTRUCTED_TABLES_PATH = (
    "extracted_knowledge/tables/reconstructed_tables.json"
)

KNOWLEDGE_UNITS_PATH = (
    "extracted_knowledge/knowledge_units.json"
)

OUTPUT_PATH = (
    "extracted_knowledge/tables/contextualized_tables.json"
)


# ============================================================
# SECTION DETECTION
# ============================================================

SECTION_TYPES = {
    "concept",
    "component",
    "procedure",
    "maintenance",
    "specification",
    "warning",
    "note",
    "troubleshooting",
    "warranty",
}


def looks_like_section(text):
    """
    Heuristic detection of section headings.

    We intentionally keep this conservative because the
    extracted knowledge units contain both real headings and
    ordinary paragraphs.
    """

    if not text:
        return False

    text = " ".join(text.split()).strip()

    if len(text) < 3 or len(text) > 150:
        return False

    # Too sentence-like
    if text.endswith("."):
        return False

    words = text.split()

    # Avoid very long paragraphs
    if len(words) > 12:
        return False

    # Common sentence indicators
    sentence_words = {
        "the",
        "this",
        "that",
        "these",
        "those",
        "please",
        "ensure",
        "make",
        "do",
        "use",
        "if",
        "when",
        "avoid",
        "always",
        "never",
    }

    first_word = words[0].lower().strip(":-")

    if first_word in sentence_words:
        return False

    # Mostly uppercase headings
    letters = [c for c in text if c.isalpha()]

    if letters:
        uppercase_ratio = sum(
            1 for c in letters if c.isupper()
        ) / len(letters)

        if uppercase_ratio >= 0.70:
            return True

    # Numbered headings such as:
    # 1. INTRODUCTION
    # 2. SYSTEM OVERVIEW
    # 3.1 AUDIO SETTINGS
    if re.match(
        r"^\d+(\.\d+)*[\s.)-]+[A-Za-z]",
        text
    ):
        return True

    return False


# ============================================================
# LOAD KNOWLEDGE UNITS
# ============================================================

def load_knowledge_units(path):
    """
    knowledge_units.json structure:

    [
        {
            "document_id": "...",
            "source_file": "...",
            "knowledge_units": [
                {...},
                {...}
            ]
        },
        ...
    ]

    Flatten this into one list of individual units.
    """

    print("\nLoading knowledge units...")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flattened = []

    if not isinstance(data, list):
        raise ValueError(
            "Expected knowledge_units.json to contain a list "
            "of document records."
        )

    for document in data:

        if not isinstance(document, dict):
            continue

        document_id = document.get("document_id")
        source_file = document.get("source_file")

        units = document.get("knowledge_units", [])

        if not isinstance(units, list):
            continue

        for unit in units:

            if not isinstance(unit, dict):
                continue

            # Preserve original information.
            unit_copy = dict(unit)

            # Fill missing document-level information.
            if not unit_copy.get("document_id"):
                unit_copy["document_id"] = document_id

            if not unit_copy.get("source_file"):
                unit_copy["source_file"] = source_file

            flattened.append(unit_copy)

    print(f"Documents loaded : {len(data)}")
    print(f"Knowledge units  : {len(flattened)}")

    return flattened


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_sections(units):

    sections = []

    for unit in units:

        # Navigation / index pages should NOT become section
        # context.
        if unit.get("type") == "navigation":
            continue

        text = unit.get("text", "")

        if not looks_like_section(text):
            continue

        sections.append({
            "id": unit.get("id"),
            "document_id": unit.get("document_id"),
            "source_file": unit.get("source_file"),
            "page_start": unit.get("page_start"),
            "page_end": unit.get("page_end"),
            "text": " ".join(text.split()),
            "type": unit.get("type"),
        })

    # Sort chronologically inside each document.
    sections.sort(
        key=lambda x: (
            x.get("document_id", ""),
            x.get("page_start") or 0,
            x.get("page_end") or 0,
        )
    )

    return sections


# ============================================================
# TABLE PAGE EXTRACTION
# ============================================================

def get_table_document_id(table):

    return (
        table.get("document_id")
        or table.get("doc_id")
    )


def get_table_page_start(table):

    if table.get("page_start") is not None:
        return table.get("page_start")

    pages = table.get("pages", [])

    if pages:
        return min(pages)

    return None


def get_table_page_end(table):

    if table.get("page_end") is not None:
        return table.get("page_end")

    pages = table.get("pages", [])

    if pages:
        return max(pages)

    return None


# ============================================================
# CONTEXT LINKING
# ============================================================

def find_context_for_table(table, sections):

    document_id = get_table_document_id(table)

    table_page_start = get_table_page_start(table)

    if document_id is None or table_page_start is None:
        return None

    # Only sections belonging to the same document.
    document_sections = [
        section
        for section in sections
        if section.get("document_id") == document_id
    ]

    # Find the latest section whose starting page is before
    # or equal to the table.
    previous_sections = [
        section
        for section in document_sections
        if (
            section.get("page_start") is not None
            and section["page_start"] <= table_page_start
        )
    ]

    if not previous_sections:
        return None

    previous_sections.sort(
        key=lambda x: (
            x.get("page_start") or 0,
            x.get("page_end") or 0,
        )
    )

    return previous_sections[-1]


# ============================================================
# MAIN CONTEXT LINKING
# ============================================================

def contextualize_tables(tables, sections):

    contextualized = []

    linked = 0
    unlinked = 0

    for table in tables:

        # Make a copy so we never modify the original
        # reconstructed table object.
        result = dict(table)

        context = find_context_for_table(
            table,
            sections
        )

        if context:

            result["context"] = {
                "section_id": context.get("id"),
                "section_title": context.get("text"),
                "section_type": context.get("type"),
                "section_page_start": context.get("page_start"),
                "section_page_end": context.get("page_end"),
            }

            linked += 1

        else:

            result["context"] = None
            unlinked += 1

        contextualized.append(result)

    return contextualized, linked, unlinked


# ============================================================
# SAVE
# ============================================================

def save_output(data, path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("======================================")
    print("TABLE CONTEXT LINKER")
    print("======================================")

    # --------------------------------------------------------
    # Load reconstructed tables
    # --------------------------------------------------------

    print("\nLoading reconstructed tables...")

    with open(
        RECONSTRUCTED_TABLES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        reconstructed_data = json.load(f)

    # reconstructed_tables.json may either be:
    #
    # [
    #     {...},
    #     {...}
    # ]
    #
    # or:
    #
    # {
    #     "tables": [...]
    # }

    if isinstance(reconstructed_data, list):

        tables = reconstructed_data

    elif isinstance(reconstructed_data, dict):

        if "tables" in reconstructed_data:
            tables = reconstructed_data["tables"]

        elif "logical_tables" in reconstructed_data:
            tables = reconstructed_data["logical_tables"]

        else:
            raise ValueError(
                "Could not find tables in reconstructed_tables.json"
            )

    else:

        raise ValueError(
            "Unexpected reconstructed_tables.json structure"
        )

    print(f"Logical tables: {len(tables)}")

    # --------------------------------------------------------
    # Load knowledge units
    # --------------------------------------------------------

    units = load_knowledge_units(
        KNOWLEDGE_UNITS_PATH
    )

    # --------------------------------------------------------
    # Extract section candidates
    # --------------------------------------------------------

    sections = extract_sections(units)

    print(f"Section candidates: {len(sections)}")

    # --------------------------------------------------------
    # Link tables to sections
    # --------------------------------------------------------

    contextualized_tables, linked, unlinked = (
        contextualize_tables(
            tables,
            sections
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_output(
        contextualized_tables,
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n======================================")
    print("TABLE CONTEXT LINKING COMPLETE")
    print("======================================")

    print(f"Logical tables : {len(tables)}")
    print(f"Context linked : {linked}")
    print(f"Unlinked       : {unlinked}")

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()