#!/usr/bin/env python3

import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

KNOWLEDGE_UNITS_PATH = (
    "extracted_knowledge/knowledge_units.json"
)

CONTEXTUALIZED_TABLES_PATH = (
    "extracted_knowledge/tables/contextualized_tables.json"
)

OUTPUT_PATH = (
    "extracted_knowledge/knowledge_model.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# These are extraction types that can become meaningful
# knowledge entities.
KNOWLEDGE_TYPES = {
    "concept",
    "procedure",
    "component",
    "warning",
    "note",
    "specification",
    "maintenance",
    "warranty",
    "troubleshooting",
}


# Navigation/index records are deliberately excluded from
# the knowledge model.
IGNORED_TYPES = {
    "navigation",
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text):
    if not text:
        return ""

    return " ".join(str(text).split()).strip()


def slugify(text):
    """
    Convert a title into a stable filesystem/OKF-friendly ID.
    """

    text = clean_text(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "-",
        text
    )

    text = text.strip("-")

    return text or "unnamed"


# ============================================================
# LOAD KNOWLEDGE UNITS
# ============================================================

def load_knowledge_units():

    print("\nLoading knowledge units...")

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

    units = []

    for document in data:

        if not isinstance(document, dict):
            continue

        document_id = document.get("document_id")
        source_file = document.get("source_file")

        document_units = document.get(
            "knowledge_units",
            []
        )

        if not isinstance(document_units, list):
            continue

        for unit in document_units:

            if not isinstance(unit, dict):
                continue

            unit_copy = dict(unit)

            unit_copy.setdefault(
                "document_id",
                document_id
            )

            unit_copy.setdefault(
                "source_file",
                source_file
            )

            if unit_copy.get("type") in IGNORED_TYPES:
                continue

            text = clean_text(
                unit_copy.get("text", "")
            )

            if not text:
                continue

            unit_copy["text"] = text

            units.append(unit_copy)

    print(
        f"Knowledge units loaded: {len(units)}"
    )

    return units


# ============================================================
# LOAD TABLES
# ============================================================

def load_tables():

    print("\nLoading contextualized tables...")

    with open(
        CONTEXTUALIZED_TABLES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        tables = json.load(f)

    if isinstance(tables, dict):

        if "tables" in tables:
            tables = tables["tables"]

        elif "logical_tables" in tables:
            tables = tables["logical_tables"]

        else:
            raise ValueError(
                "Could not find table list."
            )

    if not isinstance(tables, list):
        raise ValueError(
            "contextualized_tables.json must contain a list."
        )

    print(
        f"Contextualized tables loaded: {len(tables)}"
    )

    return tables


# ============================================================
# DOCUMENT METADATA
# ============================================================

def build_document_metadata(units):

    documents = {}

    for unit in units:

        document_id = unit.get("document_id")

        if not document_id:
            continue

        if document_id not in documents:

            documents[document_id] = {
                "document_id": document_id,
                "source_file": unit.get(
                    "source_file"
                ),
            }

    return documents


# ============================================================
# SECTION DETECTION
# ============================================================

def looks_like_heading(text):

    text = clean_text(text)

    if not text:
        return False

    if len(text) > 150:
        return False

    words = text.split()

    if len(words) > 12:
        return False

    # Sentence-like content is generally not a heading.
    if text.endswith("."):
        return False

    # Common prose starts.
    prose_starts = {
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

    if words[0].lower() in prose_starts:
        return False

    letters = [
        c for c in text
        if c.isalpha()
    ]

    if letters:

        uppercase_ratio = (
            sum(c.isupper() for c in letters)
            / len(letters)
        )

        if uppercase_ratio >= 0.70:
            return True

    # Numbered headings.
    if re.match(
        r"^\d+(\.\d+)*[\s.)-]+[A-Za-z]",
        text
    ):
        return True

    return False


def build_sections(units):

    sections_by_document = defaultdict(list)

    for unit in units:

        text = clean_text(
            unit.get("text", "")
        )

        if not looks_like_heading(text):
            continue

        document_id = unit.get(
            "document_id"
        )

        if not document_id:
            continue

        sections_by_document[
            document_id
        ].append({
            "id": unit.get("id"),
            "title": text,
            "page_start": unit.get(
                "page_start"
            ),
            "page_end": unit.get(
                "page_end"
            ),
            "source_block": unit.get(
                "source_block"
            ),
        })

    for document_id in sections_by_document:

        sections_by_document[
            document_id
        ].sort(
            key=lambda x: (
                x.get("page_start") or 0,
                x.get("page_end") or 0
            )
        )

    return sections_by_document


# ============================================================
# ASSIGN UNITS TO SECTIONS
# ============================================================

def find_section(
    unit,
    sections
):

    page = unit.get("page_start")

    if page is None:
        return None

    candidates = [
        section
        for section in sections
        if (
            section.get("page_start") is not None
            and section["page_start"] <= page
        )
    ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x.get("page_start") or 0,
            x.get("page_end") or 0
        )
    )

    return candidates[-1]


# ============================================================
# BUILD UNIT MODEL
# ============================================================

def build_unit_model(
    unit,
    sections
):

    section = find_section(
        unit,
        sections
    )

    result = {
        "id": unit.get("id"),
        "type": unit.get("type"),
        "title": None,
        "text": unit.get("text"),
        "images": unit.get("images", []),
        "provenance": {
            "document_id": unit.get(
                "document_id"
            ),
            "source_file": unit.get(
                "source_file"
            ),
            "page_start": unit.get(
                "page_start"
            ),
            "page_end": unit.get(
                "page_end"
            ),
            "source_block": unit.get(
                "source_block"
            ),
        },
    }

    # A heading-like concept becomes a candidate title.
    if looks_like_heading(
        unit.get("text", "")
    ):
        result["title"] = clean_text(
            unit.get("text")
        )

    if section:

        result["section"] = {
            "id": section.get("id"),
            "title": section.get("title"),
            "page_start": section.get(
                "page_start"
            ),
            "page_end": section.get(
                "page_end"
            ),
        }

    else:

        result["section"] = None

    return result


# ============================================================
# TABLE MODEL
# ============================================================

def get_table_id(table):

    return (
        table.get("id")
        or table.get("table_id")
        or table.get("logical_table_id")
    )


def get_table_pages(table):

    pages = table.get("pages")

    if isinstance(pages, list) and pages:
        return pages

    page_start = table.get(
        "page_start"
    )

    page_end = table.get(
        "page_end"
    )

    if page_start is not None:

        if page_end is None:
            page_end = page_start

        return list(
            range(
                page_start,
                page_end + 1
            )
        )

    return []


def build_table_model(table):

    context = table.get(
        "context"
    )

    return {
        "id": get_table_id(table),
        "type": "table",

        "document_id": (
            table.get("document_id")
            or table.get("doc_id")
        ),

        "pages": get_table_pages(table),

        "context": context,

        "source": {
            "fragments": table.get(
                "fragments",
                table.get(
                    "source_tables",
                    []
                )
            )
        },

        # Preserve original table structure.
        "table_data": table.get(
            "rows",
            table.get(
                "data",
                []
            )
        ),
    }


# ============================================================
# DOCUMENT MODEL
# ============================================================

def build_document_model(
    document_id,
    source_file,
    units,
    tables,
    sections
):

    document_units = [
        build_unit_model(
            unit,
            sections
        )
        for unit in units
        if unit.get(
            "document_id"
        ) == document_id
    ]

    document_tables = [
        build_table_model(table)
        for table in tables
        if (
            table.get("document_id")
            or table.get("doc_id")
        ) == document_id
    ]

    # Group units by knowledge type.
    by_type = defaultdict(list)

    for unit in document_units:

        unit_type = unit.get(
            "type",
            "concept"
        )

        by_type[unit_type].append(
            unit
        )

    # Group units by section.
    by_section = defaultdict(list)

    for unit in document_units:

        section = unit.get(
            "section"
        )

        if section:

            section_id = (
                section.get("id")
                or slugify(
                    section.get(
                        "title",
                        "unknown"
                    )
                )
            )

            by_section[
                section_id
            ].append(unit)

    return {
        "document_id": document_id,

        "source_file": source_file,

        "statistics": {
            "knowledge_units": len(
                document_units
            ),
            "tables": len(
                document_tables
            ),
            "sections": len(
                sections
            ),
        },

        "sections": sections,

        "units": document_units,

        "units_by_type": dict(
            by_type
        ),

        "units_by_section": dict(
            by_section
        ),

        "tables": document_tables,
    }


# ============================================================
# BUILD COMPLETE KNOWLEDGE MODEL
# ============================================================

def build_knowledge_model(
    units,
    tables
):

    documents = build_document_metadata(
        units
    )

    sections_by_document = build_sections(
        units
    )

    model = {
        "model_version": "0.1",

        "description": (
            "Intermediate knowledge model derived "
            "from extracted vehicle documentation."
        ),

        "documents": {},

        "statistics": {
            "documents": len(documents),
            "knowledge_units": len(units),
            "tables": len(tables),
            "section_candidates": sum(
                len(v)
                for v in sections_by_document.values()
            ),
        },
    }

    for document_id, metadata in documents.items():

        model["documents"][
            document_id
        ] = build_document_model(
            document_id=document_id,
            source_file=metadata.get(
                "source_file"
            ),
            units=units,
            tables=tables,
            sections=sections_by_document.get(
                document_id,
                []
            ),
        )

    return model


# ============================================================
# SAVE
# ============================================================

def save_model(model):

    Path(
        OUTPUT_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            model,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )
    print(
        "KNOWLEDGE MODELER"
    )
    print(
        "======================================"
    )

    units = load_knowledge_units()

    tables = load_tables()

    print(
        "\nBuilding knowledge model..."
    )

    model = build_knowledge_model(
        units,
        tables
    )

    save_model(model)

    print(
        "\n======================================"
    )
    print(
        "KNOWLEDGE MODEL COMPLETE"
    )
    print(
        "======================================"
    )

    print(
        f"Documents          : "
        f"{model['statistics']['documents']}"
    )

    print(
        f"Knowledge units    : "
        f"{model['statistics']['knowledge_units']}"
    )

    print(
        f"Tables             : "
        f"{model['statistics']['tables']}"
    )

    print(
        f"Section candidates : "
        f"{model['statistics']['section_candidates']}"
    )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()