#!/usr/bin/env python3

import json
import re
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "extracted_knowledge/knowledge_model.json"
)

OUTPUT_DIR = Path(
    "okf_bundle"
)

IMAGE_MANIFEST_PATH = (
    "extracted_knowledge/images/image_manifest.json"
)

OKF_IMAGE_DIR = OUTPUT_DIR / "images"

# ============================================================
# OKF CONFIGURATION
# ============================================================

OKF_VERSION = "0.2"


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text):

    if not text:
        return ""

    return " ".join(
        str(text).split()
    ).strip()


def slugify(text):

    text = clean_text(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "-",
        text
    )

    text = text.strip("-")

    return text or "unnamed"


def yaml_escape(value):

    if value is None:
        return '""'

    value = str(value)

    value = value.replace(
        "\\",
        "\\\\"
    )

    value = value.replace(
        '"',
        '\\"'
    )

    return f'"{value}"'


# ============================================================
# TYPE MAPPING
# ============================================================

TYPE_DIRECTORIES = {
    "concept": "concepts",
    "procedure": "procedures",
    "warning": "warnings",
    "specification": "specifications",
    "maintenance": "maintenance",
    "troubleshooting": "troubleshooting",
    "warranty": "warranty",
    "component": "components",
    "note": "notes",
    "table": "tables",
}


def normalize_type(unit_type):

    if unit_type in TYPE_DIRECTORIES:
        return unit_type

    return "concept"


# ============================================================
# TITLE GENERATION
# ============================================================

def title_from_unit(unit):

    title = clean_text(
        unit.get("title")
    )

    if title:
        return title

    text = clean_text(
        unit.get("text")
    )

    if not text:
        return "Untitled Knowledge"

    # Use the first sentence if available.
    first_sentence = re.split(
        r"[.!?]",
        text
    )[0].strip()

    if 3 <= len(first_sentence) <= 120:
        return first_sentence

    # Otherwise use a bounded prefix.
    if len(text) > 100:
        return text[:100].rstrip() + "..."

    return text


# ============================================================
# SOURCE INFORMATION
# ============================================================

def source_from_unit(unit):

    provenance = unit.get(
        "provenance",
        {}
    )

    return {
        "document_id": provenance.get(
            "document_id"
        ),
        "source_file": provenance.get(
            "source_file"
        ),
        "page_start": provenance.get(
            "page_start"
        ),
        "page_end": provenance.get(
            "page_end"
        ),
        "source_block": provenance.get(
            "source_block"
        ),
    }


# ============================================================
# FRONTMATTER
# ============================================================

def build_frontmatter(
    concept_id,
    concept_type,
    title,
    description,
    source_records,
    tags
):

    lines = []

    lines.append("---")

    lines.append(
        f"id: {yaml_escape(concept_id)}"
    )

    lines.append(
        f"type: {yaml_escape(concept_type)}"
    )

    lines.append(
        f"title: {yaml_escape(title)}"
    )

    lines.append(
        f"description: {yaml_escape(description)}"
    )

    lines.append(
        "tags:"
    )

    for tag in tags:
        lines.append(
            f"  - {yaml_escape(tag)}"
        )

    lines.append(
        "sources:"
    )

    for source in source_records:

        lines.append(
            "  - document_id: "
            + yaml_escape(
                source.get(
                    "document_id"
                )
            )
        )

        lines.append(
            "    source_file: "
            + yaml_escape(
                source.get(
                    "source_file"
                )
            )
        )

        lines.append(
            "    page_start: "
            + str(
                source.get(
                    "page_start"
                )
                if source.get(
                    "page_start"
                ) is not None
                else ""
            )
        )

        lines.append(
            "    page_end: "
            + str(
                source.get(
                    "page_end"
                )
                if source.get(
                    "page_end"
                ) is not None
                else ""
            )
        )

        if source.get(
            "source_block"
        ) is not None:

            lines.append(
                "    source_block: "
                + str(
                    source.get(
                        "source_block"
                    )
                )
            )

    lines.append("---")

    return "\n".join(lines)


# ============================================================
# DESCRIPTION
# ============================================================

def build_description(
    title,
    units
):

    meaningful = []

    for unit in units:

        text = clean_text(
            unit.get("text")
        )

        if not text:
            continue

        if text == title:
            continue

        meaningful.append(text)

        if len(
            " ".join(meaningful)
        ) > 300:

            break

    if not meaningful:

        return (
            f"Knowledge about {title}."
        )

    description = " ".join(
        meaningful
    )

    if len(description) > 300:
        description = (
            description[:297]
            + "..."
        )

    return description


# ============================================================
# TAG GENERATION
# ============================================================

def build_tags(
    document_id,
    concept_type,
    title
):

    tags = [
        "tata-harrier",
        concept_type,
        document_id,
    ]

    words = re.findall(
        r"[A-Za-z0-9]+",
        title.lower()
    )

    # Add useful title terms.
    for word in words:

        if len(word) < 4:
            continue

        if word in {
            "with",
            "from",
            "that",
            "this",
            "when",
            "into",
            "your",
            "have",
            "only",
            "page",
        }:
            continue

        tags.append(word)

    # Remove duplicates.
    return list(
        dict.fromkeys(tags)
    )[:12]


# ============================================================
# SECTION ID
# ============================================================

def build_section_id(
    document_id,
    section
):

    section_id = section.get(
        "id"
    )

    if section_id:
        return section_id

    title = section.get(
        "title",
        "unknown"
    )

    page = section.get(
        "page_start"
    )

    return (
        f"{document_id}_"
        f"{slugify(title)}"
        f"_p{page or 0}"
    )


# ============================================================
# UNIT GROUPING
# ============================================================

def group_units_by_section(
    document
):

    grouped = defaultdict(list)

    units = document.get(
        "units",
        []
    )

    for unit in units:

        section = unit.get(
            "section"
        )

        if not section:
            continue

        section_id = section.get(
            "id"
        )

        if not section_id:

            section_id = slugify(
                section.get(
                    "title",
                    "unknown"
                )
            )

        grouped[
            section_id
        ].append(unit)

    return grouped


# ============================================================
# TABLE GROUPING
# ============================================================

def group_tables_by_section(
    document
):

    grouped = defaultdict(list)

    tables = document.get(
        "tables",
        []
    )

    for table in tables:

        context = table.get(
            "context"
        )

        if not context:
            continue

        section_id = context.get(
            "section_id"
        )

        if not section_id:
            continue

        grouped[
            section_id
        ].append(table)

    return grouped


# ============================================================
# TABLE MARKDOWN
# ============================================================

def table_to_markdown(
    table
):

    rows = table.get(
        "table_data",
        []
    )

    if not isinstance(rows, list):
        return ""

    if not rows:
        return ""

    normalized_rows = []

    for row in rows:

        if not isinstance(row, list):
            continue

        normalized_rows.append([
            clean_text(cell)
            for cell in row
        ])

    normalized_rows = [
        row
        for row in normalized_rows
        if any(row)
    ]

    if not normalized_rows:
        return ""

    column_count = max(
        len(row)
        for row in normalized_rows
    )

    # Normalize row lengths.
    for row in normalized_rows:

        while len(row) < column_count:
            row.append("")

    header = normalized_rows[0]

    lines = []

    lines.append(
        "| "
        + " | ".join(header)
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(
            "---"
            for _ in header
        )
        + " |"
    )

    for row in normalized_rows[1:]:

        lines.append(
            "| "
            + " | ".join(row)
            + " |"
        )

    return "\n".join(lines)


# ============================================================
# BUILD SECTION DOCUMENT
# ============================================================

def build_section_markdown(
    document,
    section,
    section_units,
    section_tables,
    image_lookup
):

    document_id = document.get(
        "document_id"
    )

    title = clean_text(
        section.get(
            "title",
            "Untitled Section"
        )
    )

    section_id = build_section_id(
        document_id,
        section
    )

    # --------------------------------------------------------
    # Main concept type
    # --------------------------------------------------------

    concept_type = "concept"

    # If the section is clearly about a specialized
    # knowledge type, use that type.
    lower_title = title.lower()

    if any(
        word in lower_title
        for word in [
            "warning",
            "danger",
            "caution",
            "safety",
        ]
    ):
        concept_type = "warning"

    elif any(
        word in lower_title
        for word in [
            "maintenance",
            "service schedule",
            "service instructions",
        ]
    ):
        concept_type = "maintenance"

    elif any(
        word in lower_title
        for word in [
            "technical specification",
            "specifications",
            "dimensions",
            "fuel specification",
            "lubricant specification",
        ]
    ):
        concept_type = "specification"

    elif any(
        word in lower_title
        for word in [
            "warranty",
        ]
    ):
        concept_type = "warranty"

    elif any(
        word in lower_title
        for word in [
            "troubleshooting",
        ]
    ):
        concept_type = "troubleshooting"

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = []

    for unit in section_units:

        source = source_from_unit(
            unit
        )

        if source not in sources:
            sources.append(source)

    # Add table provenance.
    for table in section_tables:

        document_id_table = (
            table.get("document_id")
        )

        pages = table.get(
            "pages",
            []
        )

        source = {
            "document_id": document_id_table,
            "source_file": document.get(
                "source_file"
            ),
            "page_start": (
                min(pages)
                if pages
                else None
            ),
            "page_end": (
                max(pages)
                if pages
                else None
            ),
            "source_block": None,
        }

        if source not in sources:
            sources.append(source)

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    description = build_description(
        title,
        section_units
    )

    tags = build_tags(
        document_id,
        concept_type,
        title
    )

    frontmatter = build_frontmatter(
        concept_id=section_id,
        concept_type=concept_type,
        title=title,
        description=description,
        source_records=sources,
        tags=tags,
    )

    # --------------------------------------------------------
    # Markdown body
    # --------------------------------------------------------

    body = []

    body.append(
        f"# {title}"
    )

    body.append("")

    # --------------------------------------------------------
    # Knowledge content
    # --------------------------------------------------------

    if section_units:

        body.append(
            "## Knowledge"
        )

        body.append("")

        for unit in section_units:

            text = clean_text(
                unit.get("text")
            )

            if not text:
                continue

            # Don't duplicate heading itself.
            if text == title:
                continue

            unit_type = unit.get(
                "type",
                "concept"
            )

            body.append(
                f"### {unit_type.title()}"
            )

            body.append("")

            body.append(text)

            # ------------------------------------------------
            # Visual references
            # ------------------------------------------------

            images = unit.get(
                "images",
                []
            )

            if images:

                body.append("")
                body.append(
                    "#### Visual References"
                )
                body.append("")

                for image in images:

                    image_id = image.get(
                        "image_id"
                    )

                    page = image.get(
                        "page"
                    )

                    confidence = image.get(
                        "confidence"
                    )

                    if not image_id:
                        continue

                    manifest_image = (
                        image_lookup.get(
                            image_id
                        )
                    )

                    body.append(
                        f"- Image: `{image_id}` "
                        f"(page {page}, "
                        f"confidence {confidence})"
                    )

                    if manifest_image:

                        image_file = (
                            manifest_image.get(
                                "file"
                            )
                        )

                        if image_file:

                            bundle_image_path = (
                                Path("images")
                                / Path(
                                    image_file
                                ).relative_to(
                                    "extracted_knowledge/images"
                                )
                            ).as_posix()

                            body.append(
                                f"  - File: "
                                f"`{bundle_image_path}`"
                            )

            body.append("")

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    valid_tables = []

    for table in section_tables:

        markdown = table_to_markdown(
            table
        )

        if markdown:

            valid_tables.append(
                (
                    table,
                    markdown
                )
            )

    if valid_tables:

        body.append(
            "## Tables"
        )

        body.append("")

        for index, (
            table,
            markdown
        ) in enumerate(
            valid_tables,
            start=1
        ):

            body.append(
                f"### Table {index}"
            )

            body.append("")

            body.append(markdown)

            body.append("")

            pages = table.get(
                "pages",
                []
            )

            if pages:

                body.append(
                    "_Source pages: "
                    + ", ".join(
                        str(p)
                        for p in pages
                    )
                    + "_"
                )

                body.append("")

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    body.append(
        "## Provenance"
    )

    body.append("")

    body.append(
        f"- Document: `{document_id}`"
    )

    body.append(
        f"- Source file: "
        f"`{document.get('source_file')}`"
    )

    if section.get(
        "page_start"
    ) is not None:

        body.append(
            f"- Section pages: "
            f"{section.get('page_start')}"
            f"-"
            f"{section.get('page_end') or section.get('page_start')}"
        )

    body.append("")

    return (
        frontmatter
        + "\n\n"
        + "\n".join(body)
        + "\n"
    )


# ============================================================
# GENERATE INDEX
# ============================================================

def generate_index(
    model,
    generated_concepts
):

    lines = []

    lines.append(
        "# Tata Harrier Knowledge Base"
    )

    lines.append("")

    lines.append(
        "This OKF bundle contains "
        "knowledge extracted from the "
        "provided Tata Harrier documentation."
    )

    lines.append("")

    lines.append(
        "## Documents"
    )

    lines.append("")

    for document_id, document in (
        model.get(
            "documents",
            {}
        ).items()
    ):

        source_file = document.get(
            "source_file"
        )

        lines.append(
            f"- **{document_id}** — "
            f"{source_file}"
        )

    lines.append("")

    lines.append(
        "## Concepts"
    )

    lines.append("")

    for concept in generated_concepts:

        relative_path = concept[
            "relative_path"
        ]

        title = concept[
            "title"
        ]

        lines.append(
            f"- [{title}]"
            f"({relative_path})"
        )

    lines.append("")

    lines.append(
        "## Statistics"
    )

    lines.append("")

    stats = model.get(
        "statistics",
        {}
    )

    for key, value in stats.items():

        lines.append(
            f"- {key}: {value}"
        )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# GENERATE LOG
# ============================================================

def generate_log():

    return """# Knowledge Base Generation Log

This bundle was generated from the extracted knowledge model.

The source documents and extracted provenance are preserved
in the generated concept frontmatter.

Tables that could not be confidently contextualized were
preserved in the intermediate extraction outputs and are
not silently discarded.

"""


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "OKF BUNDLE GENERATOR"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading knowledge model..."
    )

    with open(
        MODEL_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        model = json.load(f)

    with open(
        IMAGE_MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        image_manifest = json.load(f)

    image_lookup = {
        image["id"]: image
        for image in image_manifest.get("images", [])
    }

    documents = model.get(
        "documents",
        {}    
    )

    print(
        f"Documents: {len(documents)}"
    )

    # --------------------------------------------------------
    # Clean previous generated bundle
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        print(
            "\nRemoving previous OKF bundle..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OKF_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy extracted images into OKF bundle
    # --------------------------------------------------------

    source_image_root = Path(
        "extracted_knowledge/images"
    )

    if source_image_root.exists():

        for image in image_manifest.get("images", []):

            image_file = image.get("file")

            if not image_file:
                continue

            source_path = Path(image_file)

            if not source_path.exists():
                continue

            relative_image_path = Path(
                image_file
            ).relative_to(
                source_image_root.parent
            )

            destination_path = (
                OKF_IMAGE_DIR
                / relative_image_path.relative_to("images")
            )

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                source_path,
                destination_path
            )
    
    generated_concepts = []

    # --------------------------------------------------------
    # Process documents
    # --------------------------------------------------------

    for document_id, document in (
        documents.items()
    ):

        print(
            f"\nProcessing {document_id}"
        )

        sections = document.get(
            "sections",
            []
        )

        units_by_section = (
            group_units_by_section(
                document
            )
        )

        tables_by_section = (
            group_tables_by_section(
                document
            )
        )

        for section in sections:

            section_id = build_section_id(
                document_id,
                section
            )

            section_units = (
                units_by_section.get(
                    section_id,
                    []
                )
            )

            # Some generated section IDs may differ
            # from the original unit section IDs.
            if not section_units:

                original_id = section.get(
                    "id"
                )

                if original_id:

                    section_units = (
                        units_by_section.get(
                            original_id,
                            []
                        )
                    )

            section_tables = (
                tables_by_section.get(
                    section_id,
                    []
                )
            )

            title = clean_text(
                section.get(
                    "title",
                    "Untitled Section"
                )
            )

            content = (
                build_section_markdown(
                    document=document,
                    section=section,
                    section_units=section_units,
                    section_tables=section_tables,
                    image_lookup=image_lookup
                )
            )

            concept_dir = (
                OUTPUT_DIR
                / "concepts"
                / document_id
            )

            concept_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            filename = (
                slugify(title)
                + ".md"
            )

            output_path = (
                concept_dir
                / filename
            )

            # Avoid accidental overwrite if the same
            # section title occurs multiple times.
            if output_path.exists():

                page = (
                    section.get(
                        "page_start"
                    )
                    or 0
                )

                filename = (
                    slugify(title)
                    + f"-p{page}.md"
                )

                output_path = (
                    concept_dir
                    / filename
                )

            output_path.write_text(
                content,
                encoding="utf-8"
            )

            relative_path = (
                output_path
                .relative_to(
                    OUTPUT_DIR
                )
                .as_posix()
            )

            generated_concepts.append({
                "id": section_id,
                "title": title,
                "document_id": document_id,
                "relative_path":
                    relative_path,
            })

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    index_content = generate_index(
        model,
        generated_concepts
    )

    (
        OUTPUT_DIR
        / "index.md"
    ).write_text(
        index_content,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    (
        OUTPUT_DIR
        / "log.md"
    ).write_text(
        generate_log(),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "OKF GENERATION COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Documents processed : "
        f"{len(documents)}"
    )

    print(
        f"OKF concepts        : "
        f"{len(generated_concepts)}"
    )

    print(
        f"Output directory    : "
        f"{OUTPUT_DIR}"
    )

    print(
        "\nGenerated structure:"
    )

    print(
        "okf_bundle/"
    )

    print(
        "├── index.md"
    )

    print(
        "├── log.md"
    )

    print(
        "└── concepts/"
    )

    for document_id in documents:

        print(
            f"    └── {document_id}/"
        )


if __name__ == "__main__":
    main()