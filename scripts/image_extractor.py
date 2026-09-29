#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path

import fitz
import yaml


# ============================================================
# PATHS
# ============================================================

DOCUMENT_CONFIG = Path(
    "config/harrier_docs.yaml"
)

SOURCE_DIR = Path(
    "factory_rag2/harrier_docs"
)

OUTPUT_DIR = Path(
    "extracted_knowledge/images"
)

MANIFEST_PATH = (
    OUTPUT_DIR / "image_manifest.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_documents():

    with open(
        DOCUMENT_CONFIG,
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    documents = config.get(
        "documents",
        []
    )

    return documents


def calculate_sha256(data):

    return hashlib.sha256(
        data
    ).hexdigest()


def safe_extension(ext):

    if not ext:
        return "png"

    ext = ext.lower()

    if ext == "jpeg":
        return "jpg"

    if ext in {
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "gif",
        "tiff",
        "webp",
    }:
        return ext

    return "png"


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_document_images(
    document_id,
    source_file
):

    pdf_path = (
        SOURCE_DIR / source_file
    )

    if not pdf_path.exists():

        print(
            f"WARNING: PDF not found: "
            f"{pdf_path}"
        )

        return []

    print(
        f"\nProcessing {document_id}: "
        f"{source_file}"
    )

    document_output_dir = (
        OUTPUT_DIR / document_id
    )

    document_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf = fitz.open(
        pdf_path
    )

    records = []

    for page_index in range(
        len(pdf)
    ):

        page = pdf[
            page_index
        ]

        page_number = (
            page_index + 1
        )

        # ----------------------------------------------------
        # Get all image blocks on page.
        #
        # This gives us both the image identity and
        # its position on the page.
        # ----------------------------------------------------

        image_blocks = page.get_text(
            "dict"
        ).get(
            "blocks",
            []
        )

        page_image_index = 0

        for block in image_blocks:

            if block.get(
                "type"
            ) != 1:
                continue

            bbox = block.get(
                "bbox"
            )

            xref = block.get(
                "xref"
            )

            width = block.get(
                "width"
            )

            height = block.get(
                "height"
            )

            image_bytes = None
            extension = "png"

            # ------------------------------------------------
            # Extract actual image from PDF.
            # ------------------------------------------------

            if xref:

                try:

                    image_info = (
                        pdf.extract_image(
                            xref
                        )
                    )

                    image_bytes = (
                        image_info.get(
                            "image"
                        )
                    )

                    extension = safe_extension(
                        image_info.get(
                            "ext"
                        )
                    )

                    if not width:
                        width = image_info.get(
                            "width"
                        )

                    if not height:
                        height = image_info.get(
                            "height"
                        )

                except Exception as exc:

                    print(
                        f"  WARNING: "
                        f"Could not extract "
                        f"xref {xref} "
                        f"on page "
                        f"{page_number}: "
                        f"{exc}"
                    )

            # ------------------------------------------------
            # Some PDFs may not expose a useful xref.
            # Render the image block as a fallback.
            # ------------------------------------------------

            if image_bytes is None:

                try:

                    rect = fitz.Rect(
                        bbox
                    )

                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(
                            2,
                            2
                        ),
                        clip=rect,
                        alpha=False
                    )

                    image_bytes = (
                        pixmap.tobytes(
                            "png"
                        )
                    )

                    extension = "png"

                    width = (
                        pixmap.width
                    )

                    height = (
                        pixmap.height
                    )

                except Exception as exc:

                    print(
                        f"  WARNING: "
                        f"Could not render "
                        f"image block on "
                        f"page {page_number}: "
                        f"{exc}"
                    )

                    continue

            if not image_bytes:
                continue

            # ------------------------------------------------
            # Hash lets us identify duplicate image content.
            # ------------------------------------------------

            sha256 = calculate_sha256(
                image_bytes
            )

            image_id = (
                f"{document_id}_"
                f"p{page_number}_"
                f"img{page_image_index:02d}"
            )

            filename = (
                f"p{page_number}_"
                f"img{page_image_index:02d}."
                f"{extension}"
            )

            output_path = (
                document_output_dir
                / filename
            )

            with open(
                output_path,
                "wb"
            ) as f:

                f.write(
                    image_bytes
                )

            record = {
                "id": image_id,

                "document_id":
                    document_id,

                "source_file":
                    source_file,

                "page":
                    page_number,

                "image_index":
                    page_image_index,

                "xref":
                    xref,

                "bbox": [
                    round(
                        float(value),
                        2
                    )
                    for value in bbox
                ] if bbox else None,

                "width":
                    width,

                "height":
                    height,

                "format":
                    extension,

                "sha256":
                    sha256,

                "file":
                    str(
                        output_path
                        .relative_to(
                            Path(".")
                        )
                    ),

                # These will be populated by the
                # image-context linker later.
                "context":
                    None,

                "linked_knowledge_units":
                    [],

                "linked_tables":
                    [],
            }

            records.append(
                record
            )

            page_image_index += 1

    pdf.close()

    print(
        f"  Images extracted: "
        f"{len(records)}"
    )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "IMAGE EXTRACTION"
    )

    print(
        "======================================"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = load_documents()

    all_images = []

    for document in documents:

        document_id = document.get(
            "id"
        )

        source_file = document.get(
            "file"
        )

        if not document_id:
            print(
                "WARNING: document has "
                "no ID"
            )
            continue

        if not source_file:
            print(
                f"WARNING: {document_id} "
                f"has no source file"
            )
            continue

        images = (
            extract_document_images(
                document_id,
                source_file
            )
        )

        all_images.extend(
            images
        )

    # --------------------------------------------------------
    # Duplicate statistics
    # --------------------------------------------------------

    hashes = {}

    duplicate_count = 0

    for image in all_images:

        sha256 = image.get(
            "sha256"
        )

        if sha256 in hashes:

            duplicate_count += 1

        else:

            hashes[sha256] = (
                image.get("id")
            )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "manifest_version": "0.1",

        "description": (
            "Image inventory extracted "
            "from source PDF documents."
        ),

        "statistics": {
            "documents": len(
                documents
            ),

            "images": len(
                all_images
            ),

            "duplicate_content_images":
                duplicate_count,
        },

        "images":
            all_images,
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
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
        "IMAGE EXTRACTION COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Documents : "
        f"{len(documents)}"
    )

    print(
        f"Images    : "
        f"{len(all_images)}"
    )

    print(
        f"Duplicate image content : "
        f"{duplicate_count}"
    )

    print(
        f"\nImage directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        f"\nManifest:"
    )

    print(
        f"  {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()