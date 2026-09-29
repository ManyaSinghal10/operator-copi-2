import json
from pathlib import Path
import fitz


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOC_CONFIG = BASE_DIR / "config" / "harrier_docs.yaml"

KNOWLEDGE_UNITS = (
    BASE_DIR / "extracted_knowledge" / "knowledge_units.json"
)

IMAGE_MANIFEST = (
    BASE_DIR
    / "extracted_knowledge"
    / "images"
    / "image_manifest.json"
)

SOURCE_DIR = BASE_DIR / "factory_rag2"/"harrier_docs"


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

# Maximum vertical distance between a knowledge block
# and an image to consider them related.
MAX_VERTICAL_DISTANCE = 300

# Horizontal overlap/proximity tolerance.
HORIZONTAL_TOLERANCE = 100

# Minimum confidence required for linking.
MIN_CONFIDENCE = 0.45


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def center(bbox):
    x1, y1, x2, y2 = bbox
    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def vertical_distance(unit_bbox, image_bbox):
    ux1, uy1, ux2, uy2 = unit_bbox
    ix1, iy1, ix2, iy2 = image_bbox

    # Image below unit
    if iy1 >= uy2:
        return iy1 - uy2

    # Image above unit
    if uy1 >= iy2:
        return uy1 - iy2

    # Vertical overlap
    return 0


def horizontal_distance(unit_bbox, image_bbox):
    ux1, uy1, ux2, uy2 = unit_bbox
    ix1, iy1, ix2, iy2 = image_bbox

    # Horizontal overlap
    if not (ix1 > ux2 or ix2 < ux1):
        return 0

    if ix1 > ux2:
        return ix1 - ux2

    return ux1 - ix2


def calculate_confidence(unit_bbox, image_bbox):
    vdist = vertical_distance(unit_bbox, image_bbox)
    hdist = horizontal_distance(unit_bbox, image_bbox)

    if vdist > MAX_VERTICAL_DISTANCE:
        return 0.0

    if hdist > HORIZONTAL_TOLERANCE:
        return 0.0

    # Vertical score
    vertical_score = max(
        0.0,
        1.0 - (vdist / MAX_VERTICAL_DISTANCE)
    )

    # Horizontal score
    horizontal_score = max(
        0.0,
        1.0 - (hdist / HORIZONTAL_TOLERANCE)
    )

    confidence = (
        0.7 * vertical_score +
        0.3 * horizontal_score
    )

    return round(confidence, 3)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("IMAGE → KNOWLEDGE LINKER")
print("-" * 60)

print("Loading knowledge units...")

with open(KNOWLEDGE_UNITS, "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)

print("Loading image manifest...")

with open(IMAGE_MANIFEST, "r", encoding="utf-8") as f:
    image_manifest = json.load(f)


# ---------------------------------------------------------
# FLATTEN KNOWLEDGE UNITS
# ---------------------------------------------------------

all_units = []

for document in knowledge_data:

    document_id = document.get("document_id")

    for unit in document.get("knowledge_units", []):

        if unit.get("type") == "navigation":
            continue

        if not unit.get("bbox"):
            continue

        unit["_document_id"] = document_id

        all_units.append(unit)


print(f"Knowledge units with spatial data: {len(all_units)}")


# ---------------------------------------------------------
# GROUP UNITS BY DOCUMENT + PAGE
# ---------------------------------------------------------

units_by_page = {}

for unit in all_units:

    document_id = unit["_document_id"]
    page = unit.get("page_start")

    if page is None:
        continue

    key = (document_id, page)

    units_by_page.setdefault(key, []).append(unit)


# ---------------------------------------------------------
# NORMALIZE IMAGE MANIFEST
# ---------------------------------------------------------

images = image_manifest

if isinstance(image_manifest, dict):

    # Some manifests may wrap images inside a key.
    if "images" in image_manifest:
        images = image_manifest["images"]

    elif "image_manifest" in image_manifest:
        images = image_manifest["image_manifest"]

    else:
        images = []


print(f"Images in manifest: {len(images)}")


# ---------------------------------------------------------
# LINK IMAGES
# ---------------------------------------------------------

linked_images = 0
candidate_links = 0

for image in images:

    document_id = image.get("document_id")
    page = image.get("page")

    if document_id is None or page is None:
        continue

    key = (document_id, page)

    page_units = units_by_page.get(key, [])

    if not page_units:
        continue

    image_bbox = image.get("bbox")

    if not image_bbox:
        continue

    candidates = []

    for unit in page_units:

        unit_bbox = unit.get("bbox")

        if not unit_bbox:
            continue

        confidence = calculate_confidence(
            unit_bbox,
            image_bbox
        )

        if confidence >= MIN_CONFIDENCE:

            candidates.append(
                (
                    confidence,
                    unit
                )
            )

    if not candidates:
        continue

    # Sort highest confidence first
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Link image to the closest/highest-confidence unit.
    confidence, best_unit = candidates[0]

    if "images" not in best_unit:
        best_unit["images"] = []

    image_id = image.get("id")

    # Prevent duplicate links
    already_linked = any(
        x.get("image_id") == image_id
        for x in best_unit["images"]
    )

    if already_linked:
        continue

    best_unit["images"].append(
        {
            "image_id": image_id,
            "relationship": "visual_reference",
            "confidence": confidence,
            "method": "spatial_proximity",
            "page": page
        }
    )

    linked_images += 1

    candidate_links += len(candidates)


# ---------------------------------------------------------
# CLEAN INTERNAL FIELDS
# ---------------------------------------------------------

for document in knowledge_data:

    for unit in document.get("knowledge_units", []):

        unit.pop("_document_id", None)


# ---------------------------------------------------------
# SAVE BACK INTO SAME FILE
# ---------------------------------------------------------

with open(
    KNOWLEDGE_UNITS,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        knowledge_data,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

units_with_images = 0
total_links = 0

for document in knowledge_data:

    for unit in document.get("knowledge_units", []):

        count = len(unit.get("images", []))

        if count > 0:
            units_with_images += 1
            total_links += count


print()
print("IMAGE LINKING COMPLETE")
print("-" * 60)
print(f"Knowledge units with images : {units_with_images}")
print(f"Image links created         : {total_links}")
print(f"Candidate relationships     : {candidate_links}")
print()
print(f"Updated file:")
print(KNOWLEDGE_UNITS)