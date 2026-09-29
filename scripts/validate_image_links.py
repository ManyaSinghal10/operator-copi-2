import json
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_UNITS = (
    BASE_DIR / "extracted_knowledge" / "knowledge_units.json"
)

IMAGE_MANIFEST = (
    BASE_DIR
    / "extracted_knowledge"
    / "images"
    / "image_manifest.json"
)


print("IMAGE LINK VALIDATION")
print("-" * 60)


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

with open(KNOWLEDGE_UNITS, "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)

with open(IMAGE_MANIFEST, "r", encoding="utf-8") as f:
    manifest = json.load(f)


images = manifest.get("images", manifest)


# ---------------------------------------------------------
# COLLECT LINKS
# ---------------------------------------------------------

linked_image_ids = set()

confidence_values = []

units_with_images = 0
total_links = 0

relationship_counts = Counter()
method_counts = Counter()

examples = []


for document in knowledge_data:

    for unit in document.get("knowledge_units", []):

        image_links = unit.get("images", [])

        if not image_links:
            continue

        units_with_images += 1

        for link in image_links:

            total_links += 1

            image_id = link.get("image_id")

            if image_id:
                linked_image_ids.add(image_id)

            confidence = link.get("confidence")

            if confidence is not None:
                confidence_values.append(confidence)

            relationship_counts[
                link.get("relationship", "unknown")
            ] += 1

            method_counts[
                link.get("method", "unknown")
            ] += 1

            if len(examples) < 10:
                examples.append(
                    {
                        "unit_id": unit.get("id"),
                        "unit_type": unit.get("type"),
                        "unit_title": unit.get("title"),
                        "image_id": image_id,
                        "page": link.get("page"),
                        "confidence": confidence,
                    }
                )


# ---------------------------------------------------------
# IMAGE COUNTS
# ---------------------------------------------------------

manifest_image_ids = set()

for image in images:

    image_id = image.get("id")

    if image_id:
        manifest_image_ids.add(image_id)


unlinked_images = (
    manifest_image_ids - linked_image_ids
)


# ---------------------------------------------------------
# CONFIDENCE DISTRIBUTION
# ---------------------------------------------------------

confidence_buckets = Counter()

for confidence in confidence_values:

    if confidence >= 0.80:
        confidence_buckets["0.80 - 1.00"] += 1

    elif confidence >= 0.65:
        confidence_buckets["0.65 - 0.79"] += 1

    elif confidence >= 0.50:
        confidence_buckets["0.50 - 0.64"] += 1

    else:
        confidence_buckets["0.45 - 0.49"] += 1


# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

print()
print("COUNTS")
print("-" * 40)

print(f"Images in manifest       : {len(manifest_image_ids)}")
print(f"Images linked            : {len(linked_image_ids)}")
print(f"Images without link      : {len(unlinked_images)}")

print(f"Units with images        : {units_with_images}")
print(f"Total image links        : {total_links}")


print()
print("CONFIDENCE DISTRIBUTION")
print("-" * 40)

for bucket in [
    "0.80 - 1.00",
    "0.65 - 0.79",
    "0.50 - 0.64",
    "0.45 - 0.49",
]:
    print(
        f"{bucket:15} : "
        f"{confidence_buckets[bucket]}"
    )


print()
print("RELATIONSHIPS")
print("-" * 40)

for key, value in relationship_counts.items():
    print(f"{key:25} : {value}")


print()
print("METHODS")
print("-" * 40)

for key, value in method_counts.items():
    print(f"{key:25} : {value}")


print()
print("SAMPLE LINKS")
print("-" * 40)

for example in examples:

    print(
        f"{example['unit_id']} | "
        f"{example['unit_title']} | "
        f"{example['image_id']} | "
        f"page={example['page']} | "
        f"confidence={example['confidence']}"
    )


print()
print("VALIDATION COMPLETE")