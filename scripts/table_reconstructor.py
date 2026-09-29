import json
import re
from pathlib import Path
from difflib import SequenceMatcher


INPUT_FILE = Path(
    "extracted_knowledge/tables/all_tables.json"
)

OUTPUT_FILE = Path(
    "extracted_knowledge/tables/reconstructed_tables.json"
)

REPORT_FILE = Path(
    "extracted_knowledge/tables/table_reconstruction_report.json"
)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    return value


def normalize_row(row):
    return [
        normalize_text(cell)
        for cell in row
    ]


# =========================================================
# SIMILARITY
# =========================================================

def row_similarity(row1, row2):

    a = " | ".join(normalize_row(row1))
    b = " | ".join(normalize_row(row2))

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def header_similarity(row1, row2):

    if len(row1) != len(row2):
        return 0.0

    if not row1 or not row2:
        return 0.0

    scores = []

    for a, b in zip(row1, row2):

        a = normalize_text(a)
        b = normalize_text(b)

        if not a and not b:
            scores.append(1.0)

        elif not a or not b:
            scores.append(0.0)

        else:
            scores.append(
                SequenceMatcher(
                    None,
                    a,
                    b
                ).ratio()
            )

    return sum(scores) / len(scores)


# =========================================================
# HEADER DETECTION
# =========================================================

HEADER_KEYWORDS = {
    "description",
    "control",
    "control elements",
    "item",
    "name",
    "type",
    "function",
    "operation",
    "details",
    "remarks",
    "specification",
    "parameter",
    "value",
    "location",
    "feature",
    "category",
    "part",
    "component",
    "model",
    "variant",
    "system",
    "condition",
    "cause",
    "action",
}


def looks_like_header(row):

    if not row:
        return False

    values = [
        normalize_text(x)
        for x in row
        if normalize_text(x)
    ]

    if not values:
        return False

    keyword_hits = 0

    for value in values:

        for keyword in HEADER_KEYWORDS:

            if keyword in value:
                keyword_hits += 1
                break

    text_cells = sum(
        bool(re.search(r"[a-zA-Z]", value))
        for value in values
    )

    numeric_cells = sum(
        bool(re.fullmatch(r"[\d\s.,:/%+-]+", value))
        for value in values
    )

    # Header should generally be textual
    if numeric_cells == len(values):
        return False

    return (
        keyword_hits >= 1
        and text_cells >= max(
            1,
            len(values) // 2
        )
    )


# =========================================================
# STRUCTURAL CHECKS
# =========================================================

def same_document(a, b):

    return (
        a["document_id"]
        == b["document_id"]
    )


def adjacent_pages(a, b):

    gap = (
        b["page_start"]
        - a["page_end"]
    )

    return 0 <= gap <= 1


def same_column_count(a, b):

    return (
        a["columns"]
        == b["columns"]
    )


# =========================================================
# STRONG TABLE CONTINUATION TEST
# =========================================================

def repeated_header_evidence(previous, current):

    if not previous.get("data"):
        return False, 0.0

    if not current.get("data"):
        return False, 0.0

    previous_header = previous["data"][0]
    current_header = current["data"][0]

    similarity = header_similarity(
        previous_header,
        current_header
    )

    return similarity >= 0.85, similarity


# =========================================================
# CONSERVATIVE MERGE DECISION
# =========================================================

def classify_pair(previous, current):

    # -----------------------------------------------------
    # Different document
    # -----------------------------------------------------

    if not same_document(previous, current):

        return {
            "decision": "NO_MERGE",
            "confidence": "high",
            "reason": "different_document",
            "score": 0.0,
        }

    # -----------------------------------------------------
    # Non-adjacent pages
    # -----------------------------------------------------

    if not adjacent_pages(previous, current):

        return {
            "decision": "NO_MERGE",
            "confidence": "high",
            "reason": "non_adjacent_pages",
            "score": 0.0,
        }

    # -----------------------------------------------------
    # Different column count
    # -----------------------------------------------------

    if not same_column_count(previous, current):

        return {
            "decision": "NO_MERGE",
            "confidence": "high",
            "reason": "different_column_count",
            "score": 0.0,
        }

    # -----------------------------------------------------
    # Empty tables
    # -----------------------------------------------------

    if (
        not previous.get("data")
        or not current.get("data")
    ):

        return {
            "decision": "NO_MERGE",
            "confidence": "high",
            "reason": "empty_table",
            "score": 0.0,
        }

    previous_header = previous["data"][0]
    current_header = current["data"][0]

    similarity = header_similarity(
        previous_header,
        current_header
    )

    # =====================================================
    # CASE 1
    # Strong repeated header
    # =====================================================

    if similarity >= 0.85:

        return {
            "decision": "CONFIRMED_MERGE",
            "confidence": "high",
            "reason": "strong_repeated_header",
            "score": round(similarity, 3),
        }

    # =====================================================
    # CASE 2
    # Similar but not identical header
    # =====================================================

    if similarity >= 0.65:

        return {
            "decision": "POSSIBLE_MERGE",
            "confidence": "medium",
            "reason": "similar_header",
            "score": round(similarity, 3),
        }

    # =====================================================
    # CASE 3
    # Same structure but no header evidence
    #
    # DO NOT automatically merge.
    # =====================================================

    if (
        looks_like_header(previous_header)
        and not looks_like_header(current_header)
    ):

        return {
            "decision": "POSSIBLE_MERGE",
            "confidence": "low",
            "reason": "possible_continuation_without_repeated_header",
            "score": round(similarity, 3),
        }

    # =====================================================
    # Everything else
    # =====================================================

    return {
        "decision": "NO_MERGE",
        "confidence": "medium",
        "reason": "insufficient_continuation_evidence",
        "score": round(similarity, 3),
    }


# =========================================================
# MERGE
# =========================================================

def merge_group(group):

    first = group[0]

    logical_table = {

        "logical_table_id": (
            f"{first['document_id']}_"
            f"p{first['page_start']}_"
            f"logical_table"
        ),

        "document_id":
            first["document_id"],

        "source_file":
            first["source_file"],

        "page_start":
            min(
                x["page_start"]
                for x in group
            ),

        "page_end":
            max(
                x["page_end"]
                for x in group
            ),

        "columns":
            first["columns"],

        "header":
            first["data"][0]
            if first.get("data")
            else [],

        "rows": [],

        "fragments": [],
    }

    for index, table in enumerate(group):

        logical_table["fragments"].append({

            "table_id":
                table["table_id"],

            "page_start":
                table["page_start"],

            "page_end":
                table["page_end"],

            "source_table_index":
                table["source_table_index"],

            "columns":
                table["columns"],

            "rows":
                table["rows"],
        })

        data = table.get("data", [])

        if not data:
            continue

        # First fragment
        if index == 0:

            logical_table["rows"].extend(
                data[1:]
            )

            continue

        # Later fragments
        similarity = header_similarity(
            logical_table["header"],
            data[0]
        )

        # Remove repeated header only
        if similarity >= 0.85:

            logical_table["rows"].extend(
                data[1:]
            )

        else:

            logical_table["rows"].extend(
                data
            )

    return logical_table


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nLoading raw extracted tables...")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        tables = json.load(f)

    print(
        f"Raw table fragments: {len(tables)}"
    )

    # Sort
    tables.sort(
        key=lambda x: (
            x["document_id"],
            x["page_start"],
            x["source_table_index"]
        )
    )

    logical_tables = []

    confirmed_merges = []
    possible_merges = []
    no_merges = []

    current_group = []

    for table in tables:

        if not current_group:

            current_group = [table]
            continue

        previous = current_group[-1]

        result = classify_pair(
            previous,
            table
        )

        event = {
            "previous_table":
                previous["table_id"],

            "current_table":
                table["table_id"],

            **result,
        }

        # -------------------------------------------------
        # Confirmed merge
        # -------------------------------------------------

        if result["decision"] == "CONFIRMED_MERGE":

            current_group.append(table)

            confirmed_merges.append(event)

        # -------------------------------------------------
        # Possible merge
        #
        # IMPORTANT:
        # Do NOT merge automatically.
        # -------------------------------------------------

        elif result["decision"] == "POSSIBLE_MERGE":

            possible_merges.append(event)

            logical_tables.append(
                merge_group(current_group)
            )

            current_group = [table]

        # -------------------------------------------------
        # No merge
        # -------------------------------------------------

        else:

            no_merges.append(event)

            logical_tables.append(
                merge_group(current_group)
            )

            current_group = [table]

    # Final group
    if current_group:

        logical_tables.append(
            merge_group(current_group)
        )

    # =====================================================
    # SAVE
    # =====================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            logical_tables,
            f,
            indent=2,
            ensure_ascii=False
        )

    # =====================================================
    # REPORT
    # =====================================================

    report = {

        "raw_table_fragments":
            len(tables),

        "logical_tables":
            len(logical_tables),

        "confirmed_merge_operations":
            len(confirmed_merges),

        "possible_merge_candidates":
            len(possible_merges),

        "no_merge_operations":
            len(no_merges),

        "confirmed_merges":
            confirmed_merges,

        "possible_merges":
            possible_merges,

        "no_merges":
            no_merges,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n======================================")
    print("STRICT TABLE RECONSTRUCTION COMPLETE")
    print("======================================")

    print(
        f"Raw fragments       : {len(tables)}"
    )

    print(
        f"Logical tables      : {len(logical_tables)}"
    )

    print(
        f"Confirmed merges    : {len(confirmed_merges)}"
    )

    print(
        f"Possible candidates : {len(possible_merges)}"
    )

    print(
        f"No merges           : {len(no_merges)}"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()