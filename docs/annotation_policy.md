# Annotation Policy

## Authority

The conversion pipeline treats the Pascal VOC XML annotations as the authoritative
source for label conversion. Supervisely JSON annotations are retained for audit
comparison only.

## Class mapping

Only the source label `fracture` is converted to YOLO class `0`.

The following labels are not converted into fracture boxes:

- `text`
- `axis`
- `periostealreaction`
- `pronatorsign`
- `softtissue`
- `metal`

The pipeline records these labels for inspection and reporting, but they are
excluded from the single-class detection target.

## Retained cases

- Images with one or more valid fracture boxes are converted as positive samples.
- Valid images without fracture boxes are retained as negative samples with empty
  label files.
- Multiple fracture boxes in one image are preserved.

## Invalid or ambiguous annotations

- Boxes with invalid coordinates are rejected during conversion.
- Duplicate boxes within the same image are deduplicated before YOLO export.
- XML and JSON annotations are compared for audit purposes, but the conversion
  path uses one authoritative representation rather than merging both sources.

## Research note

This policy is for a single-class research dataset only and does not imply a
clinical interpretation of any non-fracture labels.
