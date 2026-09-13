# Dataset contract

## Task

Helmet status detection from head regions.

| Class ID | Exact class name |
| --- | --- |
| 0 | With Helmet |
| 1 | Without Helmet |

IDs and names are fixed. Do not rename classes to helmet/no_helmet, swap IDs,
or automatically relabel annotations.

## Annotation format

YOLO Object Detection, one object per line:

```text
class_id x_center y_center width height
```

Coordinates are normalized by image width/height. Centers are in [0, 1],
width and height in (0, 1], and box edges should remain inside the image.

## Current bounding-box convention

Bounding boxes represent head/face regions; classes represent helmet status.
`With Helmet` must not be interpreted simply as a box around only the helmet
object: boxes can include both helmet and face. `Without Helmet` represents
a head region without a helmet. These observations come from inspected samples;
full annotation consistency still needs manual review in the audit phase.
The annotations are not full-person boxes and do not encode person-bike relations.

The dataset contains both motorcycle and bicycle scenes. Do not assume it
represents only motorcycle riders.

## Manual review needed in the next phase

- Very small bounding boxes.
- Occluded heads.
- Blurry images.
- Partial heads.
- Hats/caps that are not helmets.
- Missing objects.
- Potentially incorrect classes.
- Inconsistent boxes around helmet-only versus the whole head region.

## Preservation and provenance

Keep train/, valid/, test/, data.yaml and both Roboflow README files unchanged
during foundation. No cleaning, removal, balancing, augmentation or resplitting
is performed here. Counts: train 1185, valid 127, test 64 (1376 images total).
The prior inspection counted 3618 boxes; foundation verification checks image
counts and config only, not all annotations. Export history and possible image
variants remain audit questions. Roboflow README metrics are not project metrics.
