# Research Scope

## Exact task definition

Single-class object detection and localization of pediatric wrist fractures in radiographs.

## Input

- One pediatric wrist radiograph.

## Output

- Fracture bounding-box coordinates.
- Fracture class label.
- Prediction confidence score.

## Metrics

- Precision
- Recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95
- Inference time
- Parameter count
- Model weight size
- FLOPs or a supported complexity proxy
- Training duration
- Quantitative and qualitative error analysis

## Included work

- Object detection only
- Single target class: `fracture`
- Negative radiographs retained where appropriate
- Multiple fracture boxes per image supported

## Excluded work

- Fracture subtype classification
- Bone-type classification
- Treatment recommendation
- Medical report generation
- Clinical approval claims
- Image enhancement baselines
- LLM-based diagnosis

## Baseline class policy

Class 0 is `fracture`.
The project remains single-class unless dataset inspection requires a documented change.

## Clinical disclaimer

This project is a research prototype only and does not replace physician or radiologist judgment.

