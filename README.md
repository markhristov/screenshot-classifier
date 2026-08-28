# Screenshot Classifier

Screenshot Classifier is a learning project that predicts an Android
application category from text associated with a screenshot. The current model
does not process screenshot pixels. It classifies text extracted from a
screenshot, or widget captions supplied directly by the caller.

The classifier combines:

- TF-IDF features using lowercase unigrams and bigrams.
- Multiclass logistic regression.
- ONNX export for portable inference.
- A FastAPI service for cloud inference.

The exported ONNX model is also used by the companion Android application for
on-device inference.

## Repository structure

```text
src/
├── api/                         FastAPI inference service
│   ├── category_classifier.onnx
│   ├── labels.json
│   └── main.py
├── notebooks/
│   ├── models/                  Training exports
│   └── train.ipynb              Dataset preparation and model training
└── screenshot_classifier/       Python package
```

## Training process

The notebook loads `seeeeiii/RICO-WidgetCaptioning` from Hugging Face and keeps
the following useful metadata:

- Screen ID.
- Widget captions.
- Application category.
- Android package name.

The captions belonging to a row are joined into one text sample. The
application category becomes the classification label. The model is evaluated
on the dataset's validation split and then exported as an ONNX graph with a
separate ordered `labels.json` file.

## Install the Python environment

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```powershell
uv sync
```

## Run the cloud API locally

From the repository root:

```powershell
uv run uvicorn --app-dir src/api main:app --reload
```

Open `http://127.0.0.1:8000/docs` for FastAPI's interactive documentation.

The prediction endpoint accepts:

```json
{
  "text": "book a hotel and find nearby restaurants"
}
```

It returns the predicted category, the winning confidence, and the confidence
assigned to every category.

## Model artifacts

- `category_classifier.joblib` preserves the original scikit-learn pipeline for
  Python experimentation. Only load Joblib files obtained from a trusted
  source.
- `category_classifier.onnx` contains the portable inference graph.
- `labels.json` maps model output positions to category names.

The FastAPI directory contains copies of the ONNX model and labels so that it
can be deployed independently from the training notebook.

## Limitations

- This is an early text-classification prototype, not a production model.
- It ignores screenshot pixels, layout, icons, and visual context.
- Dataset category imbalance can reduce performance for uncommon categories.
- Confidence values should not be treated as guarantees.
- Text extracted by OCR can contain recognition errors that affect predictions.

## Dataset attribution

This project was trained using the
[RICO Widget Captioning dataset](https://huggingface.co/datasets/seeeeiii/RICO-WidgetCaptioning),
which combines the RICO mobile UI dataset with human-authored widget captions.
The dataset is distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
This project transforms the caption and category data by preparing text
features and training a category classifier; it does not redistribute the raw
screenshot dataset.

Please cite the original research when using this work:

- Biplab Deka, Zifeng Huang, Chad Franzen, Jeffrey Hibschman, Daniel Afergan,
  Yang Li, Jeffrey Nichols, and Ranjitha Kumar. *Rico: A Mobile App Dataset for
  Building Data-Driven Design Applications*. UIST, 2017.
- Yang Li, Gang Li, Luheng He, Jingjie Zheng, Hong Li, and Zhiwei Guan.
  *Widget Captioning: Generating Natural Language Description for Mobile User
  Interface Elements*. EMNLP, 2020.

See the dataset card for the complete citations and source information.

## Project license

No license has yet been selected for this repository's original source code.
The CC BY 4.0 dataset license applies to the source dataset, not automatically
to the original code in this repository. Add a project license before inviting
others to reuse or redistribute the code.
