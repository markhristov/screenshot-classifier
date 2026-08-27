
# Screenshot classifier

Download a compact local copy of the RICO Widget Captioning data:

```powershell
uv run python -m screenshot_classifier.download_dataset
```

The saved dataset contains only `id`, `captions`, `category`, and `package_name`.
Load it in Python with:

```python
from screenshot_classifier.data import load_compact_dataset

dataset = load_compact_dataset()
train = dataset["train"]
```
