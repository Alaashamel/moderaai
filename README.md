# moderaai

![CI](https://github.com/Alaashamel/moderaai/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

A lightweight content moderation API. Classifies text as **safe**,
**toxic**, or **spam**, with a confidence score for each label.

```bash
$ moderaai check "Congratulations! You've won a free prize, click here now!!!"
label: spam
  spam: 0.559
  safe: 0.222
  toxic: 0.219
```

## How it works

The reference classifier is TF-IDF + logistic regression (scikit-learn),
trained on a small bundled labeled dataset
(`src/moderaai/data/labeled_examples.csv`). This is a deliberate choice
for a lightweight, fully reproducible, GPU-free reference implementation
— it trains in well under a second and has zero large downloads.

**This is a reference implementation, not a production-accuracy one.**
The bundled dataset is small (60 examples), so treat scores as
indicative, not authoritative. For production use, swap in a real
transformer model — `TransformerClassifier` in `classifier.py` is a
drop-in replacement built around Hugging Face's `unitary/toxic-bert`:

```bash
pip install "moderaai[transformer]"
```

```python
from moderaai.classifier import TransformerClassifier
clf = TransformerClassifier()  # instead of LexicalClassifier
```

Both classifiers implement the same `Classifier` protocol, so the API
and CLI work unchanged either way.

## Installation

```bash
git clone https://github.com/Alaashamel/moderaai.git
cd moderaai
pip install -e .
```

## Usage

### CLI

```bash
moderaai check "some text to classify"
moderaai train          # retrain the model from the bundled dataset
```

### HTTP API

```bash
moderaai serve
```

```bash
curl -X POST http://127.0.0.1:8000/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text": "You are so stupid and worthless"}'
```

```json
{"label": "toxic", "is_safe": false, "scores": {"safe": 0.21, "toxic": 0.60, "spam": 0.19}}
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## A note on the training data

The bundled dataset intentionally avoids slurs and hate speech targeting
protected groups — the "toxic" examples are generic insults/harassment,
which keeps the repo itself safe to browse while still teaching the
model a useful signal. See `CONTRIBUTING.md` before adding new examples.

## License

MIT — see [LICENSE](./LICENSE).
