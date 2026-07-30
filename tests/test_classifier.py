from moderaai.classifier import LexicalClassifier, load_training_data, train_model


def test_training_data_has_all_three_labels():
    _texts, labels = load_training_data()
    assert set(labels) == {"safe", "toxic", "spam"}
    assert len(labels) >= 30


def test_classifier_flags_toxic_text():
    clf = LexicalClassifier(model=train_model())
    result = clf.predict("You are so stupid and worthless, get lost.")
    assert result.label == "toxic"
    assert result.is_safe is False


def test_classifier_flags_spam_text():
    clf = LexicalClassifier(model=train_model())
    result = clf.predict("CONGRATULATIONS! You've WON a free prize, click here now!!!")
    assert result.label == "spam"


def test_classifier_flags_safe_text():
    clf = LexicalClassifier(model=train_model())
    result = clf.predict("Thanks for helping me with the report yesterday, really appreciated it.")
    assert result.label == "safe"
    assert result.is_safe is True


def test_scores_sum_to_roughly_one():
    clf = LexicalClassifier(model=train_model())
    result = clf.predict("Let's meet for coffee tomorrow morning.")
    assert abs(sum(result.scores.values()) - 1.0) < 1e-6


def test_save_and_load_roundtrip(tmp_path):
    model_path = tmp_path / "model.pkl"
    clf = LexicalClassifier(model=train_model())
    clf.save(model_path)

    loaded = LexicalClassifier.load_or_train(model_path)
    result = loaded.predict("You are pathetic and everyone hates you.")
    assert result.label == "toxic"
