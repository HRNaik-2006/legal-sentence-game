"""Core, classifier, persistence, and CLI integration tests."""

import io

from app.core import END_TOKEN, START_TOKEN, TrigramLanguageModel, load_corpus, predict_next_words, preprocess_corpus
from app.llm import classify_clause
from app.cli import play
from app.storage import get_progress, load_history, record_round


def test_builtin_corpus_and_sentence_boundaries():
    corpus = load_corpus()
    assert len(corpus) >= 60
    wrapped = preprocess_corpus(["Payment is due. Notice is required!"])
    assert wrapped == [[START_TOKEN, "Payment", "is", "due", ".", END_TOKEN],
                       [START_TOKEN, "Notice", "is", "required", "!", END_TOKEN]]


def test_ngram_predictions_are_top_three_and_normalized():
    model = TrigramLanguageModel(["The party shall pay fees.", "The party shall provide notice."])
    for context in ([START_TOKEN, "the", "party"], ["unseen", "context"], ["shall"]):
        result = model.predict_next_words(list(context), 3)
        assert len(result) == 3
        assert all(word != START_TOKEN and probability > 0 for word, probability in result)
        assert abs(sum(p for _, p in result) - 1) < 1e-9
    assert predict_next_words(["the", "party"], 3)


def test_classifier_schema_and_category_recognition():
    result = classify_clause("The customer shall pay all invoices within thirty days of receipt.")
    assert set(result) == {"clause_type", "confidence", "feedback"}
    assert result["clause_type"] == "Payment Terms"
    assert 0 <= result["confidence"] <= 1
    assert result["feedback"]


def test_storage_round_trip_and_progress(tmp_path):
    path = tmp_path / "nested" / "history.json"
    record_round("Payment Terms", "Payment Terms", "Pay fees.", .8, path=path)
    assert len(load_history(path)) == 1
    assert get_progress(path) == {"rounds_played": 1, "matches": 1, "match_rate": 1.0}


def test_cli_game_records_round(tmp_path, monkeypatch):
    import app.cli as cli
    monkeypatch.setattr(cli.random, "choice", lambda values: "Payment Terms" if values and values[0] == "Confidentiality" else values[0])
    # The starter is completed immediately; one round avoids the replay prompt.
    output = io.StringIO()
    answers = iter(["."])
    play(input_fn=lambda _prompt: next(answers), output=output,
         storage_path=str(tmp_path / "game.json"), rounds=1)
    assert "Predicted clause:" in output.getvalue()
    assert len(load_history(tmp_path / "game.json")) == 1
