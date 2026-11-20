"""Legal clause corpus, preprocessing, and interpolated n-gram prediction."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Iterable

START_TOKEN = "<s>"
END_TOKEN = "</s>"
TOKEN_PATTERN = re.compile(r"<[^>]+>|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)

# A compact, entirely local training set: eight clause categories, eight examples each.
LEGAL_CLAUSES: dict[str, tuple[str, ...]] = {
    "Confidentiality": (
        "Each party shall protect the other party's confidential information using reasonable care.",
        "The receiving party may use confidential information solely to perform this agreement.",
        "Confidential information excludes material that becomes public through no breach of this agreement.",
        "A party shall promptly notify the disclosing party of any unauthorized disclosure.",
        "The receiving party shall return or destroy confidential information upon written request.",
        "Disclosure is permitted to employees who need the information and are bound by confidentiality duties.",
        "These confidentiality obligations survive termination of this agreement for five years.",
        "Each party shall safeguard personal data in accordance with applicable privacy laws.",
    ),
    "Termination": (
        "Either party may terminate this agreement upon thirty days written notice.",
        "A party may terminate immediately if the other party materially breaches and fails to cure.",
        "Upon termination each party shall return the other party's property.",
        "Termination does not affect rights or obligations accrued before the effective date.",
        "The customer may terminate the services for convenience upon written notice.",
        "Either party may terminate this agreement if the other party becomes insolvent.",
        "The supplier shall provide reasonable transition assistance after termination.",
        "Sections that by their nature survive shall remain effective after termination.",
    ),
    "Indemnification": (
        "The supplier shall indemnify and defend the customer against third party claims.",
        "The indemnified party shall promptly notify the indemnifying party of any claim.",
        "The indemnifying party shall control the defense using counsel reasonably acceptable to the other party.",
        "No settlement may impose liability or admission on the indemnified party without consent.",
        "Each party shall indemnify the other for losses arising from its gross negligence.",
        "The supplier shall cover damages resulting from infringement of intellectual property rights.",
        "The parties shall cooperate in good faith in the defense of an indemnified claim.",
        "This indemnity is subject to the limitations and procedures set out in this agreement.",
    ),
    "Governing Law": (
        "This agreement is governed by the laws of the State of New York.",
        "The parties submit to the exclusive jurisdiction of the courts located in California.",
        "Each party waives any objection to venue in the courts specified herein.",
        "The United Nations Convention on Contracts for the International Sale of Goods does not apply.",
        "Any dispute arising under this agreement shall be resolved in the state courts of Delaware.",
        "This agreement shall be construed in accordance with the laws of England and Wales.",
        "The parties consent to personal jurisdiction in the courts of the chosen forum.",
        "A judgment may be enforced in any court of competent jurisdiction.",
    ),
    "Limitation of Liability": (
        "Neither party shall be liable for any indirect or consequential damages.",
        "The total liability of either party shall not exceed fees paid during the preceding twelve months.",
        "The foregoing limitation does not apply to liability for fraud or willful misconduct.",
        "Each party's aggregate liability under this agreement is capped at the applicable fees.",
        "No party shall be responsible for lost profits or loss of anticipated savings.",
        "These limits apply regardless of the theory of liability or whether a party was advised of the loss.",
        "Nothing in this section limits liability that cannot be excluded by law.",
        "The parties acknowledge that these limitations are reflected in the fees charged.",
    ),
    "Force Majeure": (
        "Neither party is liable for delay caused by events beyond its reasonable control.",
        "The affected party shall promptly notify the other party of the force majeure event.",
        "Performance obligations are suspended for the duration of the force majeure event.",
        "If the event continues for more than sixty days either party may terminate this agreement.",
        "Force majeure events include natural disasters, war, labor disputes, and government action.",
        "The affected party shall use reasonable efforts to mitigate the effects of the event.",
        "A failure to pay undisputed amounts is not excused by this section.",
        "The parties shall resume performance as soon as reasonably practicable.",
    ),
    "Payment Terms": (
        "The customer shall pay all undisputed invoices within thirty days of receipt.",
        "Late amounts accrue interest at the lesser of one percent per month or the maximum lawful rate.",
        "The supplier shall submit invoices monthly in arrears.",
        "Fees are exclusive of taxes, which shall be paid by the customer where applicable.",
        "The customer shall notify the supplier of any invoice dispute within ten days.",
        "The parties shall work in good faith to resolve disputed invoice amounts.",
        "Payment shall be made in the currency stated in the applicable order form.",
        "The supplier may suspend services for overdue undisputed amounts after written notice.",
    ),
    "Intellectual Property": (
        "Each party retains all rights in its pre existing intellectual property.",
        "The supplier grants the customer a nonexclusive license to use the deliverables.",
        "The customer owns all work product created specifically for the customer upon payment.",
        "No rights are transferred except as expressly stated in this agreement.",
        "The supplier may use feedback without restriction or obligation.",
        "The customer shall not remove proprietary notices from the licensed materials.",
        "The supplier retains ownership of its tools, methods, and generalized know how.",
        "All intellectual property rights in the services remain with their respective owners.",
    ),
}

STARTER_PROMPTS: dict[str, dict[str, tuple[str, ...]]] = {
    category: {
        "beginner": (sentences[0].rsplit(" ", 5)[0], sentences[1].rsplit(" ", 5)[0]),
        "intermediate": (sentences[2].rsplit(" ", 7)[0], sentences[4].rsplit(" ", 7)[0]),
        "advanced": (sentences[5].rsplit(" ", 9)[0], sentences[6].rsplit(" ", 9)[0]),
    }
    for category, sentences in LEGAL_CLAUSES.items()
}


def normalize_text(text: str) -> str:
    """Normalize whitespace and typographic apostrophes while retaining punctuation."""
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize words, contract tokens, and punctuation without lowercasing tokens."""
    return TOKEN_PATTERN.findall(normalize_text(text))


def load_corpus(path: str | Path | None = None) -> list[str]:
    """Load plain text from a file, or flatten the built-in legal clause corpus."""
    if path is None:
        return [sentence for clauses in LEGAL_CLAUSES.values() for sentence in clauses]
    content = Path(path).read_text(encoding="utf-8")
    return [normalize_text(s) for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]


def preprocess_corpus(texts: Iterable[str]) -> list[list[str]]:
    """Normalize and sentence-tokenize texts, wrapping each sentence with boundary tokens."""
    sentences: list[list[str]] = []
    for text in texts:
        normalized = normalize_text(text)
        for sentence in re.split(r"(?<=[.!?])\s+", normalized):
            words = tokenize(sentence)
            if words:
                sentences.append([START_TOKEN, *words, END_TOKEN])
    return sentences


class TrigramLanguageModel:
    """Interpolated trigram model with add-one smoothing and lower-order backoff."""

    def __init__(self, corpus: Iterable[str] | None = None, *, alpha: float = 0.1) -> None:
        self.alpha = max(float(alpha), 1e-9)
        sentences = [
            [token.lower() if token not in {START_TOKEN, END_TOKEN} else token for token in sentence]
            for sentence in preprocess_corpus(corpus if corpus is not None else load_corpus())
        ]
        self.unigrams: Counter[str] = Counter()
        self.bigrams: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.trigrams: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for sentence in sentences:
            for word in sentence:
                self.unigrams[word] += 1
            for i in range(1, len(sentence)):
                self.bigrams[(sentence[i - 1],)][sentence[i]] += 1
            for i in range(2, len(sentence)):
                self.trigrams[(sentence[i - 2], sentence[i - 1])][sentence[i]] += 1
        self.vocabulary = set(self.unigrams) | {END_TOKEN}

    def predict_next_words(self, context_tokens: list[str], top_k: int = 3) -> list[tuple[str, float]]:
        """Return candidate and normalized probability pairs for a supplied token context."""
        if top_k <= 0:
            return []
        context = [t.lower() if t not in {START_TOKEN, END_TOKEN} else t for t in context_tokens]
        vocab = [word for word in self.vocabulary if word != START_TOKEN]
        if not vocab:
            return []
        tri_context = tuple(context[-2:]) if len(context) >= 2 else ()
        bi_context = (context[-1],) if context else ()
        tri = self.trigrams.get(tri_context, Counter()) if len(tri_context) == 2 else Counter()
        bi = self.bigrams.get(bi_context, Counter()) if bi_context else Counter()
        total_tri, total_bi = sum(tri.values()), sum(bi.values())
        total_uni = sum(self.unigrams.values())
        def prob(word: str) -> float:
            p3 = (tri[word] + self.alpha) / (total_tri + self.alpha * len(vocab))
            p2 = (bi[word] + self.alpha) / (total_bi + self.alpha * len(vocab))
            p1 = (self.unigrams[word] + self.alpha) / (total_uni + self.alpha * len(vocab))
            if tri:
                return .7 * p3 + .2 * p2 + .1 * p1
            if bi:
                return .6 * p2 + .4 * p1
            return p1
        ranked = sorted(((w, prob(w)) for w in vocab), key=lambda item: (-item[1], item[0]))[:top_k]
        normalizer = sum(p for _, p in ranked)
        return [(word, probability / normalizer) for word, probability in ranked]


_DEFAULT_MODEL = TrigramLanguageModel()


def predict_next_words(context_tokens: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """Predict likely next tokens using the default built-in legal language model."""
    return _DEFAULT_MODEL.predict_next_words(context_tokens, top_k)


def assemble_sentence(tokens: Iterable[str]) -> str:
    """Join drafted tokens with conventional spacing around punctuation."""
    text = " ".join(token for token in tokens if token not in {START_TOKEN, END_TOKEN})
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def validate_sentence(sentence: str) -> tuple[bool, str]:
    """Return a simple drafting completeness check and an actionable explanation."""
    clean = normalize_text(sentence)
    if len(tokenize(clean)) < 5:
        return False, "Add more detail; a complete clause should contain at least five tokens."
    if not clean.endswith((".", "!", "?")):
        return False, "Finish the sentence with terminal punctuation."
    return True, "Sentence is complete."
