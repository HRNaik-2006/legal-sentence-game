"""Command line interface for the legal sentence drafting game."""

from __future__ import annotations

import argparse
import random
import sys
from typing import Callable, TextIO

from .core import LEGAL_CLAUSES, STARTER_PROMPTS, assemble_sentence, predict_next_words, tokenize
from .llm import classify_clause
from .storage import record_round


def play(*, input_fn: Callable[[str], str] = input, output: TextIO = sys.stdout,
         storage_path: str | None = None, rounds: int | None = None) -> None:
    """Run interactive drafting rounds; injectable input/output support automation."""
    completed = 0
    categories = list(LEGAL_CLAUSES)
    while rounds is None or completed < rounds:
        target = random.choice(categories)
        starter = random.choice(STARTER_PROMPTS[target]["beginner"])
        tokens = tokenize(starter)
        print(f"\nTarget clause: {target}\nStarter: {assemble_sentence(tokens)}", file=output)
        while True:
            predictions = predict_next_words(tokens, 3)
            print("Draft: " + assemble_sentence(tokens), file=output)
            for index, (word, probability) in enumerate(predictions, 1):
                print(f"[{index}] {word} ({probability:.1%})", file=output)
            try:
                choice = input_fn("Choose 1/2/3, type a word or phrase, /undo, or . to finish: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGame ended.", file=output)
                return
            if choice in {".", "/done"}:
                break
            if choice == "/undo":
                if tokens:
                    tokens.pop()
                continue
            if choice in {"1", "2", "3"}:
                if predictions:
                    tokens.append(predictions[int(choice) - 1][0])
                continue
            tokens.extend(tokenize(choice))
        sentence = assemble_sentence(tokens)
        if sentence and not sentence.endswith((".", "!", "?")):
            sentence += "."
        result = classify_clause(sentence)
        matched = result["clause_type"] == target
        print(f"\nSentence: {sentence}\nPredicted clause: {result['clause_type']} "
              f"({result['confidence']:.1%})\nTarget match: {'Yes' if matched else 'Not yet'}\n"
              f"Feedback: {result['feedback']}", file=output)
        record_round(target, result["clause_type"], sentence, result["confidence"],
                     result["feedback"], storage_path)
        completed += 1
        if rounds is not None and completed >= rounds:
            break
        try:
            again = input_fn("Play another round? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if again in {"n", "no", "quit", "q"}:
            break


def main(argv: list[str] | None = None) -> int:
    """Run the CLI command parser and dispatch a subcommand."""
    parser = argparse.ArgumentParser(prog="legal-sentence-game", description="Practice drafting legal clauses.")
    subparsers = parser.add_subparsers(dest="command")
    hello_parser = subparsers.add_parser("hello", help="Print the starter greeting.")
    hello_parser.set_defaults(handler=lambda _args: print("Hello, legal sentence game!"))
    play_parser = subparsers.add_parser("play", help="Start an interactive clause drafting game.")
    play_parser.add_argument("--rounds", type=int, default=None, help="Number of rounds before exiting (default: keep playing).")
    play_parser.add_argument("--data", default=None, help="History JSON path (default: .data/history.json).")
    play_parser.set_defaults(handler=lambda args: play(rounds=args.rounds, storage_path=args.data))
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
