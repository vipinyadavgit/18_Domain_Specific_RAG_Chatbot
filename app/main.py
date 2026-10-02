"""Command-line chatbot.

Run it with:   uv run python -m app.main
Type "exit" or "quit" to stop.
"""

import logging

from openai import OpenAIError

from app.config import setup_logging
from app.rag_chain import generate_answer
from app.retriever import store_exists

logger = logging.getLogger(__name__)

EXIT_WORDS = {"exit", "quit"}


def print_result(result: dict) -> None:
    """Show the answer, then the sources it was based on."""
    print("\nAnswer:")
    print(result["answer"])

    if result["sources"]:
        print("\nSources:")
        for number, chunk in enumerate(result["sources"], start=1):
            page = f", page {chunk['page']}" if chunk["page"] else ""
            print(f"  [{number}] {chunk['source']}{page}  (similarity {chunk['score']:.2f})")


def main() -> None:
    setup_logging()

    print("=" * 60)
    print(" Automotive RAG Chatbot  (type 'exit' or 'quit' to stop)")
    print("=" * 60)

    if not store_exists():
        print("\nNo vector store found. First run:  uv run python -m app.ingest")
        return

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+C / Ctrl+D
            print("\nGoodbye!")
            break

        if not query:  # empty input
            print("Please type a question.")
            continue

        if query.lower() in EXIT_WORDS:
            print("Goodbye!")
            break

        try:
            print_result(generate_answer(query))
        except ValueError as error:  # e.g. missing API key
            print(f"\nConfiguration problem: {error}")
        except OpenAIError as error:  # wrong key, no credit, network problem, ...
            logger.error("OpenAI request failed: %s", error)
            print("\nThe OpenAI request failed. Check your API key, credit and internet connection.")
        except Exception as error:
            logger.exception("Unexpected error")
            print(f"\nSomething went wrong: {error}")


if __name__ == "__main__":
    main()
