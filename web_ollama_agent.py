"""
web_ollama_agent.py

Ein Kommandozeilen-Tool, das Webseiten lädt, den Text extrahiert und zusammen
mit einer Frage an ein lokales Ollama-Modell sendet. Ergebnisse werden als
JSON gespeichert.

Installation:
    pip install requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


DEFAULT_MODEL = "deepseek:7b"
DEFAULT_TIMEOUT = 15.0
DEFAULT_OUTPUT_FILE = "results.json"
MAX_TEXT_LENGTH = 12000


@dataclass
class Task:
    """Represents a single webpage processing task."""

    id: str
    url: str
    question: str


@dataclass
class TaskResult:
    """Holds the processing outcome for a task."""

    id: str
    url: str
    success: bool
    error: Optional[str]
    answer: Optional[str]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "success": self.success,
            "error": self.error,
            "answer": self.answer,
            "meta": self.meta,
        }


def load_tasks_from_file(path: Path) -> List[Task]:
    """Load and validate tasks from a JSON file."""

    logging.info("Loading tasks from %s", path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Task file must contain a JSON array")

    tasks: List[Task] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each task must be a JSON object")
        missing = [key for key in ("id", "url", "question") if key not in entry]
        if missing:
            raise ValueError(f"Task is missing required fields: {', '.join(missing)}")
        tasks.append(Task(id=str(entry["id"]), url=str(entry["url"]), question=str(entry["question"])))
    return tasks


def fetch_page(url: str, timeout: float) -> str:
    """Download a webpage and return its HTML content."""

    logging.info("Fetching URL: %s", url)
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise requests.HTTPError(f"Unexpected status code {response.status_code} for {url}")
    return response.text


def _clean_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Normalize whitespace and truncate overly long text."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_length:
        return normalized
    shortened = normalized[:max_length]
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened


def extract_text_from_html(html: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Extract readable text from HTML content using BeautifulSoup."""

    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ("script", "style", "noscript", "header", "footer", "nav"):
        for element in soup.find_all(tag_name):
            element.decompose()

    text_fragments: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"]):
        content = tag.get_text(separator=" ", strip=True)
        if content:
            text_fragments.append(content)

    # Fallback to body text if targeted tags were sparse.
    if not text_fragments:
        body_text = soup.get_text(separator=" ", strip=True)
        text_fragments.append(body_text)

    combined_text = " ".join(text_fragments)
    return _clean_text(combined_text, max_length=max_length)


def build_prompt(page_text: str, question: str, url: str) -> str:
    """Create a German-language prompt for the Ollama model."""

    return (
        "Du bist ein hilfreicher Assistent. Du bekommst den Textinhalt einer Webseite "
        "und eine Aufgabe dazu.\n\n"
        f"URL der Seite:\n{url}\n\n"
        "Auf dieser Seite gefundener Text (möglicherweise gekürzt):\n"
        "\"\"\"\n"
        f"{page_text}\n"
        "\"\"\"\n\n"
        "Deine Aufgabe:\n"
        f"{question}\n\n"
        "Anforderungen an die Antwort:\n"
        "- Antworte nur auf Basis der Informationen aus dem obigen Text.\n"
        "- Wenn eine Information im Text nicht vorkommt, sage klar, dass sie nicht im Text steht.\n"
        "- Strukturiere die Antwort übersichtlich (Listen, Absätze, ggf. Überschriften)."
    )


def call_ollama(prompt: str, model: str, timeout: float) -> str:
    """Send a prompt to the local Ollama API and return the model's reply."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    logging.info("Sending prompt to Ollama model '%s'", model)
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=timeout)
    if response.status_code != 200:
        raise requests.HTTPError(f"Ollama API returned status {response.status_code}")

    data = response.json()
    message = data.get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Unexpected Ollama response format: missing message content")
    return content


def process_task(task: Task, model: str, timeout: float) -> TaskResult:
    """Process a single task from download to model answer."""

    download_start = time.perf_counter()
    try:
        html = fetch_page(task.url, timeout=timeout)
        download_time_ms = (time.perf_counter() - download_start) * 1000

        page_text = extract_text_from_html(html)
        prompt = build_prompt(page_text, task.question, task.url)
        answer = call_ollama(prompt, model=model, timeout=timeout)
        meta = {
            "download_time_ms": round(download_time_ms, 2),
            "prompt_length": len(prompt),
            "answer_length": len(answer),
        }
        return TaskResult(id=task.id, url=task.url, success=True, error=None, answer=answer, meta=meta)
    except Exception as exc:  # noqa: BLE001 - broad to capture all task errors
        logging.error("Task %s failed: %s", task.id, exc)
        download_time_ms = (time.perf_counter() - download_start) * 1000
        return TaskResult(
            id=task.id,
            url=task.url,
            success=False,
            error=str(exc),
            answer=None,
            meta={"download_time_ms": round(download_time_ms, 2)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch webpages and query a local Ollama model.")
    parser.add_argument("tasks_file", type=Path, help="Path to JSON file containing tasks")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUTPUT_FILE), help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Using model: %s", args.model)
    logging.info("Input file: %s", args.tasks_file)
    logging.info("Output file: %s", args.out)

    try:
        tasks = load_tasks_from_file(args.tasks_file)
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to load tasks: %s", exc)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for task in tasks:
        result = process_task(task, model=args.model, timeout=args.timeout)
        results.append(result.to_dict())
        status = "success" if result.success else f"error: {result.error}"
        preview = f" - Answer preview: {result.answer[:80]}..." if result.answer else ""
        print(f"[{task.id}] {task.url} -> {status}{preview}")

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote %d results to %s", len(results), args.out)


if __name__ == "__main__":
    main()
