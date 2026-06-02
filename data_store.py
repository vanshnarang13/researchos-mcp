import json
import os
import glob
from pathlib import Path

_BASE_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = _BASE_DIR / "data" / "transcripts"


def save_interview(interview: dict) -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTS_DIR / f"{interview['id']}.json"
    with open(path, "w") as f:
        json.dump(interview, f, indent=2)


def load_interview(interview_id: str) -> dict | None:
    path = TRANSCRIPTS_DIR / f"{interview_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_interviews(topic_filter: str | None, limit: int, offset: int) -> dict:
    files = sorted(glob.glob(str(TRANSCRIPTS_DIR / "*.json")))
    interviews = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        if topic_filter and topic_filter.lower() not in data.get("topic", "").lower():
            continue
        interviews.append({
            "id": data["id"],
            "topic": data["topic"],
            "participant_persona": data["participant_persona"],
            "created_at": data["created_at"],
            "probes_triggered": data.get("probes_triggered", 0),
            "has_summary": data.get("summary") is not None
        })
    total = len(interviews)
    page = interviews[offset: offset + limit]
    return {
        "total": total,
        "count": len(page),
        "offset": offset,
        "has_more": total > offset + len(page),
        "interviews": page
    }


def count_interviews() -> int:
    if not TRANSCRIPTS_DIR.exists():
        return 0
    return len(list(TRANSCRIPTS_DIR.glob("*.json")))


def next_interview_id() -> str:
    existing = sorted(TRANSCRIPTS_DIR.glob("int_*.json")) if TRANSCRIPTS_DIR.exists() else []
    if not existing:
        return "int_001"
    last = existing[-1].stem  # e.g. "int_025"
    num = int(last.split("_")[1]) + 1
    return f"int_{num:03d}"
