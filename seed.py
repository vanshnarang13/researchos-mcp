import asyncio
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import torch
from sentence_transformers import SentenceTransformer
import chromadb

from utils import call_llm_json

TRANSCRIPTS_DIR = Path("data/transcripts")

PERSONAS = [
    "PM at a 200-person fintech company",
    "Solo founder of an early-stage SaaS",
    "Senior engineer at an enterprise company",
    "Head of Operations at a mid-size agency",
    "Customer success manager at a B2B software company",
]

TOPIC = "B2B SaaS onboarding pain points"


async def generate_transcript(interview_id: str, persona: str) -> dict:
    prompt = (
        f"Generate a realistic user research interview transcript as a JSON object.\n"
        f"Topic: {TOPIC}\n"
        f"Participant persona: {persona}\n\n"
        "Requirements:\n"
        "- 8-12 turns total, alternating moderator then participant\n"
        "- Moderator asks open-ended questions (not yes/no)\n"
        "- Mix answers: some vague (short, non-specific, like 'it was fine' or 'pretty okay'), "
        "some detailed (specific pain points, named features, concrete examples)\n"
        "- Include 2-3 natural moderator follow-up probes after vague answers\n"
        "- Probes should be natural, e.g. 'Can you give me a specific example?' "
        "or 'What happened next?'\n\n"
        "Output JSON with exactly these keys:\n"
        f"id (string, use '{interview_id}'), \n"
        "topic (string), \n"
        "participant_persona (string), \n"
        "turns (array of {role, text} objects),\n"
        "summary (null),\n"
        "probes_triggered (integer, count of probe turns in the transcript),\n"
        "created_at (ISO 8601 string, use a date in 2025)"
    )
    result_str = await call_llm_json(
        prompt,
        system="You are generating realistic synthetic UX research interview transcripts. "
               "Output only a valid JSON object. No markdown, no backticks, no explanation."
    )
    return json.loads(result_str)


def embed_all_transcripts(transcripts: list[dict]) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading sentence-transformers on device: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete and recreate collection for a clean rebuild
    try:
        client.delete_collection("research_transcripts")
        print("Deleted existing collection.")
    except Exception:
        pass
    collection = client.create_collection("research_transcripts")

    for transcript in transcripts:
        interview_id = transcript["id"]
        turns = transcript["turns"]
        chunk_count = 0
        for i, turn in enumerate(turns):
            if turn["role"] != "participant":
                continue
            question_context = ""
            if i > 0 and turns[i - 1]["role"] == "moderator":
                question_context = turns[i - 1]["text"]
            chunk_id = f"{interview_id}_turn_{i}"
            embedding = model.encode(turn["text"]).tolist()
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[turn["text"]],
                metadatas=[{
                    "interview_id": interview_id,
                    "participant_persona": transcript["participant_persona"],
                    "topic": transcript["topic"],
                    "turn_index": i,
                    "question_context": question_context
                }]
            )
            chunk_count += 1
        print(f"Embedded {interview_id} ({chunk_count} chunks)...")


async def main():
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = list(TRANSCRIPTS_DIR.glob("*.json"))

    if existing_files:
        print(f"Transcripts already exist ({len(existing_files)} files) — skipping generation. Re-embedding...")
        transcripts = []
        for f in sorted(existing_files):
            with open(f) as fp:
                transcripts.append(json.load(fp))
    else:
        print(f"Generating 25 synthetic transcripts for topic: '{TOPIC}'")
        transcripts = []
        for i in range(25):
            interview_id = f"int_{i + 1:03d}"
            persona = PERSONAS[i % len(PERSONAS)]
            print(f"Generating {interview_id} | Persona: {persona}...")
            try:
                transcript = await generate_transcript(interview_id, persona)
                # Normalize role names to lowercase
                for turn in transcript.get("turns", []):
                    turn["role"] = turn["role"].lower()
                # Ensure required fields are present
                transcript["id"] = interview_id
                transcript["topic"] = TOPIC
                transcript["participant_persona"] = persona
                if "summary" not in transcript:
                    transcript["summary"] = None
                if "probes_triggered" not in transcript:
                    transcript["probes_triggered"] = 0
                if "created_at" not in transcript:
                    transcript["created_at"] = datetime.now(timezone.utc).isoformat()
                save_path = TRANSCRIPTS_DIR / f"{interview_id}.json"
                with open(save_path, "w") as f:
                    json.dump(transcript, f, indent=2)
                transcripts.append(transcript)
                print(f"  Saved {interview_id} ({len(transcript['turns'])} turns)")
            except Exception as e:
                print(f"  ERROR generating {interview_id}: {e}")

        print(f"\nGenerated {len(transcripts)} transcripts.")

    print("\nEmbedding all transcripts into ChromaDB...")
    embed_all_transcripts(transcripts)
    print("\nDone! ChromaDB populated.")


if __name__ == "__main__":
    asyncio.run(main())
