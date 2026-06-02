import json

from utils import call_llm_json
from data_store import list_interviews, load_interview


async def synthesize_findings(topic_filter: str | None, max_interviews: int) -> dict:
    listing = list_interviews(topic_filter, limit=max_interviews, offset=0)

    if listing["total"] == 0:
        return {
            "error": "No interviews found matching the filter.",
            "hint": "Run researchos_list_interviews to see available interviews."
        }

    transcript_blocks = []
    for item in listing["interviews"]:
        interview = load_interview(item["id"])
        if not interview:
            continue
        turns_text = "\n".join(
            [f"  {t['role'].capitalize()}: {t['text']}" for t in interview["turns"]]
        )
        transcript_blocks.append(
            f"--- Interview {interview['id']} | Persona: {interview['participant_persona']} ---\n"
            f"{turns_text}"
        )

    all_transcripts = "\n\n".join(transcript_blocks)
    n = len(transcript_blocks)

    prompt = (
        f"Analyze the following {n} user research interviews and produce a synthesis report.\n\n"
        f"TRANSCRIPTS:\n{all_transcripts}\n\n"
        "CRITICAL: Only use quotes that appear VERBATIM in the transcripts above. "
        "Do not paraphrase, invent, or combine quotes. "
        "If you cannot find a verbatim quote, omit it.\n\n"
        "Return a JSON object with exactly these keys:\n"
        "- themes: array of objects, each with:\n"
        "    name (string)\n"
        "    description (string)\n"
        "    frequency (int — how many interviews mentioned it)\n"
        "    supporting_quotes: array of objects with quote (verbatim) and interview_id\n"
        "- contradictions: array of strings (findings that differ across participants)\n"
        "- top_pain_points: array of top 3 pain point strings\n"
        "- recommended_research_actions: array of 2-4 next-step strings\n"
        "- interviews_analyzed: integer"
    )

    result_str = await call_llm_json(
        prompt,
        system=(
            "You are an expert UX research analyst. "
            "Synthesize findings accurately. Output only valid JSON. "
            "Never fabricate quotes."
        ),
        max_tokens=3000
    )
    return json.loads(result_str)
