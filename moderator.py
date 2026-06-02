import json
import random
from datetime import datetime, timezone

from utils import call_llm, call_llm_json
from data_store import save_interview, next_interview_id
from search import embed_and_store_interview

PERSONAS = [
    "PM at a 200-person fintech company",
    "Solo founder of an early-stage SaaS",
    "Senior engineer at an enterprise company",
    "Head of Operations at a mid-size agency",
    "Customer success manager at a B2B software company",
]


async def generate_interview_guide(topic: str, num_questions: int) -> list[str]:
    prompt = (
        f"Generate exactly {num_questions} open-ended user research interview questions "
        f"about: {topic}\n\n"
        "Requirements:\n"
        "- No yes/no questions\n"
        "- Each question should elicit a story or specific example\n"
        "- Start broad, get more specific\n"
        "- Return ONLY a numbered list, one question per line, no extra text"
    )
    response = await call_llm(prompt, system="You are an expert UX researcher.")
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    questions = []
    for line in lines:
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:num_questions]


async def llm_probe_judge(answer: str, question: str) -> bool:
    prompt = (
        f"Question asked: {question}\n"
        f"Participant answer: {answer}\n\n"
        "Is this answer too vague to be useful for research? "
        "A vague answer gives general sentiment without specific details, "
        "concrete examples, named features, or actionable pain points.\n"
        "Reply with only the word 'probe' or 'accept'."
    )
    result = await call_llm(prompt)
    return "probe" in result.lower()


def should_probe_heuristic(answer: str) -> bool:
    words = answer.split()
    if len(words) < 10:
        return True
    vague_phrases = [
        "it was fine", "pretty good", "not bad", "okay i guess",
        "seems alright", "i guess so", "pretty okay", "it was okay",
        "not really", "kind of", "sort of good", "was fine"
    ]
    lower = answer.lower()
    if any(p in lower for p in vague_phrases) and len(words) < 20:
        return True
    return False


async def should_probe(answer: str, question: str) -> bool:
    if should_probe_heuristic(answer):
        return True
    if len(answer.split()) < 25:
        return await llm_probe_judge(answer, question)
    return False


async def generate_probe(question: str, answer: str) -> str:
    prompt = (
        f"A participant gave a vague answer in a user research interview.\n"
        f"Original question: {question}\n"
        f"Vague answer: {answer}\n\n"
        "Generate ONE natural follow-up probe question that asks for a specific "
        "example, concrete detail, or what happened next. "
        "Keep it conversational, one sentence only."
    )
    return await call_llm(prompt)


async def simulate_participant_answer(
    question: str,
    persona: str,
    topic: str,
    conversation_history: list[dict]
) -> str:
    history_text = "\n".join(
        [f"{t['role'].capitalize()}: {t['text']}" for t in conversation_history]
    )
    prompt = (
        f"You are roleplaying as a research participant: {persona}\n"
        f"Research topic: {topic}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Moderator: {question}\n\n"
        "Respond naturally as this participant. "
        "Sometimes give detailed specific answers (mention product names, what broke, "
        "specific steps). Sometimes be vague (just 'it was fine' or 'pretty manageable'). "
        "Vary the specificity — do not always be vague or always be detailed. "
        "1-4 sentences only. No quotes around your answer."
    )
    return await call_llm(prompt)


async def generate_interview_summary(turns: list[dict], topic: str) -> dict:
    turns_text = "\n".join([f"{t['role'].capitalize()}: {t['text']}" for t in turns])

    prompt = (
        f"Analyze this user research interview about '{topic}' and produce a summary.\n\n"
        f"TRANSCRIPT:\n{turns_text}\n\n"
        "IMPORTANT: Only use quotes that appear VERBATIM in the transcript above. "
        "Do not paraphrase or invent quotes.\n\n"
        "Return a JSON object with exactly these keys:\n"
        "- themes: array of 2-4 theme strings\n"
        "- key_quotes: array of 3-5 verbatim quote strings from participant turns\n"
        "- sentiment: one of 'positive', 'negative', 'mixed'\n"
        "- recommended_followup_topics: array of 2-3 string suggestions"
    )
    result_str = await call_llm_json(
        prompt,
        system="You are a UX research analyst. Output only valid JSON."
    )
    return json.loads(result_str)


async def run_interview_session(
    topic: str,
    num_questions: int,
    simulate: bool,
    participant_persona: str | None
) -> dict:
    interview_id = next_interview_id()
    persona = participant_persona or random.choice(PERSONAS)
    questions = await generate_interview_guide(topic, num_questions)

    turns = []
    probes_triggered = 0

    for question in questions:
        turns.append({"role": "moderator", "text": question})

        if simulate:
            answer = await simulate_participant_answer(question, persona, topic, turns)
        else:
            answer = "[INTERACTIVE MODE — participant answer goes here]"
        turns.append({"role": "participant", "text": answer})

        if await should_probe(answer, question):
            probes_triggered += 1
            probe_question = await generate_probe(question, answer)
            turns.append({"role": "moderator", "text": probe_question})

            if simulate:
                probe_answer = await simulate_participant_answer(
                    probe_question, persona, topic, turns
                )
            else:
                probe_answer = "[INTERACTIVE MODE — probe answer goes here]"
            turns.append({"role": "participant", "text": probe_answer})

    summary = await generate_interview_summary(turns, topic)

    interview = {
        "id": interview_id,
        "topic": topic,
        "participant_persona": persona,
        "turns": turns,
        "summary": summary,
        "probes_triggered": probes_triggered,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    save_interview(interview)
    embed_and_store_interview(interview)
    return interview
