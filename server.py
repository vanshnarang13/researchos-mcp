import os
import json
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import moderator
import synthesizer
import data_store
from search import search_insights

mcp = FastMCP("researchos_mcp")


# ── Input models ──────────────────────────────────────────

class RunInterviewInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    topic: str = Field(...,
        description="Research topic to explore, e.g. 'B2B SaaS onboarding friction' "
                    "or 'checkout flow usability'. Be specific for better questions.",
        min_length=5, max_length=300)
    num_questions: int = Field(default=5,
        description="Number of interview questions to ask (2-8 recommended)",
        ge=2, le=10)
    simulate: bool = Field(default=True,
        description="If True, simulate a synthetic participant for demo purposes.")
    participant_persona: Optional[str] = Field(default=None,
        description="Persona for the simulated participant, e.g. 'PM at a 50-person startup'. "
                    "If not provided, a random persona is selected.")


class SearchInsightsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    query: str = Field(...,
        description="Natural language question, e.g. 'What do users struggle with "
                    "during initial setup?' or 'confusion about API documentation'",
        min_length=3, max_length=500)
    top_k: int = Field(default=5,
        description="Number of quote results to return",
        ge=1, le=20)
    include_synthesis: bool = Field(default=False,
        description="If True, also generate a 2-3 sentence AI synthesis of the top results.")


class SynthesizeFindingsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    topic_filter: Optional[str] = Field(default=None,
        description="Filter interviews by topic keyword, e.g. 'onboarding'. "
                    "If None, all interviews are included.")
    max_interviews: int = Field(default=10,
        description="Maximum number of interviews to include. More = richer but slower.",
        ge=1, le=50)


class ListInterviewsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    limit: int = Field(default=10,
        description="Maximum number of interviews to return",
        ge=1, le=50)
    offset: int = Field(default=0,
        description="Number of interviews to skip (for pagination)",
        ge=0)
    topic_filter: Optional[str] = Field(default=None,
        description="Filter by topic keyword (case-insensitive substring match)")


# ── Tools ─────────────────────────────────────────────────

@mcp.tool(
    name="researchos_run_interview",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def researchos_run_interview(params: RunInterviewInput) -> str:
    """Run a complete AI-moderated user research interview on a given topic.

    The AI moderator generates open-ended questions, detects vague answers using
    a two-layer heuristic+LLM probe detection system, and generates follow-up probes
    to get specific, actionable responses. At the end produces a structured summary
    with themes, key quotes, and recommended follow-up topics.

    The completed interview is automatically saved to disk and embedded into the
    vector database for future search and synthesis.

    Returns JSON with:
      - interview_id: unique identifier (e.g. 'int_026')
      - topic: the research topic
      - participant_persona: participant background
      - turns: full transcript [{role, text}]
      - summary: {themes, key_quotes, sentiment, recommended_followup_topics}
      - probes_triggered: number of times probe detection fired
      - created_at: ISO timestamp
    """
    try:
        result = await moderator.run_interview_session(
            topic=params.topic,
            num_questions=params.num_questions,
            simulate=params.simulate,
            participant_persona=params.participant_persona
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@mcp.tool(
    name="researchos_search_insights",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def researchos_search_insights(params: SearchInsightsInput) -> str:
    """Search across all past interview transcripts using semantic similarity.

    Embeds the query locally (no API cost) and retrieves the most semantically
    similar participant quotes from the vector store. Each result includes the
    verbatim quote, which interview it came from, the participant persona, and
    the question context that prompted the answer.

    Optionally generates a brief AI synthesis of what the top results say about
    the query topic.

    Returns JSON with:
      - query: the original search query
      - results: [{quote, interview_id, participant_persona,
                   question_context, relevance_score}]
      - synthesis: string (only if include_synthesis=True)
      - total_interviews_in_db: int
    """
    try:
        results = search_insights(params.query, params.top_k)
        response = {
            "query": params.query,
            "results": results,
            "total_interviews_in_db": data_store.count_interviews()
        }
        if params.include_synthesis and results:
            from utils import call_llm
            quotes_text = "\n".join([f"- {r['quote']}" for r in results])
            synthesis = await call_llm(
                f"In 2-3 sentences, synthesize what these user research quotes reveal "
                f"about: {params.query}\n\nQuotes:\n{quotes_text}"
            )
            response["synthesis"] = synthesis
        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"{type(e).__name__}: {str(e)}",
            "hint": "Make sure seed.py has been run to populate the database."
        })


@mcp.tool(
    name="researchos_synthesize_findings",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def researchos_synthesize_findings(params: SynthesizeFindingsInput) -> str:
    """Generate a structured synthesis report across multiple research interviews.

    Loads up to max_interviews transcripts, passes them to the LLM as context,
    and produces a structured analysis with recurring themes, supporting quotes,
    contradictions across participants, top pain points, and recommended actions.

    All quotes in the output are verbatim from source transcripts — the prompt
    explicitly prohibits hallucinated or paraphrased quotes.

    Returns JSON with:
      - themes: [{name, description, frequency,
                  supporting_quotes: [{quote, interview_id}]}]
      - contradictions: [string]
      - top_pain_points: [string]
      - recommended_research_actions: [string]
      - interviews_analyzed: int
    """
    try:
        result = await synthesizer.synthesize_findings(
            topic_filter=params.topic_filter,
            max_interviews=params.max_interviews
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@mcp.tool(
    name="researchos_list_interviews",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def researchos_list_interviews(params: ListInterviewsInput) -> str:
    """List all research interviews stored in the system, with pagination support.

    Returns a paginated index showing interview IDs, topics, participant personas,
    creation dates, probe counts, and whether a summary has been generated.

    Use this tool first to understand what research exists before calling
    search or synthesize — it shows which topics are covered and which
    interviews may be most relevant.

    Returns JSON with:
      - total: int
      - count: int (interviews in this page)
      - offset: int
      - has_more: bool
      - interviews: [{id, topic, participant_persona, created_at,
                      probes_triggered, has_summary}]
    """
    try:
        result = data_store.list_interviews(
            topic_filter=params.topic_filter,
            limit=params.limit,
            offset=params.offset
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


# ── Entrypoint ────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "streamable_http":
        port = int(os.getenv("PORT", "8000"))
        mcp.run(transport="streamable_http", port=port)
    else:
        mcp.run()
