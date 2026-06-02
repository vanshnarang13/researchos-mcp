# ResearchOS MCP

An MCP server that gives Claude the ability to run user research interviews, search past transcripts, and synthesize findings across sessions. All four capabilities are exposed as tools so Claude can chain them from a single prompt.

```
"Run a 5-question interview about checkout friction, search what came up about
 confusion, then synthesize findings into a report."
```

---

## Tools

| Tool | Description |
|------|-------------|
| `researchos_run_interview` | Generates open-ended questions, detects vague answers with a two-layer probe system, and returns a structured summary with themes and key quotes |
| `researchos_search_insights` | Semantic search over all past transcripts. Embeddings run locally, no extra API cost per query |
| `researchos_synthesize_findings` | Cross-interview analysis with recurring themes, verbatim supporting quotes, contradictions across participants, and recommended next steps |
| `researchos_list_interviews` | Paginated list of past interviews, good starting point before searching or synthesizing |

> **Note on demo data:** The 25 seed transcripts are synthetically generated and simulated. They exist so you can try search and synthesis immediately without running real interviews first. The interview engine itself works the same way on real topics.

---

## Architecture

```
Claude Desktop
     |
     |  MCP (stdio / streamable-HTTP)
     v
+------------------------------------------+
|              server.py                   |
|         FastMCP · 4 tools                |
+----+-------------+----------+------------+
     |             |          |
     v             v          v
moderator.py   search.py   synthesizer.py
     |             |
     |    sentence-transformers
     |    (all-MiniLM-L6-v2, MPS)
     |             |
     |          ChromaDB
     |       (in-process, on-disk)
     |
  OpenAI
 gpt-4o-mini
```

**Probe detection flow inside moderator.py:**

```
participant answer
       |
       v
heuristic check ---- too short or vague phrase? --> fire probe
       |
    unclear
       |
       v
  under 25 words?
       | yes
       v
  LLM judge ------------------------------------> fire probe / accept
```

---

## Local setup

Requires Python 3.11+ and an OpenAI API key.

```bash
git clone https://github.com/vanshnarang13/researchos-mcp
cd researchos-mcp

python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your OPENAI_API_KEY to .env

python seed.py    # generates 25 synthetic interviews and embeds them, takes about 2 min
python server.py  # starts the server
```

## Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "researchos": {
      "command": "/absolute/path/to/researchos-mcp/venv/bin/python",
      "args": ["/absolute/path/to/researchos-mcp/server.py"]
    }
  }
}
```

Restart Claude Desktop and the four `researchos_*` tools will show up.

## Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector venv/bin/python server.py
```

---

## Design decisions

**Two-layer probe detection**

The vague answer check runs a heuristic first: if the response is under 10 words, or contains a filler phrase like "it was fine" alongside a short word count, it fires a probe immediately. If the heuristic is inconclusive and the answer is under 25 words, it falls back to an LLM judge. This avoids calling the LLM on every single turn, which would add latency and cost, while still catching the cases that word count alone misses.

**Turn-based chunking**

Each participant turn is stored as one vector. The moderator question that preceded it is stored as `question_context` metadata. This keeps each chunk semantically self-contained and preserves the Q&A pairing when results are surfaced. Splitting on token count would break answers mid-sentence and lose the context of what was asked.

**Four separate tools**

List, search, interview, and synthesize are exposed as individual tools rather than one combined endpoint. This lets Claude compose them from natural language prompts without the caller needing to know the full workflow upfront. It also makes partial workflows possible, for example running search without synthesis.

**Local embeddings and in-process vector store (demo choice)**

For this demo, `all-MiniLM-L6-v2` runs on-device via MPS and ChromaDB persists to disk. Semantic search has no marginal API cost and low latency. In a production system I would swap these out: `text-embedding-3-small` from OpenAI or a Cohere embed model for embeddings, and either pgvector on Postgres or Pinecone/Qdrant as the vector store. pgvector is a good default if you are already running Postgres since it keeps embeddings and relational data in one place and you get transactions, row-level security, and backups for free. A dedicated store like Pinecone or Qdrant makes more sense if the vector index needs to scale independently or if you are doing high-throughput nearest-neighbor lookups across hundreds of millions of vectors.

**Hallucination guard on synthesis**

The synthesis prompt explicitly says: "Only use quotes that appear verbatim in the transcripts above. Do not paraphrase, invent, or combine quotes." LLMs tend to smooth and recombine language when summarizing, which in a research context means fabricated quotes. Enforcing verbatim sourcing at the prompt level keeps output traceable back to real participant words.

---

## What I would build next

**Real participant interviews via streaming** - the current interview engine simulates a participant so the demo works end to end. The real version would use SSE over streamable-HTTP transport so a human participant can answer in a browser while the moderator responds in real time.

**Cluster before synthesizing at scale** - passing 10 transcripts as context works well. At 100+ it hits context limits and the synthesis gets diluted. The fix is to cluster embeddings first, synthesize per cluster, then merge. Quality stays consistent regardless of corpus size.

**Quote-level contradiction tracking** - right now contradictions appear as a flat list in the synthesis output. A more useful version would tag each quote with the claim it supports and automatically pair up contradicting quotes with links back to the specific interviews and personas. That gives researchers something they can actually act on.
