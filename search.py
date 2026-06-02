from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import chromadb

_BASE_DIR = Path(__file__).parent

device = "mps" if torch.backends.mps.is_available() else "cpu"
_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
_chroma = chromadb.PersistentClient(path=str(_BASE_DIR / "chroma_db"))
_collection = _chroma.get_or_create_collection("research_transcripts")


def embed_text(text: str) -> list[float]:
    return _model.encode(text).tolist()


def embed_and_store_interview(interview: dict) -> None:
    turns = interview["turns"]
    for i, turn in enumerate(turns):
        if turn["role"] != "participant":
            continue
        question_context = ""
        if i > 0 and turns[i - 1]["role"] == "moderator":
            question_context = turns[i - 1]["text"]
        chunk_id = f"{interview['id']}_turn_{i}"
        try:
            _collection.delete(ids=[chunk_id])
        except Exception:
            pass
        _collection.add(
            ids=[chunk_id],
            embeddings=[embed_text(turn["text"])],
            documents=[turn["text"]],
            metadatas=[{
                "interview_id": interview["id"],
                "participant_persona": interview["participant_persona"],
                "topic": interview["topic"],
                "turn_index": i,
                "question_context": question_context
            }]
        )


def search_insights(query: str, top_k: int) -> list[dict]:
    query_embedding = embed_text(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, _collection.count())
    )
    output = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        output.append({
            "quote": results["documents"][0][i],
            "interview_id": meta["interview_id"],
            "participant_persona": meta["participant_persona"],
            "question_context": meta["question_context"],
            "relevance_score": round(1 - distance, 3)
        })
    return sorted(output, key=lambda x: x["relevance_score"], reverse=True)
