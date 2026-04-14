# Documentation Assistant API
A RAG system that lets you query in plain english.
Upload any document, ask question, get answer grounded in your content with source reference.

# Live demo
comming soon - deploying to railway

# Architecture
Document -> token-aware chunking -> sentence-transformer(384-dm) -> pgvector storage -> ivfflat index -> cosine similarity search -> Groq LLaMA 3.3 70B generation

## key technical decision

**Token aware chunking over sentence spliting**
Sentence spliting breaks on poorly formatted documents.
Token-aware chunking with overlap guarantees consistent chunk sizes matching the embedding model's input limit.

**ivfflat index with probes=3**
full cosine scan is O(n)-usuable at scale. Ivfflat clusters vectors into lists, reducing search to nearby clusters only. Probes=3 searches 3 nearest clusters, balancing recall (-95%) vs speed.

**flush before embded_batch**
Ensure document ID is available for chunk foreignkey while keeping entire operations in one transaction. If embedding fails document and chunks both rollback.

**Groq over OpenAI**
200-400 tokens/sec vs 20-40 tokens/sec . RAG has already latency from embedding + vector search. Groq keeps total response time under 2 seconds.

## Stack
- FastAPI
- Postgresql + pgvector
- sentence-transformer(all-MiniLM-L6-v2)
- SqlAlchemy + Alembic
- Groq LLaMA 3.3 70B
- Docker

## Run locally

### Prerequisties
- Docker and Docker Compose
- Groq API Key

### Setup
```bash
git clone https://github.com/contactmahiul/doc-assistant-api
cd doc-assistant-api
cp .env.example
docker compose up
```
### Run migration
```bash
docker  exec app alembic upgrade head
```
### API Endpoints
- `POST /api/v1/document/` — ingest a document
- `POST /api/v1/query/`     — semantic search
- `POST /api/v1/chat/`      — RAG question answering
- `GET  /docs`              — Swagger UI

## API Usage

**Ingest a document:**
```json
POST /api/v1/document/
{
  "title": "Your Document Title",
  "content": "Your document content here..."
}
```

**Ask a question:**
```json
POST /api/v1/chat/
{
  "question": "What is X?",
  "top_k": 5
}
```