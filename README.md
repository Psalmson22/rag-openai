# 📄 Document Q&A RAG System

A Document Q&A RAG (Retrieval-Augmented Generation) system built with the **OpenAI Agents SDK** and **OpenAI Vector Store**.

## Project Structure

```
rag-app/
├── src/
│   ├── index.ts          # CLI entrypoint & chat loop
│   ├── agent.ts          # RAG agent definition & workflow runner
│   └── vectorStore.ts    # Vector Store management (create, upload, list)
├── package.json
├── tsconfig.json
└── .env                  # Your API key goes here (create this file)
```

## Setup

### 1. Install dependencies
```bash
npm install
```

### 2. Create a `.env` file
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Load env and run
```bash
# Development
export $(cat .env | xargs) && npm run dev

# Or build and run
npm run build && npm start
```

## How It Works

```
User Question
     │
     ▼
[ RAG Agent ]
     │
     ├──► File Search Tool → Vector Store → Top-5 relevant chunks
     │
     └──► GPT-4o generates answer grounded in retrieved chunks
```

1. **Upload documents** → OpenAI chunks, embeds, and indexes them into a Vector Store
2. **User asks a question** → Agent queries the Vector Store for relevant chunks
3. **GPT-4o answers** → Using only the retrieved document context

## Supported File Types
`.pdf`, `.txt`, `.md`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.json`, `.csv`, `.py`, `.js`, `.ts`

## Features
- ✅ Multi-turn conversation memory
- ✅ Automatic document chunking & embedding via OpenAI
- ✅ Top-5 chunk retrieval per query
- ✅ Grounded answers — agent cites document sources
- ✅ Interactive CLI with setup wizard
- ✅ Reuse existing Vector Stores across sessions
