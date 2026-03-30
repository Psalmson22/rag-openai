import "dotenv/config";
import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import multer from "multer";
import * as path from "path";
import * as fs from "fs";
import OpenAI from "openai";
import { runWorkflow } from "./agent";
import {
  createVectorStore,
  uploadFileToStore,
  listVectorStores,
  deleteVectorStore
} from "./vectorStore";
import { AgentInputItem } from "@openai/agents";

const app = express();
const upload = multer({ dest: "uploads/" });
const getClient = () => new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "../public")));

// ── API Key Auth (protects /api routes from public access) ────────────────────
const requireApiKey = (req: Request, res: Response, next: NextFunction) => {
  const appSecret = process.env.APP_SECRET_KEY;
  if (!appSecret) return next(); // no secret set = open (dev mode)
  const provided = req.headers["x-api-key"] ?? req.query.api_key;
  if (provided !== appSecret) return res.status(401).json({ error: "Unauthorized" });
  next();
};
app.use("/api", requireApiKey);

// In-memory session store  { sessionId -> { vectorStoreId, history } }
const sessions: Record<string, { vectorStoreId: string; history: AgentInputItem[] }> = {};

// ── Vector Stores ─────────────────────────────────────────────────────────────
app.get("/api/stores", async (_req, res) => {
  try {
    const stores = await getClient().vectorStores.list();
    res.json({ stores: stores.data.map((s: any) => ({ id: s.id, name: s.name, fileCount: s.file_counts.completed })) });
  } catch (e: any) { res.status(500).json({ error: e.message }); }
});

app.post("/api/stores", async (req, res) => {
  try {
    const { name } = req.body;
    if (!name) return res.status(400).json({ error: "name is required" });
    const store = await getClient().vectorStores.create({ name });
    res.json({ id: store.id, name: store.name });
  } catch (e: any) { res.status(500).json({ error: e.message }); }
});

app.delete("/api/stores/:id", async (req, res) => {
  try {
    await deleteVectorStore(req.params.id);
    res.json({ success: true });
  } catch (e: any) { res.status(500).json({ error: e.message }); }
});

// ── File Upload ───────────────────────────────────────────────────────────────
app.post("/api/stores/:id/upload", upload.array("files"), async (req, res) => {
  const files = req.files as Express.Multer.File[];
  if (!files?.length) return res.status(400).json({ error: "No files provided" });

  const results: { name: string; status: string }[] = [];

  for (const file of files) {
    try {
      await uploadFileToStore(req.params.id, file.path);
      results.push({ name: file.originalname, status: "ok" });
    } catch (e: any) {
      results.push({ name: file.originalname, status: `error: ${e.message}` });
    } finally {
      fs.unlinkSync(file.path); // cleanup temp
    }
  }

  res.json({ results });
});

// ── Chat ──────────────────────────────────────────────────────────────────────
app.post("/api/chat", async (req, res) => {
  const { message, vectorStoreId, sessionId } = req.body;
  if (!message || !vectorStoreId || !sessionId) {
    return res.status(400).json({ error: "message, vectorStoreId, and sessionId are required" });
  }

  if (!sessions[sessionId]) {
    sessions[sessionId] = { vectorStoreId, history: [] };
  }

  const session = sessions[sessionId];

  try {
    const result = await runWorkflow({
      input_as_text: message,
      vectorStoreId,
      conversationHistory: session.history
    });

    session.history = result.updatedHistory;
    res.json({ reply: result.output_text });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

// ── Clear session ─────────────────────────────────────────────────────────────
app.delete("/api/chat/:sessionId", (req, res) => {
  delete sessions[req.params.sessionId];
  res.json({ success: true });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
