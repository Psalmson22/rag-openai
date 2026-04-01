import "dotenv/config";
import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import multer from "multer";
import * as path from "path";
import * as fs from "fs";
import OpenAI from "openai";
import { runWorkflow } from "./agent";
import { uploadFileToStore, deleteVectorStore } from "./vectorStore";
import { AgentInputItem } from "@openai/agents";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { z } from "zod";

const getClient = () => {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY is not set in environment variables");
  return new OpenAI({ apiKey: key });
};

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "../public")));

// ── OAuth for ChatGPT ─────────────────────────────────────────────────────────
const VALID_TOKEN = process.env.APP_SECRET_KEY || "dev-token";
const CLIENT_ID = process.env.OAUTH_CLIENT_ID || "docmind-client";
const CLIENT_SECRET = process.env.OAUTH_CLIENT_SECRET || VALID_TOKEN;

app.get("/oauth/authorize", (req, res) => {
  const { redirect_uri, state } = req.query;
  const code = Buffer.from(VALID_TOKEN).toString("base64");
  res.redirect(`${redirect_uri}?code=${code}&state=${state}`);
});

app.post("/oauth/token", (req, res) => {
  const { code, client_id, client_secret, grant_type } = req.body;
  if (client_id !== CLIENT_ID || client_secret !== CLIENT_SECRET) {
    return res.status(401).json({ error: "invalid_client" });
  }
  if (grant_type === "client_credentials") {
    return res.json({ access_token: VALID_TOKEN, token_type: "bearer", expires_in: 86400 });
  }
  const decoded = Buffer.from(code as string, "base64").toString();
  if (decoded !== VALID_TOKEN) return res.status(400).json({ error: "invalid_grant" });
  res.json({ access_token: VALID_TOKEN, token_type: "bearer", expires_in: 86400 });
});

app.get("/.well-known/oauth-authorization-server", (req, res) => {
  const base = `${req.protocol}://${req.get("host")}`;
  res.json({
    issuer: base,
    authorization_endpoint: `${base}/oauth/authorize`,
    token_endpoint: `${base}/oauth/token`,
    response_types_supported: ["code"],
    grant_types_supported: ["authorization_code", "client_credentials"],
    token_endpoint_auth_methods_supported: ["client_secret_post"]
  });
});

// ── Auth middleware ───────────────────────────────────────────────────────────
const requireBearer = (req: Request, res: Response, next: NextFunction) => {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith("Bearer ")) return res.status(401).json({ error: "Missing bearer token" });
  if (auth.slice(7) !== VALID_TOKEN) return res.status(401).json({ error: "Invalid token" });
  next();
};

const requireApiKey = (req: Request, res: Response, next: NextFunction) => {
  const appSecret = process.env.APP_SECRET_KEY;
  if (!appSecret) return next();
  const provided = req.headers["x-api-key"] ?? req.query.api_key;
  if (provided !== appSecret) return res.status(401).json({ error: "Unauthorized" });
  next();
};

app.use("/api", requireApiKey);
app.use("/mcp", requireBearer);

// In-memory session store
const sessions: Record<string, { vectorStoreId: string; history: AgentInputItem[] }> = {};

// ── REST API (Web UI) ─────────────────────────────────────────────────────────
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
    } finally { fs.unlinkSync(file.path); }
  }
  res.json({ results });
});

app.post("/api/chat", async (req, res) => {
  const { message, vectorStoreId, sessionId } = req.body;
  if (!message || !vectorStoreId || !sessionId) {
    return res.status(400).json({ error: "message, vectorStoreId, and sessionId are required" });
  }
  if (!sessions[sessionId]) sessions[sessionId] = { vectorStoreId, history: [] };
  const session = sessions[sessionId];
  try {
    const result = await runWorkflow({ input_as_text: message, vectorStoreId, conversationHistory: session.history });
    session.history = result.updatedHistory;
    res.json({ reply: result.output_text });
  } catch (e: any) { res.status(500).json({ error: e.message }); }
});

app.delete("/api/chat/:sessionId", (req, res) => {
  delete sessions[req.params.sessionId];
  res.json({ success: true });
});

// ── MCP (ChatGPT) — new McpServer instance per connection ────────────────────
const createMcpServer = () => {
  const server = new McpServer({ name: "DocMind RAG", version: "1.0.0" });

  server.tool(
    "listKnowledgeBases",
    "List all available knowledge bases with their IDs and file counts",
    {},
    async () => {
      const stores = await getClient().vectorStores.list();
      const result = stores.data.map((s: any) => ({
        id: s.id, name: s.name || "Unnamed", fileCount: s.file_counts.completed
      }));
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    "askQuestion",
    "Ask a question and get an answer grounded in the documents of a knowledge base",
    {
      message: z.string().describe("The user question"),
      vectorStoreId: z.string().describe("Knowledge base ID from listKnowledgeBases"),
      sessionId: z.string().describe("Unique session ID, reuse across conversation turns")
    },
    async ({ message, vectorStoreId, sessionId }) => {
      if (!sessions[sessionId]) sessions[sessionId] = { vectorStoreId, history: [] };
      const session = sessions[sessionId];
      const outcome = await runWorkflow({ input_as_text: message, vectorStoreId, conversationHistory: session.history });
      session.history = outcome.updatedHistory;
      return { content: [{ type: "text", text: outcome.output_text }] };
    }
  );

  return server;
};

const transports: Record<string, SSEServerTransport> = {};

app.get("/mcp/sse", async (req, res) => {
  const transport = new SSEServerTransport("/mcp/messages", res);
  transports[transport.sessionId] = transport;
  res.on("close", () => delete transports[transport.sessionId]);
  // Create a fresh McpServer instance for each connection
  const server = createMcpServer();
  await server.connect(transport);
});

app.post("/mcp/messages", async (req, res) => {
  const sessionId = req.query.sessionId as string;
  const transport = transports[sessionId];
  if (!transport) return res.status(400).json({ error: "No active session" });
  await transport.handlePostMessage(req, res);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));