import OpenAI from "openai";
import * as fs from "fs";
import * as path from "path";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ─── Supported file types by OpenAI Vector Store ─────────────────────────────
const SUPPORTED_EXTENSIONS = new Set([
  ".pdf", ".txt", ".md", ".docx", ".pptx", ".xlsx",
  ".html", ".json", ".csv", ".py", ".js", ".ts"
]);

// ─── Create a new Vector Store ────────────────────────────────────────────────
export const createVectorStore = async (name: string): Promise<string> => {
  const store = await openai.beta.vectorStores.create({ name });
  console.log(`✅ Vector Store created: ${store.id} ("${name}")`);
  return store.id;
};

// ─── Upload a single file to an existing Vector Store ────────────────────────
export const uploadFileToStore = async (
  vectorStoreId: string,
  filePath: string
): Promise<void> => {
  const ext = path.extname(filePath).toLowerCase();

  if (!SUPPORTED_EXTENSIONS.has(ext)) {
    console.warn(`⚠️  Skipping unsupported file type: ${filePath}`);
    return;
  }

  console.log(`📤 Uploading: ${path.basename(filePath)} ...`);

  const fileStream = fs.createReadStream(filePath);
  await openai.beta.vectorStores.fileBatches.uploadAndPoll(vectorStoreId, {
    files: [fileStream]
  });

  console.log(`✅ Uploaded & indexed: ${path.basename(filePath)}`);
};

// ─── Upload an entire folder to an existing Vector Store ─────────────────────
export const uploadFolderToStore = async (
  vectorStoreId: string,
  folderPath: string
): Promise<void> => {
  const entries = fs.readdirSync(folderPath);

  for (const entry of entries) {
    const fullPath = path.join(folderPath, entry);
    const stat = fs.statSync(fullPath);

    if (stat.isFile()) {
      await uploadFileToStore(vectorStoreId, fullPath);
    }
  }
};

// ─── List all Vector Stores ───────────────────────────────────────────────────
export const listVectorStores = async (): Promise<void> => {
  const stores = await openai.beta.vectorStores.list();
  if (stores.data.length === 0) {
    console.log("No vector stores found.");
    return;
  }
  console.log("\n📚 Your Vector Stores:");
  for (const store of stores.data) {
    console.log(`  - ${store.id}  "${store.name}"  (${store.file_counts.completed} files)`);
  }
};

// ─── Delete a Vector Store ────────────────────────────────────────────────────
export const deleteVectorStore = async (vectorStoreId: string): Promise<void> => {
  await openai.beta.vectorStores.del(vectorStoreId);
  console.log(`🗑️  Deleted vector store: ${vectorStoreId}`);
};
