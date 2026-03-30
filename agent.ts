import { Agent, AgentInputItem, Runner, withTrace } from "@openai/agents";
import { fileSearchTool } from "@openai/agents";

// ─── Agent Factory ────────────────────────────────────────────────────────────
// Call createRAGAgent() once you have a vectorStoreId ready.
export const createRAGAgent = (vectorStoreId: string) =>
  new Agent({
    name: "Document Q&A Agent",
    instructions: `You are a helpful document assistant. 
Answer questions using ONLY the information retrieved from the provided documents.
If the answer is not found in the documents, say so clearly — do not make things up.
Always cite which part of the document supports your answer when possible.`,
    model: "gpt-4o",
    tools: [
      fileSearchTool([vectorStoreId], {
        maxNumResults: 5,          // top-5 chunks per query
        includeSearchResults: true // attach raw chunks to output
      })
    ],
    modelSettings: {
      store: true
    }
  });

// ─── Workflow Types ───────────────────────────────────────────────────────────
export type WorkflowInput = {
  input_as_text: string;
  vectorStoreId: string;
  conversationHistory?: AgentInputItem[];
};

export type WorkflowOutput = {
  output_text: string;
  updatedHistory: AgentInputItem[];
};

// ─── Main Workflow ────────────────────────────────────────────────────────────
export const runWorkflow = async (
  workflow: WorkflowInput
): Promise<WorkflowOutput> => {
  return await withTrace("RAG Document Q&A Workflow", async () => {
    const agent = createRAGAgent(workflow.vectorStoreId);

    // Build / extend conversation history
    const conversationHistory: AgentInputItem[] = [
      ...(workflow.conversationHistory ?? []),
      {
        role: "user",
        content: [{ type: "input_text", text: workflow.input_as_text }]
      }
    ];

    const runner = new Runner({
      traceMetadata: { __trace_source__: "rag-agent" }
    });

    const result = await runner.run(agent, conversationHistory);

    // Append assistant response to history for multi-turn support
    const updatedHistory: AgentInputItem[] = [
      ...conversationHistory,
      ...result.newItems.map((item) => item.rawItem)
    ];

    if (!result.finalOutput) {
      throw new Error("Agent returned no output.");
    }

    return {
      output_text: result.finalOutput,
      updatedHistory
    };
  });
};
