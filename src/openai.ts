import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import * as pdfjs from 'pdfjs-dist/legacy/build/pdf.mjs';
import Highlight from "./types/highlight";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { captureImageFromPDF } from "./pdf";
import { HumanMessage } from "@langchain/core/messages";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/legacy/build/pdf.worker.min.mjs`;

// Interface definition for caching
interface VectorStoreCache {
    [key: string]: {
        [embeddingModel: string]: MemoryVectorStore;
    };
}

// Global cache object declaration
let vectorStoreCache: VectorStoreCache = {};

function readShowRetrievalDetailLogs(): boolean {
    return Boolean((logseq.settings as any)?.showRetrievalDetailLogs);
}

async function showVectorStoreDebug(message: string, level: "success" | "warning" | "error" = "warning") {
    console.log(`[storePdfOnVectorStore] ${message}`);
    if (readShowRetrievalDetailLogs()) {
        await logseq.UI.showMsg(`[storePdfOnVectorStore] ${message}`, level);
    }
}

const RETRIEVAL_LOG_MAX_DOCS = 12;
const RETRIEVAL_LOG_MAX_CHARS_PER_DOC = 600;
const RETRIEVAL_LOG_MAX_TOTAL_CHARS = 12000;

function formatRetrievedDocumentsForMsg(context: unknown): string {
    if (!Array.isArray(context) || context.length === 0) {
        return `Retrieved documents: ${context === undefined ? "none" : JSON.stringify(context)?.slice(0, 500)}`;
    }
    const lines: string[] = [`Retrieved ${context.length} chunk(s):`];
    let total = lines.join("\n").length;
    const maxDocs = Math.min(context.length, RETRIEVAL_LOG_MAX_DOCS);
    for (let i = 0; i < maxDocs; i++) {
        const doc = context[i] as { pageContent?: string; metadata?: Record<string, unknown> };
        const meta = doc.metadata && Object.keys(doc.metadata).length > 0
            ? JSON.stringify(doc.metadata)
            : "{}";
        let text = (doc.pageContent ?? "").replace(/\s+/g, " ").trim();
        if (text.length > RETRIEVAL_LOG_MAX_CHARS_PER_DOC) {
            text = `${text.slice(0, RETRIEVAL_LOG_MAX_CHARS_PER_DOC)}…`;
        }
        const block = `\n---\n[${i + 1}] ${meta}\n${text}`;
        if (total + block.length > RETRIEVAL_LOG_MAX_TOTAL_CHARS) {
            lines.push("\n---\n… (truncated)");
            break;
        }
        lines.push(block);
        total += block.length;
    }
    if (context.length > maxDocs) {
        lines.push(`\n… and ${context.length - maxDocs} more chunk(s) not shown.`);
    }
    return lines.join("");
}

async function showRetrievalContextLog(context: unknown) {
    if (!readShowRetrievalDetailLogs()) return;
    const msg = formatRetrievedDocumentsForMsg(context);
    await logseq.UI.showMsg(msg, "success");
}

function splitIntoBatches<T>(items: T[], batchSize: number) {
    const batches: T[][] = [];
    for (let i = 0; i < items.length; i += batchSize) {
        batches.push(items.slice(i, i + batchSize));
    }
    return batches;
}

export function readOpenAiAPIKey(): string | null {
    return (logseq.settings as any)["openaiApiKey"] ?? logseq.settings?.["openaiApiKey"] ?? null;
}

export function readEmbeddingModelHost(): string | null {
    return (logseq.settings as any)["embeddingModelHost"] ?? logseq.settings?.["embeddingModelHost"];
}

export function readEmbeddingModel(): string {
    return (logseq.settings as any)["embeddingModel"] ?? logseq.settings?.["embeddingModel"] ?? "text-embedding-3-small";
}

export function readLLMModelHost(): string | null {
    return (logseq.settings as any)["llmModelHost"] ?? logseq.settings?.["llmModelHost"];
}

export function readLLMModel(): string {
    return (logseq.settings as any)["llmModel"] ?? logseq.settings?.["llmModel"] ?? "gpt-4o-mini";
}

export async function storePdfOnVectorStore(pdf: Blob, openaiApiKey: string, embeddingModelHost: string | null, embeddingModel: string, pdfPath: string) {
    // Check vector store from cache
    if (vectorStoreCache[pdfPath] && vectorStoreCache[pdfPath][embeddingModel]) {
        console.log("Using cached vector store");
        return vectorStoreCache[pdfPath][embeddingModel];
    }

    try {
        console.log("Creating new vector store");
        const embeddings = new OpenAIEmbeddings({
            openAIApiKey: openaiApiKey,
            model: embeddingModel,
            configuration: embeddingModelHost ? {
                baseURL: embeddingModelHost,
            } : undefined
        });
        const loader = new PDFLoader(pdf, {
            pdfjs: () => pdfjs as any,
        });
        const docs = await loader.load();
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 600,
            chunkOverlap: 300,
        });
        const splitDocs = await splitter.splitDocuments(docs);
        await showVectorStoreDebug(`documents split: docs=${splitDocs.length}`, "success");

        await showVectorStoreDebug("building in-memory vector store");
        const vectorStore = new MemoryVectorStore(embeddings);
        const batches = splitIntoBatches(splitDocs, 10);

        for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            await showVectorStoreDebug(`embedding batch ${i + 1}/${batches.length}: docs=${batch.length}`);
            await vectorStore.addDocuments(batch);
            await showVectorStoreDebug(`embedded batch ${i + 1}/${batches.length}`, "success");
        }

        if (!vectorStoreCache[pdfPath]) {
            vectorStoreCache[pdfPath] = {};
        }
        vectorStoreCache[pdfPath][embeddingModel] = vectorStore;

        return vectorStore;
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        await showVectorStoreDebug(`failed: ${message}`, "error");
        throw error;
    }
}

export async function invoke(highlight: Highlight, pdf: Blob, openaiApiKey: string, llmModelHost: string | null, llmModel: string, vectorStore: MemoryVectorStore, userPrompt?: string) {
    const llm = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        model: llmModel,
        configuration: llmModelHost ? {
            basePath: llmModelHost,
        } : undefined
    });

    if (!highlight.content.image) {
        let promptTemplate: ChatPromptTemplate;
        let input: string | undefined;

        if (userPrompt && highlight.content.text) {
            // User provided a custom prompt alongside the highlight — use the highlight as context for the question
            promptTemplate = ChatPromptTemplate.fromTemplate(
                `Context:\n{context}\n---\nHighlighted text: {input}\n---\nBased on the highlighted text and the context above, answer the following question in markdown format: ${userPrompt}`
            );
            input = highlight.content.text;
        } else {
            promptTemplate = ChatPromptTemplate.fromTemplate(
                (logseq.settings as any)["promptTemplateForText"] ?? logseq.settings?.["promptTemplateForText"] ?? `Context:\n{context}\n---\nExplain following concept and write in markdown format: {input}`
            );
            input = userPrompt || highlight.content.text;
        }

        const combineDocsChain = await createStuffDocumentsChain({
            llm,
            prompt: promptTemplate,
        });
        const retriever = vectorStore.asRetriever();
        const retrievalChain = await createRetrievalChain({
            combineDocsChain,
            retriever,
        });

        if (!input) {
            return null;
        }

        try {
            const result = await retrievalChain.invoke({
                input: input,
            });
            await showRetrievalContextLog((result as { context?: unknown }).context);
            return result;
        } catch (e) {
            console.log(e);
            logseq.UI.showMsg(e as any, "error");
            return null;
        }
    } else {
        // with image
        // directly using the image as the query for vector store is not supported (TODO)
        const image = await captureImageFromPDF(pdf, highlight.position);
        if (!image) {
            logseq.UI.showMsg("Failed to capture image from PDF", "error");
            return null;
        }

        console.log(image);

        // template for image description
        const imageDescriptionMessage = new HumanMessage({
            content: [
                {
                    "type": "text",
                    "text": "Please describe the image below:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                    }
                },
            ]
        });

        // ask the model to describe the image
        const llm = new ChatOpenAI({
            openAIApiKey: openaiApiKey,
            model: llmModel,
        });

        const imageDescription = await llm.invoke([imageDescriptionMessage]);
        console.log(`Image description: ${imageDescription.content}`);

        // query the vector store with the image description
        const promptTemplate = ChatPromptTemplate.fromTemplate(
            (logseq.settings as any)["promptTemplateForImage"] ?? logseq.settings?.["promptTemplateForImage"] ?? `Context:\n{context}\n---\nExplain following described image and write in markdown format: {input}`
        );

        const combineDocsChain = await createStuffDocumentsChain({
            llm,
            prompt: promptTemplate,
        });

        const retriever = vectorStore.asRetriever();

        const retrievalChain = await createRetrievalChain({
            combineDocsChain,
            retriever,
        });

        try {
            const result = await retrievalChain.invoke({
                input: userPrompt || (imageDescription.content as string),
            });
            await showRetrievalContextLog((result as { context?: unknown }).context);
            return result;
        } catch (e) {
            console.log(e);
            logseq.UI.showMsg(e as any, "error");
            return null;
        }
    }
}
