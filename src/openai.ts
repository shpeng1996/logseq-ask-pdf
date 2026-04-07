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
// import blockEntity from LSPlugin
import { BlockEntity } from "@logseq/libs/dist/LSPlugin";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/legacy/build/pdf.worker.min.mjs`;

// Interface definition for caching
interface VectorStoreCache {
    [key: string]: {
        [embeddingModel: string]: MemoryVectorStore;
    };
}

// Global cache object declaration
let vectorStoreCache: VectorStoreCache = {};

function readEnableDetailLogs(): boolean {
    return Boolean((logseq.settings as any)?.EnableDetailLogs);
}

async function showDebug(message: string, level: "success" | "warning" | "error" = "warning") {
    console.log(`${message}`);
    if (readEnableDetailLogs()) {
        await logseq.UI.showMsg(`${message}`, level);
    }
}


function splitIntoBatches<T>(items: T[], batchSize: number) {
    const batches: T[][] = [];
    for (let i = 0; i < items.length; i += batchSize) {
        batches.push(items.slice(i, i + batchSize));
    }
    return batches;
}

function messageContentToString(content: unknown): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        return content
            .map((part) =>
                typeof part === "object" && part !== null && "text" in part
                    ? String((part as { text: string }).text)
                    : ""
            )
            .filter(Boolean)
            .join("\n");
    }
    return String(content ?? "");
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
        await showDebug(`[storePdfOnVectorStore] documents split: docs=${splitDocs.length}`, "success");

        await showDebug("[storePdfOnVectorStore] building in-memory vector store");
        const vectorStore = new MemoryVectorStore(embeddings);
        const batches = splitIntoBatches(splitDocs, 10);

        for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            await showDebug(`[storePdfOnVectorStore] embedding batch ${i + 1}/${batches.length}: docs=${batch.length}`);
            await vectorStore.addDocuments(batch);
            await showDebug(`[storePdfOnVectorStore] embedded batch ${i + 1}/${batches.length}`, "success");
        }

        if (!vectorStoreCache[pdfPath]) {
            vectorStoreCache[pdfPath] = {};
        }
        vectorStoreCache[pdfPath][embeddingModel] = vectorStore;

        return vectorStore;
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        await showDebug(`[storePdfOnVectorStore] failed: ${message}`, "error");
        throw error;
    }
}

export async function invoke(highlight: Highlight, 
            pdf: Blob, openaiApiKey: string, llmModelHost: string | null, 
            llmModel: string, vectorStore: MemoryVectorStore, 
            userPrompt?: string, currentBlock?: BlockEntity) {

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
            // await showRetrievalContextLog((result as { context?: unknown }).context);
        
            if (readEnableDetailLogs()) {
                const currentBlockId = currentBlock?.uuid;
                if (currentBlockId && input) {
                    const contextHeader = "*Ask PDF Retrieval INPUT*";
                    const contextBlock = await logseq.Editor.insertBlock(currentBlockId, contextHeader);
                    if (contextBlock) {
                        // Run the chain first to get the actual context
                        // const result = await retrievalChain.invoke({ input: input });
                        const retrievedDocs = (result.context as any[])
                            ?.map((d: any) => d.pageContent)
                            .join("\n---\n") ?? "(no context retrieved)";
                        const fullPrompt = await promptTemplate.format({ context: retrievedDocs, input: input });

                        // remove metadata from fullPrompt
                        const cleanedPrompt = fullPrompt.replace(/^\s*[\w-]+::.*\n?/gm, "");
                        await logseq.Editor.insertBlock(contextBlock.uuid, cleanedPrompt);
                    }
                }
            }
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
                    "text": "Please briefly describe the image below:",
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
        const imageDescription = await llm.invoke([imageDescriptionMessage]);
        
        console.log(`Image description: ${imageDescription.content}`);
        const describedImage = (imageDescription.content as string) ?? "";
        if (!describedImage.trim()) {
            logseq.UI.showMsg("Empty image description from model", "error");
            return null;
        }

        // query the vector store with the image description ({input} is always the described region, like highlighted text in text mode)
        let promptTemplate: ChatPromptTemplate;
        if (userPrompt) {
            promptTemplate = ChatPromptTemplate.fromTemplate(
                `Context:\n{context}\n---\nDescribed image region: {input}\n---\nBased on the described image and the context above, answer the following question in markdown format: ${userPrompt}`
            );
        } else {
            promptTemplate = ChatPromptTemplate.fromTemplate(
                (logseq.settings as any)["promptTemplateForImage"] ?? logseq.settings?.["promptTemplateForImage"] ?? `Context:\n{context}\n---\nExplain following described image and write in markdown format: {input}`
            );
        }

        const retriever = vectorStore.asRetriever();

        try {
            const contextDocs = await retriever.invoke(describedImage);
            const retrievedDocs = contextDocs.map((d) => d.pageContent).join("\n---\n");

            const basePrompt = await promptTemplate.format({ context: retrievedDocs, input: describedImage });
            const instructionText = `${basePrompt}\n\nThe next part is the original image of the highlighted region from the PDF. Use it together with the context and your instructions when answering.`;

            const finalMessage = new HumanMessage({
                content: [
                    { type: "text", text: instructionText },
                    { type: "image_url", image_url: { url: image } },
                ],
            });

            const answerMessage = await llm.invoke([finalMessage]);
            const result = {
                answer: messageContentToString(answerMessage.content),
                context: contextDocs,
            };

            if (readEnableDetailLogs()) {
                const currentBlockId = currentBlock?.uuid;
                if (currentBlockId && describedImage) {
                    const contextHeader = "*Ask PDF Retrieval INPUT*";
                    const contextBlock = await logseq.Editor.insertBlock(currentBlockId, contextHeader);
                    if (contextBlock) {
                        const fullPrompt = basePrompt;
                        const cleanedPrompt = fullPrompt.replace(/^\s*[\w-]+::.*\n?/gm, "");
                        await logseq.Editor.insertBlock(contextBlock.uuid, cleanedPrompt);
                    }
                }
            }
            return result;
        } catch (e) {
            console.log(e);
            logseq.UI.showMsg(e as any, "error");
            return null;
        }
    }
}
