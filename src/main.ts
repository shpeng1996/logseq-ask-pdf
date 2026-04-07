import "@logseq/libs";
import settingUI from "./settings";
import { Buffer } from 'buffer'
import { invoke, readOpenAiAPIKey, readEmbeddingModelHost, readEmbeddingModel, readLLMModelHost, readLLMModel, storePdfOnVectorStore } from "./openai";
import { findPageProperty } from "./page";
import { findHighlightFromEdnByUuid, findUuidFromAnnotationBlock, findUuidOfCurrentLine, formatPdfDebugInfo, getPdfAndEdnByPdfPath } from "./pdf";
import { PageEntity } from "@logseq/libs/dist/LSPlugin";

globalThis.Buffer = Buffer


async function main() {
    settingUI();

    logseq.Editor.registerSlashCommand(
        "ask pdf",
        async () => {
            const currentBlock = await logseq.Editor.getCurrentBlock();
            if (currentBlock == null) return;

            ///////////////////////////////
            // check openai api key exist in settings
            ///////////////////////////////
            const openaiApiKey = readOpenAiAPIKey();
            const embeddingModelHost = readEmbeddingModelHost();
            const embeddingModel = readEmbeddingModel();
            const llmModelHost = readLLMModelHost();
            const llmModel = readLLMModel();
            if (!openaiApiKey) {
                await logseq.UI.showMsg("OpenAI API key is not set. Please set it in the plugin settings.", "error")
                return
            }
            
            ///////////////////////////////
            // parse current block
            ///////////////////////////////
            const uuid =
                findUuidFromAnnotationBlock(currentBlock) ?? findUuidOfCurrentLine(currentBlock.content);
            if (!uuid) {
                await logseq.UI.showMsg(`Please check whether the highlight uuid is on current line.`, "warning");
                return;
            }

            // Extract user prompt: everything in the block content except the ((uuid)) reference
            const userPrompt = currentBlock.content.replace(/\(\(.*?\)\)/g, "").trim() || undefined;

            ///////////////////////////////
            // find pdf
            ///////////////////////////////
            const block = await logseq.Editor.getBlock(uuid);
            let pdfPath: string = "";

            if (block?.page) {
                const pageId =
                    typeof block.page === "number"
                        ? block.page
                        : block.page.id;

                const page = await logseq.Editor.getPage(pageId);

                if (page) {
                    pdfPath = findPageProperty(page as PageEntity, "filePath");
                }
            }
            if (!pdfPath) {
                await logseq.UI.showMsg(`Before using the plugin, set 'ask-pdf-path' property.`, "warning")
                return;
            }

            const pdfInfo = await getPdfAndEdnByPdfPath(pdfPath);
            if (!pdfInfo.ok) {
                await logseq.UI.showMsg(`Please check whether the pdfPath is valid.\n${formatPdfDebugInfo(pdfInfo.debug)}`, "warning");
                return;
            }

            const { pdf, edn } = pdfInfo;

            ///////////////////////////////
            // find highlights
            ///////////////////////////////

            const highlight = findHighlightFromEdnByUuid(uuid, edn);
            if (!highlight) {
                await logseq.UI.showMsg(`Please check whether the highlight uuid is on current line.`, "warning");
                return;
            }

            ///////////////////////////////
            // upload pdf to langchain vec db
            ///////////////////////////////
            
            const embeddingBlock = await logseq.Editor.insertBlock(currentBlock.uuid, "EMBEDDING.....");
            const vectorStore = await storePdfOnVectorStore(pdf, openaiApiKey, embeddingModelHost, embeddingModel, pdfPath);
            if (embeddingBlock) await logseq.Editor.removeBlock(embeddingBlock.uuid);

            ///////////////////////////////
            // ask to gpt
            ///////////////////////////////
            const loadingBlock = await logseq.Editor.insertBlock(currentBlock.uuid, "LOADING.....");

            const chatResponse = await invoke(highlight, pdf, openaiApiKey, llmModelHost, llmModel, vectorStore, userPrompt, currentBlock);

            if (loadingBlock) await logseq.Editor.removeBlock(loadingBlock.uuid);
            if (chatResponse) {
                let answerText = chatResponse.answer.trim();

                // extract thinking content from <think>...</think> tags (for thinking models)
                let thinkingContent = "";
                const thinkMatch = answerText.match(/<think>([\s\S]*?)<\/think>/);
                if (thinkMatch) {
                    thinkingContent = thinkMatch[1].trim();
                    answerText = answerText.replace(/<think>[\s\S]*?<\/think>/, "").trim();
                }

                // remove wrapping ``` code fences if present
                const lines = answerText.split("\n");
                if (lines[0].trim().startsWith("```")) lines.shift();
                if (lines.length > 0 && lines[lines.length - 1].trim().startsWith("```")) lines.pop();
                answerText = lines.join("\n").trim();

                ///////////////////////////////
                // write the answer under the current block
                ///////////////////////////////
                const askPdfBlock = await logseq.Editor.insertBlock(currentBlock.uuid, "*Ask PDF Response*");
                if (askPdfBlock) {
                    if (thinkingContent) {
                        await logseq.Editor.insertBlock(askPdfBlock.uuid, "#+BEGIN_NOTE\n" + thinkingContent + "\n#+END_NOTE");
                    }
                    // await logseq.Editor.insertBlock(askPdfBlock.uuid, answerText);
                    let lastParentBlock = askPdfBlock;
                    for (const line of lines) {
                        if (line.trim() === "") continue;
                        const listMatch = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)/);
                        if (listMatch) {
                            // list item → insert as child of the last non-list parent block
                            await logseq.Editor.insertBlock(lastParentBlock.uuid, listMatch[1]);
                        } else {
                            // normal text → insert as child of askPdfBlock, track as new parent
                            const inserted = await logseq.Editor.insertBlock(askPdfBlock.uuid, line);
                            if (inserted) lastParentBlock = inserted;
                        }
                    }
                }
            } else {
                await logseq.UI.showMsg(`Please retry`, "error");
            }
        },
    )
}

logseq.ready(main).catch(console.error)