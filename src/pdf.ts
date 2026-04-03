import { parseEDNString } from "edn-data";
import Highlight from "./types/highlight";
import * as pdfjs from 'pdfjs-dist/legacy/build/pdf.mjs';
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/legacy/build/pdf.worker.min.mjs`;


function normalizePathSeparators(path: string) {
    return path.replace(/\\/g, "/");
}

function stripLeadingSlash(path: string) {
    return path.replace(/^\/+/, "");
}

function normalizeRelativeGraphPath(path: string) {
    const normalizedPath = normalizePathSeparators(path);
    const segments = normalizedPath.split("/");
    const resolvedSegments: string[] = [];

    for (const segment of segments) {
        if (!segment || segment === ".") continue;
        if (segment === "..") {
            // Treat leading ".." as graph-root relative so "../assets/..." still works.
            if (resolvedSegments.length > 0) resolvedSegments.pop();
            continue;
        }
        resolvedSegments.push(segment);
    }

    return resolvedSegments.join("/");
}

function makeFileUrlFromAssetUrl(assetUrl: string) {
    if (!assetUrl.startsWith("assets://")) {
        return assetUrl.replace("assets", "file");
    }

    const assetFilePath = decodeURIComponent(assetUrl.slice("assets://".length));
    const normalizedAssetFilePath = normalizePathSeparators(assetFilePath);
    const pathname = normalizedAssetFilePath.startsWith("/")
        ? normalizedAssetFilePath
        : `/${normalizedAssetFilePath}`;

    return encodeURI(`file://${pathname}`);
}

async function resolveGraphRelativePdfPath(pdfPath: string) {
    const trimmedPath = pdfPath.trim();
    const normalizedPath = normalizePathSeparators(trimmedPath);
    const currentGraph = await logseq.App.getCurrentGraph();
    if (!currentGraph) return null;
    const graphPath = normalizePathSeparators(currentGraph.path);
    const normalizedGraphPath = graphPath.endsWith("/") ? graphPath : `${graphPath}/`;

    if (/^[a-zA-Z]:\//.test(normalizedPath)) {
        if (!normalizedPath.startsWith(normalizedGraphPath)) return null;
        return normalizedPath.slice(normalizedGraphPath.length);
    }

    if (normalizedPath.startsWith("file:///")) {
        const decodedPath = decodeURIComponent(normalizedPath.slice("file:///".length));
        const normalizedDecodedPath = normalizePathSeparators(decodedPath);
        if (!normalizedDecodedPath.startsWith(normalizedGraphPath)) return null;
        return normalizedDecodedPath.slice(normalizedGraphPath.length);
    }

    return normalizeRelativeGraphPath(stripLeadingSlash(normalizedPath));
}

export type PdfDebugInfo = {
    inputPath: string;
    graphRelativePdfPath: string | null;
    assetPath: string | null;
    filePath: string | null;
    ednPath: string | null;
    error: string | null;
};

export type PdfLookupResult =
    | { ok: true; pdf: Blob; edn: { "highlights": Highlight[] }; debug: PdfDebugInfo }
    | { ok: false; debug: PdfDebugInfo };

export function formatPdfDebugInfo(debug: PdfDebugInfo) {
    return [
        `inputPath: ${debug.inputPath}`,
        `graphRelativePdfPath: ${debug.graphRelativePdfPath ?? "null"}`,
        `assetPath: ${debug.assetPath ?? "null"}`,
        `filePath: ${debug.filePath ?? "null"}`,
        `ednPath: ${debug.ednPath ?? "null"}`,
        `error: ${debug.error ?? "null"}`,
    ].join("\n");
}

export async function getPdfAndEdnByPdfPath(pdfPath: string): Promise<PdfLookupResult> {
    const debug: PdfDebugInfo = {
        inputPath: pdfPath,
        graphRelativePdfPath: null,
        assetPath: null,
        filePath: null,
        ednPath: null,
        error: null,
    };

    const graphRelativePdfPath = await resolveGraphRelativePdfPath(pdfPath);
    debug.graphRelativePdfPath = graphRelativePdfPath;
    if (!graphRelativePdfPath) {
        debug.error = "Path is outside the current graph or resolves above graph root.";
        return { ok: false, debug };
    }

    const assetPath = await logseq.Assets.makeUrl(graphRelativePdfPath);
    debug.assetPath = assetPath;
    const filePath = makeFileUrlFromAssetUrl(assetPath);
    const ednPath = filePath.replace(/\.pdf$/i, ".edn");
    debug.filePath = filePath;
    debug.ednPath = ednPath;

    let fileResponse;
    let ednResponse;

    try {
        fileResponse = await fetch(filePath);
        ednResponse = await fetch(ednPath);
    } catch (error) {
        debug.error = error instanceof Error ? `Fetch failed: ${error.message}` : "Fetch failed.";
        return { ok: false, debug };
    }

    if (!fileResponse.ok || !ednResponse.ok) {
        debug.error = `Fetch returned non-OK status. pdf=${fileResponse.status}, edn=${ednResponse.status}`;
        return { ok: false, debug };
    }

    const fileArrayBuffer = await fileResponse.arrayBuffer();
    const fileUint8Array = new Uint8Array(fileArrayBuffer);
    const pdf = new Blob([fileUint8Array], { type: "application/pdf" });
    const edn = parseEDNString(
        await ednResponse.text(),
        { mapAs: "object", keywordAs: "string" },
    ) as { "highlights": Highlight[] };

    return { ok: true, pdf, edn, debug };
}

export function findUuidOfCurrentLine(line: string) {
    const regex = /\(\((.*?)\)\)/;
    const match = line.match(regex);
    return match ? match[1] : null;
}

export function findHighlightFromEdnByUuid(uuid: string, edn: { "highlights": Highlight[] }) {
    const highlights = edn["highlights"];
    for (const highlight of highlights) {
        if (highlight["id"]["val"] === uuid) {
            return highlight;
        }
    }
    return null;
}

// 이미지 크기를 조정하는 함수
function resizeCanvas(canvas: HTMLCanvasElement, maxArea: number): HTMLCanvasElement {
    let width = canvas.width;
    let height = canvas.height;
    const aspectRatio = width / height;

    // 이미지 면적이 maxArea를 초과하는 경우 크기 조정
    if (width * height > maxArea) {
        width = Math.sqrt(maxArea * aspectRatio);
        height = width / aspectRatio;
    }

    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = width;
    resizedCanvas.height = height;
    const ctx = resizedCanvas.getContext('2d');
    if (ctx) {
        ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, width, height);
    }

    return resizedCanvas;
}

// Canvas를 base64로 변환하는 함수
function canvasToBase64(canvas: HTMLCanvasElement): string {
    const resizedCanvas = resizeCanvas(canvas, 250000);
    return resizedCanvas.toDataURL('image/jpeg');
}

export async function captureImageFromPDF(pdfBlob: Blob, position: Highlight['position']) {
    const pdf = await pdfjs.getDocument(await pdfBlob.arrayBuffer()).promise;
    const page = await pdf.getPage(position.page);

    // 캡처 시점의 PDF 크기
    const captureWidth = position.bounding.width;
    const captureHeight = position.bounding.height;

    // 렌더링 스케일 (고해상도를 위해)
    const renderScale = 2;

    const viewport = page.getViewport({ scale: renderScale });

    const canvas = document.createElement('canvas');
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const context = canvas.getContext('2d');
    if (!context) return null;

    await page.render({ canvasContext: context, viewport: viewport }).promise;

    const { x1, y1, x2, y2 } = position.bounding;

    // 상대 좌표 계산
    const relativeX1 = x1 / captureWidth;
    const relativeY1 = y1 / captureHeight;
    const relativeX2 = x2 / captureWidth;
    const relativeY2 = y2 / captureHeight;

    // 렌더링된 캔버스에서의 실제 좌표 계산
    const startX = relativeX1 * viewport.width;
    const startY = relativeY1 * viewport.height; // Y축 좌표 수정
    const width = (relativeX2 - relativeX1) * viewport.width;
    const height = (relativeY2 - relativeY1) * viewport.height;

    const extractedCanvas = document.createElement('canvas');
    extractedCanvas.width = width;
    extractedCanvas.height = height;
    const extractedContext = extractedCanvas.getContext('2d');
    if (!extractedContext) return null;

    // 이미지 추출
    extractedContext.drawImage(
        canvas,
        startX, startY, width, height,
        0, 0, width, height
    );

    return canvasToBase64(extractedCanvas);
}
