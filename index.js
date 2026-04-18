import * as dotenv from "dotenv";
dotenv.config();

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";

async function indexDocument() {
    try {
        console.log("Starting indexing...");

        const pdfLoader = new PDFLoader("./java.pdf");
        const rawDocs = await pdfLoader.load();
        console.log("PDF loaded. Pages/docs:", rawDocs.length);

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
        console.log("Chunks created:", chunkedDocs.length);

        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GEMINI_API_KEY,
            model: "gemini-embedding-001",
        });

        const testVector = await embeddings.embedQuery("hello world");
        console.log("Embedding length:", testVector.length);

        const pinecone = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });

        const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

        await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
            pineconeIndex,
            maxConcurrency: 5,
        });

        console.log("Document indexed successfully!");
    } catch (error) {
        console.error("Indexing failed:");
        console.error(error);
    }
}

indexDocument();