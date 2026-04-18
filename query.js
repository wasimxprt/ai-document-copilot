import * as dotenv from "dotenv";
dotenv.config();

import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({});
const History = [];

async function chatting(question) {
    console.log("You asked: ", question);

    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'gemini-embedding-001',
    });

    const queryVector = await embeddings.embedQuery(question);

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    const searchResults = await pineconeIndex.query({
        topK: 10,
        vector: queryVector,
        includeMetadata: true,
    });


    const context = searchResults.matches
        .map(match => match.metadata.text)
        .join("\n\n---\n\n");

    History.push({
        role: 'user',
        parts: [{ text: question }]
    })


    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: History,
        config: {
            systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    If the answer is not in the context, you must say "I could not find the answer in the provided document."
    Keep your answers clear, concise, and educational.
      
      Context: ${context}
      `,
        },
    });


    History.push({
        role: 'model',
        parts: [{ text: response.text }]
    })

    console.log("\n");
    console.log(response.text);
}

async function main() {
    const question = readlineSync.question("Ask me anything--> ");
    await chatting(question);
    main();
}


main();