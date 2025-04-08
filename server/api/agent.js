import OpenAI from "openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { pipeline } from "@huggingface/transformers";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import crypto from "crypto";
import pkg from 'pdf.js-extract';
import fs from 'fs/promises';
import { readFile } from "fs/promises";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, '../.env') });

// Configuration constants
const EMBEDDING_CACHE_PATH = path.resolve(__dirname, '../caches/embedding-cache.json');
const CHUNK_CACHE_PATH = path.resolve(__dirname, '../caches/chunk-cache.json');
const TEXT_CACHE_PATH = path.resolve(__dirname, '../caches/extracted-text.txt');
const MAX_HISTORY_LENGTH = 7;  // Keeps last 3 exchanges + current

// System prompt template
const systemPrompt = `You are an expert psychology consultant with extensive knowledge in clinical psychology, 
therapeutic approaches, and mental health support. You have access to a comprehensive psychology textbook, whose most matching content you are getting. 
For any questions: 1. Analyze relevant textbook sections 2. Provide actionable guidance 
3. Acknowledge limitations 4. Maintain contextual awareness of previous questions. Respond with academic rigor 
and practical compassion. 5. On a side note try to make it fast, efficient and relatable.`;

async function getTextHash(text) {
    return crypto.createHash('sha256').update(text).digest('hex');
}

async function loadOrGenerateEmbeddings(text) {
    try {
        const [cache, existingChunks] = await Promise.all([
            fs.readFile(EMBEDDING_CACHE_PATH, 'utf-8').then(JSON.parse).catch(() => null),
            fs.readFile(CHUNK_CACHE_PATH, 'utf-8').then(JSON.parse).catch(() => null)
        ]);

        const currentHash = await getTextHash(text);
        
        if (cache?.hash === currentHash && existingChunks?.length) {
            console.log('Using cached embeddings and chunks');
            return { embeddings: cache.embeddings, chunks: existingChunks };
        }
    } catch (error) {
        console.log('Cache invalid, regenerating...');
    }

    // Generate fresh embeddings if no valid cache
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 300,
    });
    const chunks = await textSplitter.splitText(text);
    const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { device: 'cpu' });

    const embeddings = [];
    for (const chunk of chunks) {
        const result = await extractor(chunk, { pooling: "mean", normalize: true });
        embeddings.push(Array.from(result.data));
    }

    const processedEmbeddings = embeddings.filter(e => e.length === 384);
    
    // Save to cache
    await Promise.all([
        fs.writeFile(EMBEDDING_CACHE_PATH, JSON.stringify({
            hash: await getTextHash(text),
            embeddings: processedEmbeddings
        })),
        fs.writeFile(CHUNK_CACHE_PATH, JSON.stringify(chunks)),
        fs.writeFile(TEXT_CACHE_PATH, text)
    ]);

    return { embeddings: processedEmbeddings, chunks };
}

// Cosine similarity function
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0, magnitudeA = 0, magnitudeB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magnitudeA += vecA[i] ** 2;
        magnitudeB += vecB[i] ** 2;
    }
    return dotProduct / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB) || 1e-9);
}

class RetrievalTool {
    constructor(embeddings, chunks) {
        this.embeddings = embeddings;
        this.chunks = chunks;
        this.extractor = null;
    }

    async initializeExtractor() {
        if (!this.extractor) {
            this.extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { device: 'cpu' });
        }
    }

    async retrieve(query, k = 5) {
        await this.initializeExtractor();
        const [queryEmbedding] = await this.extractor([query], { pooling: "mean", normalize: true });
        const queryVec = Array.from(queryEmbedding.data);

        const similarities = this.embeddings.map((embedding, index) => ({
            index,
            text: this.chunks[index],
            similarity: cosineSimilarity(queryVec, embedding)
        }));

        return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, k);
    }
}

class Agent {
    constructor(llm, retrievalTool) {
        this.llm = llm;
        this.retrievalTool = retrievalTool;
        this.conversationHistory = [];
    }

    trimHistory() {
        if (this.conversationHistory.length > MAX_HISTORY_LENGTH) {
            this.conversationHistory = this.conversationHistory.slice(-MAX_HISTORY_LENGTH);
        }
    }

    async answerQuestion(query, k = 5) {
        try {
            this.conversationHistory.push({ role: 'user', content: query });
            this.trimHistory();

            const relevantChunks = await this.retrievalTool.retrieve(query, k);
            const contextText = relevantChunks.map(result => result.text).join("\n\n");

            const messages = [
                {
                    role: "system",
                    content: `${systemPrompt}\n\nConversation Context: ${this.conversationHistory
                        .slice(0, -1)
                        .map(m => `${m.role}: ${m.content}`)
                        .join('\n')}`
                },
                ...this.conversationHistory.slice(-3),
                {
                    role: "user",
                    content: `Document Context:\n${contextText}\n\nCurrent Question: ${query}`
                }
            ];

            const response = await this.llm.chat.completions.create({
                model: "mistralai/mistral-small-3.1-24b-instruct:free",
                messages,
            });

            const answer = response.choices[0].message.content;
            this.conversationHistory.push({ role: 'assistant', content: answer });
            this.trimHistory();

            return answer;
        } catch (error) {
            console.error("Agent error:", error);
            return "Sorry, I encountered an error while processing your request.";
        }
    }
}

// Initialization
// async function initializeAgent() {
//     // Load source text
//     const chat = await readFile(TEXT_CACHE_PATH, "utf-8").catch(() => "");
    
//     // Load or generate embeddings
//     const { embeddings, chunks } = await loadOrGenerateEmbeddings(chat);

//     // Initialize components
//     const llm = new OpenAI({
//         baseURL: 'https://openrouter.ai/api/v1',
//         apiKey: process.env.OPENROUTER_API_KEY,
//     });
//     const retrievalTool = new RetrievalTool(embeddings, chunks);
//     return new Agent(llm, retrievalTool);
// }

export async function initializeAgent() {
    try {
        // Load source text
        const text = await readFile(TEXT_CACHE_PATH, "utf-8").catch(async () => {
            console.log("No text cache found, creating empty file");
            await fs.mkdir(path.dirname(TEXT_CACHE_PATH), { recursive: true });
            // You should replace this with your actual document text source
            const defaultText = "This is a placeholder for the psychology textbook content.";
            await fs.writeFile(TEXT_CACHE_PATH, defaultText);
            return defaultText;
        });
        
        // Load or generate embeddings
        const { embeddings, chunks } = await loadOrGenerateEmbeddings(text);

        // Initialize components
        const llm = new OpenAI({
            baseURL: 'https://openrouter.ai/api/v1',
            apiKey: process.env.OPENROUTER_API_KEY,
        });
        const retrievalTool = new RetrievalTool(embeddings, chunks);
        return new Agent(llm, retrievalTool);
    } catch (error) {
        console.error("Failed to initialize agent:", error);
        throw error;
    }
}

// Query handler
async function handleQuery(agent, query) {
    console.log(`\n=== Processing Query: "${query}" ===`);
    const start = Date.now();
    
    const answer = await agent.answerQuestion(query);
    
    console.log("\n=== Final Answer ===");
    console.log(answer);
    console.log(`Processed in ${(Date.now() - start)/1000}s`);
    return answer;
}

// Execution
// (async () => {
//     const agent = await initializeAgent();
    
//     // Example conversation
//     await handleQuery(agent, "What should I do if I am stressed?");
//     await handleQuery(agent, "How does that apply to work-related stress specifically?");
//     await handleQuery(agent, "Can you suggest some quick relaxation techniques?");
// })();