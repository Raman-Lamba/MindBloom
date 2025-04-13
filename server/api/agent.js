import OpenAI from "openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { pipeline } from "@huggingface/transformers";
import dotenv from "dotenv";
import { readFile } from "fs/promises";
dotenv.config();

// const {PDFExtract, PDFExtractOptions} = pkg;
// const pdfExtract = new PDFExtract();
// const options = {}; /* see below */
// const pdfPath = './documents/psch1.pdf';
// const outputPath = './extracted-text.txt';

// async function extractTextFromPDF(pdfPath) {
//     try {
//         const data = await pdfExtract.extract(pdfPath, options);
//         // return data.pages[23].content[56].str;
//         const fullText = data.pages.map(
//             page => page.content.map(
//                 item => item.str
//             ).join(' ')
//         ).join('\n');
//         return fullText;
//     } catch (err) {
//         console.error('Error extracting text from PDF:', err);
//         return null;
//     }
// }
// const text = await extractTextFromPDF(pdfPath);

// await fs.writeFile(outputPath, text, 'utf-8');

// // Prepare data
const outputPath = './theHolyGrail.txt';

async function loadChat() {
    try{
        const chat = await readFile(outputPath,"utf-8");
        // console.log("Chat loaded", chat.slice(0,1000));
        return chat;
    }catch (err){
        console.error("Error reading file:",err);
        return "";
    }
}

const chat = await loadChat();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 300,
});
const chunks = await textSplitter.splitText(chat);
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { device: 'cpu' });

// function batchArray(array, batchSize) {
//     const batches = [];
//     for (let i = 0; i < array.length; i += batchSize) {
//         batches.push(array.slice(i, i + batchSize));
//     }
//     return batches;
// }
// const batchedChunks = batchArray(chunks, 16); // Adjust batch size based on memory
// Clear existing embeddings
// Clear existing embeddings
// Replace the entire embedding processing loop with this:

let embeddings = [];

for (const chunk of chunks) {
    try {
        const result = await extractor(chunk, { 
            pooling: "mean", 
            normalize: true 
        });

        // Convert tensor to array properly
        let embeddingVector;
        if (result && result.data) {
            // Handle tensor format
            embeddingVector = Array.from(result.data);
        } else if (Array.isArray(result)) {
            // Direct array case
            embeddingVector = result;
        } else {
            console.error("Unexpected embedding format:", result);
            continue;
        }

        // Validate embedding dimensions
        if (embeddingVector.length === 384) { // MiniLM-L6-v2 has 384-dim embeddings
            embeddings.push(embeddingVector);
        } else {
            console.warn(`Invalid embedding dimension: ${embeddingVector.length}`);
        }

    } catch (error) {
        console.error("Error processing chunk:", error);
    }
}
const processedEmbeddings = embeddings.filter(e => e && e.length); // Only keep valid embeddings
// Function to calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) {
        throw new Error("Vectors must be the same length for cosine similarity.");
    }

    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vecA.length; i++) {
        const a = vecA[i];
        const b = vecB[i];

        dotProduct += a * b;
        magnitudeA += a * a;
        magnitudeB += b * b;
    }

    const denominator = Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB);
    return denominator === 0 ? 0 : dotProduct / denominator;
}


class RetrievalTool {
    constructor(embeddings, chunks) {
        this.embeddings = embeddings;
        this.chunks = chunks;
    }

    flattenEmbedding(embedding) {
        if (embedding && embedding.data) {
            return Array.from(embedding.data);
        }
        // Handle array format
        if (Array.isArray(embedding)) {
            return embedding;
        }
        // Fallback
        return Array.from(embedding);
    }

    async retrieve(query, k = 5) {
        const queryEmbedding = await extractor([query], { pooling: "mean", normalize: true });
        const queryVec = this.flattenEmbedding(queryEmbedding[0]);
        
        console.log(`Query vector length: ${queryVec.length}`);
        console.log(`First embedding vector length: ${this.flattenEmbedding(this.embeddings[0]).length}`);
    

        const similarities = this.embeddings.map((embedding, index) => {
            const embeddingVec = this.flattenEmbedding(embedding);
            if (queryVec.length !== embeddingVec.length) {
                console.warn(`Vector mismatch at index ${index}: query ${queryVec.length}, embedding ${embeddingVec.length}`);
            }
            return {
                index,
                text: this.chunks[index],
                similarity: cosineSimilarity(queryVec, embeddingVec)
            };
        });
    
        return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, k);
    }
}

class Agent {
    constructor(llm, retrievalTool) {
        this.llm = llm;
        this.retrievalTool = retrievalTool;
    }

    async answerQuestion(query, k = 5) {
        try {
            const relevantChunks = await this.retrievalTool.retrieve(query, k);
            
            console.log("\n=== Retrieved Information ===");
            relevantChunks.forEach((result, i) => {
                console.log(`${i+1}. Similarity: ${result.similarity.toFixed(4)}`);
                console.log(`   Text: "${result.text}"`);
            });

            const contextText = relevantChunks.map(result => result.text).join("\n\n");
            const response = await this.llm.chat.completions.create({
                model: "mistralai/mistral-small-3.1-24b-instruct:free",
                messages: [{
                    role: "system",
                    content: "You are an expert psychology consultant with extensive knowledge in clinical psychology, therapeutic approaches, and mental health support. You have context of a book.For any questions user asks: 1.Analyze the context and answer the question based on the context. 2. If the context doesn't have the answer then say 'I don't know'. 3. Give a detailed answer that satisfies the user's question."
                }, {
                    role: "user",
                    content: `Context:\n${contextText}\n\nQuestion: ${query}`
                }],
            });

            return response.choices[0].message.content;
        } catch (error) {
            console.error("Agent error:", error);
            return "Sorry, I encountered an error while processing your request.";
        }
    }
}

// Initialize components
const llm = new OpenAI({
    baseURL: 'https://openrouter.ai/api/v1',
    apiKey: process.env.OPENROUTER_API_KEY,
});
const retrievalTool = new RetrievalTool(processedEmbeddings, chunks);
const agent = new Agent(llm, retrievalTool);

// Query handler with formatted output
async function handleQuery(query) {
    console.log(`\n=== Processing Query: "${query}" ===`);
    const start = Date.now();
    
    const answer = await agent.answerQuestion(query);
    
    console.log("\n=== Final Answer ===");
    console.log(answer);
    console.log(`\nProcessed in ${(Date.now() - start)/1000} seconds`);
}

// Execute sample query
handleQuery("What is stress?");
