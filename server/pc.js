import { Pinecone } from "@pinecone-database/pinecone";
import { readFileSync, writeFileSync } from 'fs';
// import records from './chunks_output.js';
import dotenv from 'dotenv';
import { pipeline } from "@huggingface/transformers";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAI } from "openai";
dotenv.config();

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

const indexName = 'big-dense-index';
const indexHost = 'https://big-dense-index-ctw204e.svc.aped-4627-b74a.pinecone.io';
// async function createPineconeIndex(){
//   try{
//     await pc.createIndex({
//       name: indexName,
//       dimension: 384,
//       metric: 'cosine',
//       spec: {
//         serverless:{
//            cloud: 'aws',
//            region: 'us-east-1'
//         }
//       }
//     });
//     console.log(`Index ${indexName} created successfully`);
//   }catch(error){
//     console.error('Error creating index:', error);
//     throw error;
//   }
// }

const generateEmbeddings = await pipeline(
  'feature-extraction',
  'Xenova/all-MiniLM-L6-v2',
  {device: 'cpu'}
);

// const splitter = new RecursiveCharacterTextSplitter({
//   chunkSize: 1000,
//   chunkOverlap: 200,
// });

// async function processTextFile(){
//   const text = readFileSync('theHolyGrail.txt', 'utf8');
//   const chunks = await splitter.splitText(text);
//   const vectors = [];

//   for(let i = 0;i<chunks.length;i++){
//     const output = await generateEmbeddings(chunks[i],{
//       pooling: 'mean',
//       normalize: true,
//     })

//     vectors.push({
//       id: `rec${i+1}`,
//       values: Array.from(output.data),
//       metadata: {
//         text: chunks[i],
//         source: 'theHolyGrail.txt'
//       }
//     });
//   }
//   return vectors;
// }

// function chunks(array, batchSize = 100){
//   const batches = [];
//   for(let i = 0;i<array.length;i+=batchSize){
//     batches.push(array.slice(i,i+batchSize));
//   }
//   return batches;
// }

// async function main(){
//   await createPineconeIndex();
//   const index = pc.Index(indexName);

//   const vectors = await processTextFile();
//   const batchSize = 100;
//   const vectorBatches = chunks(vectors,batchSize);

//   for(const batch of vectorBatches){
//     try{
//       await index.upsert(batch);
//       console.log(`Upserted ${batch.length} vectors`);
//     }catch(error){
//       console.error('Error upserting vectors:', error);
//       throw error;
//     }
//   }
// }

// main().catch(console.error);

const index = pc.index(indexName,indexHost);

async function handleQuery(query){
  const queryEmbedding = await generateEmbeddings(query,{
    pooling: 'mean',
    normalize: true,
  })

  const results = await index.namespace("").query({
    vector: Array.from(queryEmbedding.data),
    topK: 3,
    includeValues: true,
    includeMetadata: true,
  })

  return results;
}

class Agent{
  constructor(llm){
    this.llm = llm;
  }

  async answerQuestion(query){
    const matches = await handleQuery(query);
    const contextText = matches.matches.map(match => match.metadata.text).join("\n\n");

    const response = await this.llm.chat.completions.create({
      model: "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
      messages: [{
        role: "system",
        content: "You are an expert psychology consultant with extensive knowledge in clinical psychology, therapeutic approaches, and mental health support. You have context of a book.For any questions user asks: 1.Analyze the context and answer the question based on the context. 2. If the context doesn't have the answer then say 'I don't know'. 3. Give a detailed answer that satisfies the user's question."
    }, {
        role: "user",
        content: `Context:\n${contextText}\n\nQuestion: ${query}`
      }], 
    });

    return response.choices[0].message.content;
  }
}

const llm = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY,
});

export {Agent};



































// const chunkText = (text,chunkSize = 1000, overlapSize = 200) => {
//     const chunks = [];
//     let startPos = 0;
//     while(startPos < text.length){
//         const endPos = Math.min(startPos + chunkSize, text.length);
//         const chunk = text.substring(startPos, endPos);
//         chunks.push(chunk);

//         startPos += chunkSize - overlapSize;

//         if(startPos >= text.length - overlapSize){
//             break;
//         }
//     }
//     return chunks;
// };

// const processFile = (filePath, chunkSize = 1000, overlapSize = 200) => {
//     try{
//         const text = readFileSync(filePath, 'utf8');

//         const chunks = chunkText(text, chunkSize, overlapSize);

//         const records = chunks.map((chunk,index)=> ({
//             _id: `rec${index+1}`,
//             chunk_text: chunk,
//             category: 'psychology'
//         }));

//         const outputFile = 'chunks_output.js';
//         writeFileSync(
//             outputFile,
//             `const records = ${JSON.stringify(records,null,2)};\n\nexport default records;`
//         );
//         console.log(`Processed ${chunks.length} chunks and saved to ${outputFile}`);
//         return records;
//     }catch(error){
//         console.error('Error processing file:', error);
//         return null;
//     }
// };

// const filePath = 'theHolyGrail.txt';
// const records = processFile(filePath,1000,200);
