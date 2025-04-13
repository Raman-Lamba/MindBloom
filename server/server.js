import express from 'express';
import cors from 'cors';
import http from 'http';
import { Server } from 'socket.io';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import bodyParser from 'body-parser';
import { Agent } from './pc.js'; // Import your existing Agent class

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, '../.env') });

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: process.env.NODE_ENV === 'production' ? false : ["http://localhost:3000"],
        methods: ["GET", "POST"],
    },
});

app.use(express.json());
app.use(cors({
    origin: process.env.NODE_ENV === 'production' ? false : ["http://localhost:3000"],
}));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, '../client/build')));
}

// Initialize the agent with Pinecone and model setup
async function initializeAgent() {
    try {
        const { OpenAI } = await import("openai");
        
        // Initialize your LLM configuration
        const llm = new OpenAI({
            baseURL: 'https://openrouter.ai/api/v1',
            apiKey: process.env.OPENROUTER_API_KEY,
        });

        // Create agent instance
        return new Agent(llm);
    } catch (error) {
        console.error('Error initializing agent:', error);
        throw error;
    }
}

let agentPromise = initializeAgent();

// WebSocket handler
io.on("connection", async (socket) => {
    console.log("New client connected:", socket.id);

    const agent = await agentPromise;

    socket.on('message', async (data) => {
        try {
            const response = await agent.answerQuestion(data.message);
            socket.emit('response', { message: response });
        } catch (error) {
            console.error("Error processing message:", error);
            socket.emit('response', { message: "An error occurred while processing your request. Please try again later." });
        }
    });

    socket.on('disconnect', () => {
        console.log("Client disconnected:", socket.id);
    });
});

// REST API handler
app.post('/api/query', async (req, res) => {
    try {
        const { query } = req.body;
        if (!query) {
            return res.status(400).json({ error: "Query is required" });
        }

        const agent = await agentPromise;
        const answer = await agent.answerQuestion(query);
        res.json({ answer });
    } catch (error) {
        console.error("Error processing query:", error);
        res.status(500).json({ error: "An error occurred while processing your request. Please try again later." });
    }
});

// Catch-all handler
app.get('*', (req, res) => {
    if (process.env.NODE_ENV === 'production') {
        res.sendFile(path.resolve(__dirname, '../client/build', 'index.html'));
    } else {
        res.status(404).send('API endpoint not found. In development, frontend is served separately.');
    }
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});