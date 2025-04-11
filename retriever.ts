import dotenv from 'dotenv';
dotenv.config();

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

async function createRetriever() {
  const urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
  ];

  // 1. Load the documents from URLs
  const docs = await Promise.all(
    urls.map((url) => new CheerioWebBaseLoader(url).load()),
  );
  const docsList = docs.flat();

  // 2. Split the documents into smaller chunks
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const docSplits = await textSplitter.splitDocuments(docsList);

  // 3. Creating the vector store
  const vectorStore = await MemoryVectorStore.fromDocuments(
    docSplits,
    new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      modelName: "embedding-001" //Google embedding model
    }),
  );

  return vectorStore.asRetriever();
}

export const retriever = await createRetriever();
