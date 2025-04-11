import dotenv from 'dotenv';
dotenv.config();

import { Annotation } from "@langchain/langgraph";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { StateGraph, START, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { createRetrieverTool } from "langchain/tools/retriever";

// Importar o retriever que você já criou
import { retriever } from './retriever.js';

// Definir o estado do grafo
const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  })
});

// Criar a ferramenta de recuperação
const tool = createRetrieverTool(
  retriever,
  {
    name: "retrieve_blog_posts",
    description:
      "Pesquise e retorne informações sobre os posts do blog de Lilian Weng sobre agentes LLM, engenharia de prompts e ataques adversários em LLMs.",
  },
);
const tools = [tool];

// Criar o nó de ferramentas
const toolNode = new ToolNode<typeof GraphState.State>(tools);

// Funções para os nós do grafo

/**
 * Decide se o agente deve recuperar mais informações ou encerrar o processo.
 */
function shouldRetrieve(state: typeof GraphState.State): string {
  const { messages } = state;
  console.log("---DECIDIR SE VAI RECUPERAR---");
  const lastMessage = messages[messages.length - 1];

  if ("tool_calls" in lastMessage && Array.isArray(lastMessage.tool_calls) && lastMessage.tool_calls.length) {
    console.log("---DECISÃO: RECUPERAR---");
    return "retrieve";
  }

  console.log("---DECISÃO: NÃO RECUPERAR---");
  // Se não houver chamadas de ferramentas, terminamos.
  return END;
}

/**
 * Avalia a relevância dos documentos recuperados.
 */
async function gradeDocuments(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---OBTER RELEVÂNCIA---");

  const { messages } = state;
  const tool = {
    name: "give_relevance_score",
    description: "Dê uma pontuação de relevância aos documentos recuperados.",
    schema: z.object({
      binaryScore: z.string().describe("Pontuação de relevância 'sim' ou 'não'"),
    })
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    `Você é um avaliador que analisa a relevância dos documentos recuperados para uma pergunta do usuário.
  Aqui estão os documentos recuperados:
  \n ------- \n
  {context} 
  \n ------- \n
  Aqui está a pergunta do usuário: {question}
  Se o conteúdo dos documentos for relevante para a pergunta do usuário, classifique-os como relevantes.
  Dê uma pontuação binária 'sim' ou 'não' para indicar se os documentos são relevantes para a pergunta.
  Sim: Os documentos são relevantes para a pergunta.
  Não: Os documentos não são relevantes para a pergunta.`,
  );

  const model = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-pro",
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY,
  }).bindTools([tool], {
    tool_choice: tool.name,
  });

  const chain = prompt.pipe(model);

  const lastMessage = messages[messages.length - 1];

  const score = await chain.invoke({
    question: messages[0].content as string,
    context: lastMessage.content as string,
  });

  return {
    messages: [score]
  };
}

/**
 * Verifica a relevância da chamada de ferramenta LLM anterior.
 */
function checkRelevance(state: typeof GraphState.State): string {
  console.log("---VERIFICAR RELEVÂNCIA---");

  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  if (!("tool_calls" in lastMessage)) {
    throw new Error("O nó 'checkRelevance' requer que a mensagem mais recente contenha chamadas de ferramentas.")
  }
  const toolCalls = (lastMessage as AIMessage).tool_calls;
  if (!toolCalls || !toolCalls.length) {
    throw new Error("A última mensagem não era uma mensagem de função");
  }

  if (toolCalls[0].args.binaryScore === "sim") {
    console.log("---DECISÃO: DOCUMENTOS RELEVANTES---");
    return "yes";
  }
  console.log("---DECISÃO: DOCUMENTOS NÃO RELEVANTES---");
  return "no";
}

/**
 * Invoca o modelo de agente para gerar uma resposta com base no estado atual.
 */
async function agent(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---CHAMAR AGENTE---");

  const { messages } = state;
  // Encontre a AIMessage que contém a chamada de ferramenta `give_relevance_score`,
  // e remova-a se existir. Isso ocorre porque o agente não precisa saber
  // a pontuação de relevância.
  const filteredMessages = messages.filter((message) => {
    if ("tool_calls" in message && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      return message.tool_calls[0].name !== "give_relevance_score";
    }
    return true;
  });

  const model = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-pro",
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY,
    streaming: true,
  }).bindTools(tools);

  const response = await model.invoke(filteredMessages);
  return {
    messages: [response],
  };
}

/**
 * Transforma a consulta para produzir uma pergunta melhor.
 */
async function rewrite(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---TRANSFORMAR CONSULTA---");

  const { messages } = state;
  const question = messages[0].content as string;
  const prompt = ChatPromptTemplate.fromTemplate(
    `Observe a entrada e tente raciocinar sobre a intenção/significado semântico subjacente. \n 
Aqui está a pergunta inicial:
\n ------- \n
{question} 
\n ------- \n
Formule uma pergunta aprimorada:`,
  );

  const model = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-pro",
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY,
    streaming: true,
  });
  const response = await prompt.pipe(model).invoke({ question });
  return {
    messages: [response],
  };
}

/**
 * Gera resposta
 */
async function generate(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---GERAR---");

  const { messages } = state;
  const question = messages[0].content as string;
  // Extrair a ToolMessage mais recente
  const lastToolMessage = messages.slice().reverse().find((msg) => msg._getType() === "tool");
  if (!lastToolMessage) {
    throw new Error("Nenhuma mensagem de ferramenta encontrada no histórico da conversa");
  }

  const docs = lastToolMessage.content as string;

  const prompt = ChatPromptTemplate.fromTemplate(
    `Você é um assistente útil que responde a perguntas com base no contexto fornecido.
    
    Contexto:
    {context}
    
    Pergunta: {question}
    
    Responda à pergunta com base apenas no contexto fornecido. Se o contexto não contiver informações suficientes para responder à pergunta, diga que não tem informações suficientes.`
  );

  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-pro",
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY,
    streaming: true,
  });

  const ragChain = prompt.pipe(llm);

  const response = await ragChain.invoke({
    context: docs,
    question,
  });

  return {
    messages: [response],
  };
}

// Definir o grafo
const workflow = new StateGraph(GraphState)
  .addNode("agent", agent)
  .addNode("retrieve", toolNode)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate);

// Chamar o nó do agente para decidir recuperar ou não
workflow.addEdge(START, "agent");

// Decidir se deve recuperar
workflow.addConditionalEdges(
  "agent",
  // Avaliar decisão do agente
  shouldRetrieve,
);

workflow.addEdge("retrieve", "gradeDocuments");

// Arestas tomadas após a chamada do nó `action`.
workflow.addConditionalEdges(
  "gradeDocuments",
  // Avaliar decisão do agente
  checkRelevance,
  {
    // Chamar nó de ferramenta
    yes: "generate",
    no: "rewrite", // espaço reservado
  },
);

workflow.addEdge("generate", END);
workflow.addEdge("rewrite", "agent");

// Compilar
const app = workflow.compile();

// Executar o grafo
async function main() {
  const inputs = {
    messages: [
      new HumanMessage(
        "Quais são os tipos de memória de agente com base no post do blog de Lilian Weng?",
      ),
    ],
  };
  
  let finalState;
  for await (const output of await app.stream(inputs)) {
    for (const [key, value] of Object.entries(output)) {
      const lastMsg = output[key].messages[output[key].messages.length - 1];
      console.log(`Saída do nó: '${key}'`);
      console.dir({
        type: lastMsg._getType(),
        content: lastMsg.content,
        tool_calls: lastMsg.tool_calls,
      }, { depth: null });
      console.log("---\n");
      finalState = value;
    }
  }

  console.log(JSON.stringify(finalState, null, 2));
}

main().catch(console.error); 