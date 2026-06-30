import os
import openai
import chromadb
from sentence_transformers import SentenceTransformer
from retrieve_rerank import retrieve_and_rerank, format_context

# Support agent to answer user queries
class SupportAgent:
    def __init__(self, collection_name: str="vela_support"):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key= os.getenv("OPENROUTER_API_KEY")
        )

        # Initialize the embedding model and the ChromaDB collection which will be used for retrieval and rerank
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = chromadb.PersistentClient(path="./chroma_db").get_collection(collection_name)

        # Initialize the history of the current conversation since the model's memory
        # will refresh every new conversation
        self.history = []

        # Defining the tools that the agent can use to answer user queries
        self.tools = [
            {
                "name" : "search_knowledge_base", # Tool to search Vela Support knowledge base
                "description" : (
                    "Search the Vela support database for relevant information about products, pricing, "
                    "and troubleshooting. Use this whenever a customer asks a question that is specific to Vela's"
                    "features, plans, setup or commodities. Aim to provide the most relevant information from"
                    "the knowledge base to the customer."
                ),
                "input_schema" : {
                    "type" : "object",
                    "properties" : {
                        "query" : {
                            "type" : "string",
                            "description" : (
                                "The search query. Phrase this as a natural question on a topic using"
                                "single, plain and easily-understandable english."
                            )
                        }
                    },
                    "required" : ["query"]
                }
            },
            {
                # Web search will be inactive for now
                "name" : "web_search",
                "description" : (
                    "Search the web for current information about Vela's products, pricing, and troubleshooting"
                    "outside Vela's knowledge base. Use this for general questions irrelevant to Vela's services"
                    "or when needing real-time information outside Vela's services."
                ),

                "input_schema" : {
                    "type" : "object",
                    "properties" : {
                        "query" : {
                            "type" : "string",
                            "description" : (
                                "The web search query . Phrase this as a natural question on a topic using"
                                "single, plain and easily-understandable english to get the clearest and most"
                                "direct results from the World Wide Web."
                            )
                        }
                    },
                    "required" : ["query"]
                }
            }
        ]
    
    # Handles which tool is to be called
    def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "search_knowledge_base":
            query = tool_input["query"]
            print(f"\n[Agent searching knowledge base for: {query}]")
            top_chunks = retrieve_and_rerank(query, self.collection, self.embedding_model)
            context = format_context(top_chunks)

            result = f"Knowledge base search results for {query}:\n\n{context}"
            return result

        elif tool_name == "web_search":
            print(f"\n[Agent web search not available for this model]")

            result = (
                "Web search is not available in this configuration. "
                "Please rephrase your question to something the Vela knowledge base can answer, "
                "or ask a general question I can answer from my training data."
            )

            return result
    
    # Creates a response using the LLM model and available tools
    def create_response(self):
        msg_with_sys = [
            {
                "role" : "system",
                "content" : (
                    "You are a helpful customer support agent for Vela, a cloud-based inventory and order management"
                    "platform. Analyse the customer query and determine whether to call a tool or use your general,"
                    "trained knowledge base. Answer customer questions about Vela's features, pricing, billing, and"
                    "troubleshooting. You may use the search_knowledge_base tool to find in the support documentation."
                    "Be concise, friendly, supportive and unambiguous, using simple language and avoiding jargon."
                    "Remember to cite your source. And if a question is outside Vela's scope, be honest about it and"
                    "do not create fake information."
                )
            }
        ] + self.history

        response = self.client.chat.completions.create(
            model="qwen/qwen3-next-80b-a3b-instruct:free",
            max_tokens=2000,
            tools=self.tools,
            tool_choice="auto",
            messages=msg_with_sys
        )

        return response
    
    # To reset history the chat history when the user refreshes the page
    def reset_history(self):
        self.history = []
    
    # The interface that allows the user to chat with the chatbot
    def chat(self, user_message: str) -> str:
        self.history.append({
            "role": "user",
            "content": user_message
        })

        response = self.create_response()
        print(f"[DEBUG] Initial response finish_reason: {response.choices[0].finish_reason}")

        while response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls

            if not tool_calls:
                print("[DEBUG] No tool calls found, breaking loop")
                break
            
            tool_call = tool_calls[0]
            tool_name = tool_call.function.name
            tool_input = eval(tool_call.function.arguments)
            tool_call_id = tool_call.id

            print(f"[DEBUG] Calling tool: {tool_name}")
            tool_result = self._handle_tool_call(tool_name, tool_input)

            self.history.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_call.function.arguments
                    }
                }]
            })

            self.history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result
            })

            response = self.create_response()
            print(f"[DEBUG] After tool call, finish_reason: {response.choices[0].finish_reason}")
            print(f"[DEBUG] Message content: {response.choices[0].message.content}")

        final_answer = response.choices[0].message.content or ""
        print(f"[DEBUG] Final answer: {final_answer}")
        
        self.history.append({
            "role": "assistant",
            "content": final_answer
        })

        return final_answer