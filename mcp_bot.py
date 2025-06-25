# Suggested content for your mcp_bot.py
import os
from sentence_transformers import SentenceTransformer
import chromadb
from mcp import MCPModel # Assuming mcp.py and MCPModel class are correctly defined
from urllib.parse import urlparse # For formatting source links
import logging # Optional: for logging within this module

# Setup logging for this module if you want specific logs
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# --- Config (can be centralized or passed as arguments if preferred) ---
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
CHROMA_DB_PATH = './hodl_chroma_db'
COLLECTION_NAME = 'hodl_content'
N_RESULTS_TO_RETRIEVE = 4 # Number of chunks to retrieve for context

# --- Initialize embedding model and vector DB (globally for this module) ---
print("MCP_Bot: Initializing embedding model...")
embedding_model_mcp = SentenceTransformer(EMBEDDING_MODEL_NAME) # Renamed to avoid clash if main_bot also has one
print("MCP_Bot: Embedding model initialized.")

print(f"MCP_Bot: Connecting to ChromaDB at: {CHROMA_DB_PATH}")
chroma_client_mcp = chromadb.PersistentClient(path=CHROMA_DB_PATH) # Renamed
try:
    collection_mcp = chroma_client_mcp.get_collection(name=COLLECTION_NAME) # Renamed
    print(f"MCP_Bot: Successfully connected to ChromaDB collection: '{COLLECTION_NAME}' with {collection_mcp.count()} items.")
except Exception as e:
    print(f"MCP_Bot: Error connecting to ChromaDB collection: {e}. Please ensure it's populated.")
    collection_mcp = None # Handle gracefully if DB isn't ready

# --- Initialize MCP model (globally for this module) ---
HUGGINGFACE_TOKEN_MCP = os.getenv("HUGGINGFACE_TOKEN") # Ensure .env is loaded by the main script
print("MCP_Bot: Initializing MCPModel (LLM)...")
mcp_instance = MCPModel(hf_token=HUGGINGFACE_TOKEN_MCP) # Using default model_name and use_4bit=True from mcp.py
print("MCP_Bot: MCPModel initialized.")


def search_relevant_chunks(query: str):
    """Retrieve relevant chunks and their metadatas from ChromaDB."""
    if not collection_mcp:
        print("MCP_Bot: ChromaDB collection not available for search.")
        return [], []
    try:
        query_embedding = embedding_model_mcp.encode(query).tolist() # Corrected: encode query directly
        results = collection_mcp.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS_TO_RETRIEVE,
            include=['documents', 'metadatas', 'distances'] # Include metadatas and distances
        )
        if results and results.get('documents') and results['documents'][0]:
            # Log retrieved chunks for debugging
            log_msg = f"MCP_Bot: Retrieved {len(results['documents'][0])} chunks for query '{query}'.\n"
            for i, doc_text in enumerate(results['documents'][0]):
                dist = results['distances'][0][i] if results.get('distances') and results['distances'][0] else 'N/A'
                meta = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                log_msg += f"  Chunk {i+1} (Source: {meta.get('source_url', 'Unknown')}, Distance: {dist:.4f}): {doc_text[:100].replace(chr(10), ' ')}...\n"
            print(log_msg) # Or use logger.info(log_msg)
            return results['documents'][0], results['metadatas'][0], results['distances'][0]
    except Exception as e:
        print(f"MCP_Bot: Error during ChromaDB search: {e}") # Or use logger.error
    return [], [], []

def build_llm_prompt(question: str, context_chunks_texts: list[str], user_query_for_llm_context: str) -> str:
    """Builds the prompt for the LLM, instructing on formatting."""
    context_str = "\n\n---\n\n".join(context_chunks_texts)

    system_instructions = (
        "You are the HODLToken AI Assistant, an expert in HODLToken, providing clear, concise, and accurate information.\n"
        "Your task is to answer the user's question based *ONLY* on the following context from the hodltoken.net website.\n"
        "Synthesize the information from the context into an **easy-to-read answer**. Use **clear paragraphs** (using a double newline between paragraphs). If listing multiple items, steps, or key features, use **bullet points** (e.g., starting lines with '*' or '-'). Aim for a helpful but not overly verbose response.\n\n"
        "Instructions for answering:\n"
        "1. If the context directly and completely answers the question, provide that answer.\n"
        "2. If the context provides partial information, give a focused answer based on what is available. You MUST explicitly state if some aspects of the question cannot be fully addressed (e.g., 'Based on the provided information, [partial answer]. However, specific details about [missing aspect] are not covered in this context.').\n"
        "3. If the context does not contain any relevant information to the user's question, you MUST state: 'I could not find specific information about that in the HODLToken documentation I have access to. You could try rephrasing or asking about a different HODLToken feature.'\n"
        "4. Do NOT add any information that is not explicitly present in the provided context. Do NOT make assumptions or infer beyond what the text clearly states.\n"
        "5. Ensure your entire response is well-formatted for readability."
    )
    
    # For Mistral Instruct
    prompt = f"<s>[INST] {system_instructions}\n\nUser's Question: {user_query_for_llm_context}\n\nProvided Context from hodltoken.net:\n{context_str} [/INST]"
    return prompt

def answer_question_with_mcp_rules(user_question: str, max_context_distance=0.75) -> str:
    """
    Retrieve context, build prompt, generate answer with MCPModel, and append sources.
    Uses a distance threshold to decide if context is relevant.
    """
    if not mcp_instance or not mcp_instance.generator: # Check if LLM is ready
        print("MCP_Bot: LLM (MCPModel) is not initialized properly.")
        return "My apologies, I'm having a bit of trouble with my main thinking process right now. Please try again shortly."

    documents, metadatas, distances = search_relevant_chunks(user_question)
    
    context_is_relevant = False
    min_distance = float('inf')
    if documents and distances:
        min_distance = min(distances)
        if min_distance < max_context_distance:
            context_is_relevant = True

    final_answer = ""
    sources_markdown = ""

    if context_is_relevant:
        print(f"MCP_Bot: Relevant context found (best distance: {min_distance:.4f}). Using RAG.")
        # Reconstruct context texts from documents for the prompt
        context_texts_for_prompt = [doc for doc in documents]
        prompt = build_llm_prompt(user_question, context_texts_for_prompt, user_question)
        
        llm_generated_text = mcp_instance.generate(prompt, max_new_tokens=350) # max_new_tokens from your mcp.py was 256, increased slightly

        final_answer = llm_generated_text # mcp.generate should return only the answer

        # Prepare source links
        if metadatas:
            unique_source_urls = set()
            for meta in metadatas: # Use metadatas corresponding to the retrieved documents
                source_url = meta.get("source_url")
                if source_url:
                    unique_source_urls.add(source_url)
            
            if unique_source_urls:
                sources_markdown = "\n\n---\n**Sources:**\n"
                for url_idx, url in enumerate(list(unique_source_urls)[:min(len(unique_source_urls), N_RESULTS_TO_RETRIEVE)]): 
                    parsed_url_obj = urlparse(url)
                    path_parts = [part for part in parsed_url_obj.path.split('/') if part]
                    link_text = path_parts[-1] if path_parts else parsed_url_obj.netloc
                    if not link_text or link_text == "_": link_text = "HODLToken Source"
                    link_text_safe = link_text.replace("_", "\\_").replace("*", "\\*").replace("[", "\\[").replace("]", "\\]")
                    sources_markdown += f"• [{link_text_safe}]({url})\n"
                final_answer += sources_markdown
    else:
        print(f"MCP_Bot: Context not relevant enough (best distance: {min_distance:.4f}) or no context. Using general prompt.")
        general_system_prompt = (
            "You are a helpful and professional conversational AI assistant representing HODLToken. "
            "Politely answer the user's query. If it's about specific HODLToken features or data you don't have internal knowledge of, state that you can primarily provide information found on the hodltoken.net website and suggest they ask a question related to HODLToken's publicly available information. "
            "Keep responses concise and on-brand. Format your answer clearly, using paragraphs if needed (double newlines between paragraphs)."
        )
        prompt = f"<s>[INST] {general_system_prompt}\n\nUser's message: {user_question} [/INST]"
        final_answer = mcp_instance.generate(prompt, max_new_tokens=150) # Shorter for general chat

    return final_answer if final_answer else "I'm sorry, I couldn't formulate a response for that."

# Example usage (optional, for testing mcp_bot.py directly)
# if __name__ == '__main__':
#     load_dotenv() # Make sure .env is loaded if testing this file directly
#     HUGGINGFACE_TOKEN_MCP = os.getenv("HUGGINGFACE_TOKEN")
#     if not HUGGINGFACE_TOKEN_MCP and mcp_instance.hf_token is None:
#         mcp_instance.hf_token = input("Enter Hugging Face Token for MCP Model: ")
#         mcp_instance._load_model() # Re-try loading if token was missing

#     if collection_mcp and mcp_instance and mcp_instance.generator:
#         test_query = "what are hodl nfts?"
#         print(f"\nTesting mcp_bot.py with query: '{test_query}'")
#         response = answer_question_with_mcp_rules(test_query)
#         print("\nFormatted Response from MCP Bot:")
#         print(response)

#         test_query_general = "how are you today?"
#         print(f"\nTesting mcp_bot.py with general query: '{test_query_general}'")
#         response_general = answer_question_with_mcp_rules(test_query_general)
#         print("\nFormatted Response from MCP Bot (General):")
#         print(response_general)
#     else:
#         print("MCP_Bot: Cannot run test, critical components (DB or LLM) not initialized.")