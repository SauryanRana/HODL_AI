import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import re
from urllib.parse import urlparse # Already imported, used for source link text

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN not found in .env file or environment variables.")
    exit()

CHROMA_DB_PATH = "./hodl_chroma_db"
COLLECTION_NAME = "hodl_content"
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
N_RESULTS_TO_RETRIEVE = 4 # Refined: fewer, more focused chunks often better than many
MAX_CONTEXT_RELEVANCE_DISTANCE = 0.75 # Keep as is, tune based on logs

# LLM Configuration
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Or v0.1 if you prefer and have it cached
LOAD_IN_8BIT = False
LOAD_IN_4BIT = True  # Enabling 4-bit for RTX 4080

# --- Initialize models and DB connection ---
# ... (Embedding model and ChromaDB initialization - kept as is from your file) ...
print("Initializing embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model initialized.")

print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Successfully connected to ChromaDB collection '{COLLECTION_NAME}' with {collection.count()} items.")
except Exception as e:
    print(f"Error connecting to ChromaDB collection: {e}")
    exit()

print(f"Initializing LLM: {LLM_MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Attempting to run LLM on: {device.upper()}")

llm_pipeline = None
try:
    tokenizer_kwargs = {}
    model_kwargs = {}

    if HUGGINGFACE_HUB_TOKEN:
        tokenizer_kwargs["token"] = HUGGINGFACE_HUB_TOKEN
        model_kwargs["token"] = HUGGINGFACE_HUB_TOKEN
        print("Using Hugging Face token from .env file for model and tokenizer download.")

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        if LOAD_IN_4BIT:
            print("Loading model in 4-bit for GPU...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # Important for bnb
            print("Note: 4-bit loading requires 'bitsandbytes' and 'accelerate'.")
        elif LOAD_IN_8BIT:
            print("Loading model in 8-bit for GPU...")
            model_kwargs["load_in_8bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
            print("Note: 8-bit loading requires 'bitsandbytes' and 'accelerate'.")
        else:
            print("Loading model in float16 for GPU...")
            model_kwargs["torch_dtype"] = torch.float16
    else:
        print("Loading model on CPU. This will be significantly slower.")

    print(f"Loading tokenizer for {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, **tokenizer_kwargs)
    print("Tokenizer loaded.")

    print(f"Loading model {LLM_MODEL_NAME}...")
    model_for_pipeline = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, **model_kwargs)
    print("Model loaded.")

    print("Initializing text-generation pipeline...")
    pipeline_creation_args = {
        "task": "text-generation",
        "model": model_for_pipeline,
        "tokenizer": tokenizer,
    }
    if "device_map" not in model_kwargs:
         pipeline_creation_args["device"] = 0 if device == "cuda" else (-1 if device == "cpu" else torch.device("mps"))

    llm_pipeline = pipeline(**pipeline_creation_args)
    print(f"LLM '{LLM_MODEL_NAME}' initialized successfully.")

except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Troubleshooting tips:\n1. Check internet connectivity & Hugging Face token/terms.\n2. Ensure 'bitsandbytes' and 'accelerate' are installed for quantization.\n3. Verify CUDA/driver setup if using GPU.\n4. Consider resource limitations (RAM/VRAM).")
    llm_pipeline = None

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

def generate_llm_answer(prompt_text):
    if not llm_pipeline:
        return "My apologies, I'm currently unable to generate a response. Please try again a bit later."
    
    logger.info(f"Sending prompt to LLM (length: {len(prompt_text)} chars). Preview (first 300): {prompt_text[:300]}...")
    try:
        sequences = llm_pipeline(
            prompt_text,
            do_sample=True,
            temperature=0.7,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=llm_pipeline.tokenizer.eos_token_id,
            max_new_tokens=300 # Your setting
        )
        generated_text_full = sequences[0]['generated_text']
        answer = ""
        prompt_end_marker = "[/INST]"
        # Check if the original prompt is part of the output (common)
        # Find the end of the *input* prompt within the *output* string
        # This handles cases where the model might add preamble before echoing the prompt.
        # We search for the last occurrence of [/INST] which signals the end of instructions.
        last_inst_pos_in_output = generated_text_full.rfind(prompt_end_marker)
        if last_inst_pos_in_output != -1:
            answer = generated_text_full[last_inst_pos_in_output + len(prompt_end_marker):].strip()
        else:
            # Fallback if [/INST] is not found in the output (less common for instruct models)
            # or if the prompt structure itself didn't contain it.
            # This assumes the answer is everything after the original prompt length.
            if generated_text_full.startswith(prompt_text):
                 answer = generated_text_full[len(prompt_text):].strip()
            else: # If prompt is not a prefix, take the whole output as answer (could happen if pipeline configured differently)
                 answer = generated_text_full.strip()
        
        logger.info(f"LLM Raw Answer (first 200 chars): {answer[:200]}...")
        return answer if answer else "I formulated a thought, but it seems to have vanished! Could you try asking in a different way?"
    except Exception as e:
        logger.error(f"Error during LLM inference: {e}", exc_info=True)
        return "I'm having a little trouble processing that. Please try asking again in a moment."

# ... (start and help_command functions - kept as is from your file) ...
def start(update, context):
    update.message.reply_text('Hi! I am your HODL AI Assistant, here to help with information from hodltoken.net. How can I assist you?')

def help_command(update, context):
    help_text = (
        "I can help you with information about HODL, including:\n"
        "➡️ $HODL Token details & Tokenomics\n"
        "➡️ BNB rewards system\n"
        "➡️ HODL NFTs (HODL Hands®, Gem Fighter NFTs)\n"
        "➡️ Play-to-Earn games (Crypto Slash, Gem Miner, etc.)\n"
        "➡️ The HODL App features and guides\n"
        "➡️ Project roadmap, whitepaper, and security.\n\n"
        "Just ask your question! For example: 'Tell me about Gem Fighter NFTs' or 'How does HODL staking work?'"
    )
    update.message.reply_text(help_text)
    
def format_chatgpt_style(text: str) -> str:
    import re

    # Remove boilerplate
    text = re.sub(r"(?i)checking my hodl knowledge for.*?\.\.\.", "", text)
    text = re.sub(r"(?i)based on the provided (context|information)[,:]?\s*", "", text)

    # Turn numbered lists into clean bullet lists
    text = re.sub(r'\s*[\-–•]?\s*\d+\.\s+', '\n\n- ', text)

    # Turn inline lists into bullets
    text = re.sub(r'(?i)(?:including|such as):?\s*', r'\n- ', text)

    # Fix spacing and paragraphs
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()




def handle_message(update, context):
    user_query = update.message.text.strip()
    logger.info(f"User query: {user_query}")
    
    normalized_query = user_query.lower()
    
    if normalized_query in ["what are you?", "who are you?"]:
        update.message.reply_text("I'm the HODL AI Assistant! I'm here to provide information based on the official hodltoken.net website.")
        return
    if normalized_query in ["who made you?", "who created you?"]:
        update.message.reply_text("I'm an AI assistant developed to help with your HODL token questions, using data from the official website.")
        return
    
    greetings = ["hi", "hello", "hey", "yo", "what's up", "good morning", "good afternoon", "good evening", "sup"]
    greeting_starters = ["hi ", "hello ", "hey "]
    if normalized_query in greetings or any(normalized_query.startswith(g) for g in greeting_starters):
        update.message.reply_text("Hey there! 👋 How can I help you with HODL token today?")
        return

    update.message.reply_text(f"Checking my HODL knowledge for: '{user_query}'...")

    try:
        query_embedding = embedding_model.encode(user_query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS_TO_RETRIEVE,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks_for_prompt_str_list = []
        actual_retrieved_metadatas = [] # To store metadatas of chunks used for context
        context_is_relevant = False
        min_distance = float('inf') 

        if results and results.get('documents') and results['documents'][0]:
            log_msg = f"Retrieved {len(results['documents'][0])} chunks for query '{user_query}'.\n"
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                min_distance = min(min_distance, distance)
                
                # Log retrieved chunk info
                chunk_info_for_log = f"  Chunk {i+1} (Source: {metadata.get('source_url', 'Unknown')}, Distance: {distance:.4f}): {doc_text[:150].replace(chr(10), ' ')}...\n"
                log_msg += chunk_info_for_log
                
                retrieved_chunks_for_prompt_str_list.append(f"Content from {metadata.get('source_url', 'source')}:\n{doc_text}") # Slightly modified context formatting
                actual_retrieved_metadatas.append(metadata) # Store metadata for source linking
            logger.info(log_msg)
            
            if min_distance < MAX_CONTEXT_RELEVANCE_DISTANCE:
                context_is_relevant = True
        
        final_llm_answer = ""
        sources_markdown = ""

        if context_is_relevant:
            logger.info(f"Relevant context found (best distance: {min_distance:.4f}). Using RAG.")
            context_for_llm = "\n\n---\n\n".join(retrieved_chunks_for_prompt_str_list)
            
            system_instructions_rag = (
                "You are the HODL AI Assistant, an expert in HODL Token, providing clear, concise, and accurate information.\n"
                "Your task is to answer the user's question based *ONLY* on the following context from the hodltoken.net website.\n"
                "Synthesize the information from the context into an **easy-to-read answer**. Use **clear paragraphs**. If listing multiple items, steps, or key features, consider using **bullet points** (e.g., using '*' or '-'). Aim for a helpful but not overly verbose response (e.g., 2-5 sentences if appropriate, but provide necessary detail if clearly present in the context).\n\n"
                "Instructions for answering:\n"
                "1. If the context directly and completely answers the user's question, provide that answer.\n"
                "2. If the context provides partial information, give a focused answer based on what is available. You MUST explicitly state if some aspects of the question cannot be fully addressed with the given context (e.g., 'Based on the provided information, [partial answer]. However, specific details about [missing aspect] are not covered in this context.').\n"
                "3. If the context does not contain any relevant information to the user's question, you MUST state: 'I could not find specific information about that in the HODL Token documentation I have access to. You could try rephrasing or asking about a different HODL Token feature.'\n"
                "4. Do NOT add any information that is not explicitly present in the provided context. Do NOT make assumptions or infer beyond what the text clearly states."
            )
            prompt = f"<s>[INST] {system_instructions_rag}\n\nUser's Question: {user_query}\n\nProvided Context from hodltoken.net:\n{context_for_llm} [/INST]"
            
            final_llm_answer = generate_llm_answer(prompt)

            # Prepare source links if RAG was used
            unique_source_urls = set()
            for meta in actual_retrieved_metadatas: # Use metadatas from chunks that formed the context
                source_url = meta.get("source_url")
                if source_url:
                    unique_source_urls.add(source_url)
            
           

        else: # General conversation if context not relevant enough
            logger.info(f"Retrieved context not relevant enough (best distance: {min_distance:.4f} >= {MAX_CONTEXT_RELEVANCE_DISTANCE}) or no context found. Using general knowledge prompt.")
            system_instructions_general = (
                "You are a helpful and professional conversational AI assistant representing HODL Token. "
                "Politely answer the user's query. If it's about specific HODL Token features or data you don't have internal knowledge of, state that you can primarily provide information found on the hodltoken.net website and suggest they ask a question related to HODLToken's publicly available information. "
                "Keep responses concise and on-brand. Format your answer clearly, using paragraphs if needed."
            )
            prompt = f"<s>[INST] {system_instructions_general}\n\nUser's message: {user_query} [/INST]"
            final_llm_answer = generate_llm_answer(prompt)
        
        logger.info(f"Final answer to user (pre-truncation): {final_llm_answer[:300]}...")
        if len(final_llm_answer) > 4000: 
            final_llm_answer = final_llm_answer[:3950] + "\n\n[...Answer truncated due to length...]"
        formatted_answer = format_chatgpt_style(final_llm_answer)
        update.message.reply_text(formatted_answer, parse_mode='Markdown')


    except Exception as e:
        logger.error(f"Error in handle_message: {e}", exc_info=True)
        update.message.reply_text("I'm sorry, I encountered an issue while trying to process your request. Please try again.")
        


def error_handler(update, context):
    logger.warning(f'Update "{update}" caused error "{context.error}"')

def main_bot():
    if not llm_pipeline:
        logger.error("LLM pipeline is not initialized. Bot cannot generate answers. Exiting.")
        return

    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_error_handler(error_handler)

    updater.start_polling()
    logger.info("HODLToken Assistant Bot started and polling...")
    updater.idle()

if __name__ == '__main__':
    main_bot()