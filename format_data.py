import json
import re
from urllib.parse import urljoin, urlparse

def load_data(json_path):
    """Loads the crawled data from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def remove_known_boilerplate(text, header_patterns, footer_patterns):
    """Removes known header and footer boilerplate text using regex."""
    original_text_len = len(text)
    
    # Attempt to remove headers
    temp_text_after_header_removal = text
    for pattern in header_patterns:
        # Find the first match for the header pattern
        # REMOVED flags=re.IGNORECASE | re.DOTALL
        match = re.search(pattern, temp_text_after_header_removal) 
        if match:
            # Remove everything from the start of the text up to the end of the match
            temp_text_after_header_removal = temp_text_after_header_removal[match.end():]
            break # Assuming one primary header block at the start

    text_after_header_removal = temp_text_after_header_removal.strip()

    # Attempt to remove footers
    temp_text_after_footer_removal = text_after_header_removal
    for pattern in footer_patterns:
        # Find the last match for the footer pattern
        # Search from the end of the string
        best_match = None
        # REMOVED flags=re.IGNORECASE | re.DOTALL
        for m in re.finditer(pattern, temp_text_after_footer_removal): 
            best_match = m
        if best_match:
            # Remove everything from the start of the match to the end of the text
            temp_text_after_footer_removal = temp_text_after_footer_removal[:best_match.start()]
            break # Assuming one primary footer block at the end
            
    cleaned_text = temp_text_after_footer_removal.strip()

    return cleaned_text

def is_content_problematic(text, problematic_phrases):
    """Checks if the text primarily consists of problematic phrases (e.g., 404s, JS required)."""
    text_lower = text.lower().strip()
    if not text_lower:
        return True # Empty text is problematic
        
    # Check for 404 phrases
    for phrase in problematic_phrases:
        if phrase.lower() in text_lower:
            # If the text is very short and contains a 404 message, it's likely just a 404 page.
            if len(text_lower) < len(phrase) + 150: # Allow some leeway for minimal extra text
                return True
            # If the phrase makes up a large percentage of the text
            if (text_lower.count(phrase.lower()) * len(phrase)) / len(text_lower) > 0.5: # If >50% of text is this phrase
                 return True

    # Check for "JavaScript appears to be disabled"
    js_disabled_phrase = "JavaScript appears to be disabled"
    if js_disabled_phrase.lower() in text_lower and len(text_lower) < len(js_disabled_phrase) + 100:
        return True
        
    return False

def clean_text_artifacts(text):
    """Cleans common text artifacts and normalizes whitespace."""
    # Remove initial artifacts like ".\n.\n.\n" more carefully
    text = re.sub(r"^\s*(\.\s*?\n\s*){2,}", "", text)
    
    # Normalize sequences of newlines and spaces.
    # Replace multiple newlines with a double newline (standard paragraph separator).
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Replace any remaining sequence of one or more newlines with a single newline.
    # This might be too aggressive if single newlines within paragraphs are meaningful.
    # Consider keeping \n\n as paragraph separators and \n for line breaks if needed.
    # For now, let's aim to separate paragraphs clearly.
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text) # Clean newlines surrounded by spaces/tabs
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    # Filter out empty lines that might result from cleaning, then join
    text = "\n".join([line for line in lines if line.strip()])
    
    # Convert text separated by single newlines (likely list items or tight text)
    # into more readable paragraphs if they form blocks.
    # This step needs careful consideration based on desired output for RAG.
    # For now, we'll keep single newlines if they exist after previous cleaning.
    # Let's try to ensure paragraphs are separated by at least one empty line (double newline).
    text = re.sub(r'\n([A-Z])', r'\n\n\1', text) # Heuristic: newline followed by uppercase suggests new paragraph

    return text.strip()

def chunk_text_by_paragraph(text, min_chunk_char_len=100, max_chunk_char_len=600):
    """
    Chunks text by paragraphs (separated by one or more newlines after cleaning).
    Tries to keep chunks within a reasonable size.
    """
    # Paragraphs are now expected to be separated by single newlines after clean_text_artifacts
    # or double newlines if they were preserved. Let's split by one or more newlines.
    raw_paragraphs = re.split(r'\n{2,}', text) # Split by two or more newlines for paragraphs
    
    chunks = []
    current_chunk = ""

    for p in raw_paragraphs:
        p_stripped = p.strip()
        if not p_stripped:
            continue

        if not current_chunk:
            current_chunk = p_stripped
        elif len(current_chunk) + len(p_stripped) + 2 <= max_chunk_char_len: # +2 for potential \n\n
            current_chunk += "\n\n" + p_stripped
        else:
            # Add the current_chunk if it's of minimum length
            if len(current_chunk) >= min_chunk_char_len:
                chunks.append(current_chunk)
            # Start a new chunk with the current paragraph
            current_chunk = p_stripped
            # If this new paragraph itself is too long, it will be added as is (or could be further split)
            if len(current_chunk) > max_chunk_char_len:
                 if len(current_chunk) >= min_chunk_char_len: # Ensure it's not too small
                    chunks.append(current_chunk) # Add the oversized paragraph as its own chunk
                 current_chunk = "" # Reset for next paragraph

    # Add the last accumulated chunk if it's not empty and meets min length
    if current_chunk and len(current_chunk) >= min_chunk_char_len:
        chunks.append(current_chunk)
        
    # Final pass to ensure no chunks are too small after all processing
    final_chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_char_len]
    return final_chunks


# --- Main Processing ---
# Define boilerplate patterns (these may need refinement)
HEADER_PATTERNS = [
    re.compile(r"^\s*(\.\s*){2,}\s*\nKey Info.*?Claim BNB\s*\nKey Info.*?Useful Links\s*\nSecurity.*?Timeline", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*Connect\s*\nKey Info.*?Claim BNB\s*\nConnect\s*\nKey Info.*?Useful Links\s*\nSecurity.*?Timeline", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*Trade\s*\nSwap\s*\nPerps.*?Docs", re.IGNORECASE | re.DOTALL) # For pages like /buy (pancakeswap)
]
FOOTER_PATTERNS = [
    re.compile(r"Key Info\s*\nAbout.*?Disclaimers\s*\n© HODLVERSE Inc.*?AdRoll PIXEL\s*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"Chatting with HODL\s*\nHi, please help me\s*\n\*{5} Google Analytics PIXEL \*{5}\s*\n\*{5} AdRoll PIXEL \*{5}\s*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"Key Info\s*\nAbout.*?© HODLVERSE Inc.*$", re.IGNORECASE | re.DOTALL), # Broader footer
    re.compile(r"About\s*\nTokenomics.*?Docs\s*$", re.IGNORECASE | re.DOTALL) # For pages like /buy (pancakeswap)
]

PROBLEMATIC_PHRASES = [
    "404\nThe page could not be found.",
    "This content requires JavaScript",
    "JavaScript appears to be disabled",
    "Wähle ein anderes Land oder eine andere Region", # From /apple-app
    "JavaScript muss aktiviert sein, damit Sie Google Drive verwenden können" # From /promo-kit
]

# Define path prefixes that often lead to 404s or irrelevant content based on your JSON
# These are paths on hodltoken.net that crawler might have constructed
IRRELEVANT_PATH_PREFIXES = [
    "/t.me/",
    "/hodltoken.net/", # For malformed relative links like /hodltoken.net/roadmap
    "/store/",
    "/careers/uk/",
    "/legal/warranty/",
    "/compliance/", # If these are just legal boilerplate without substance for RAG
    "/choose-country-region/",
    "/buy", # This page seems to be an external integration (PancakeSwap)
    "/swap", #Likely external
    "/liquidity/",
    "/position-managers",
    "/cake-staking/",
    "/pools", #PancakeSwap
    "/bridge", #PancakeSwap
    "/prediction", #PancakeSwap
    "/lottery", #PancakeSwap
    "/info/v3", #PancakeSwap
    "/burn-dashboard", #PancakeSwap
    "/ifo", #PancakeSwap
    "/voting" #PancakeSwap
]


input_json_path = "hodltoken.net_data_playwright.json"
output_json_path = "hodltoken_formatted_chunks.json"

crawled_data = load_data(input_json_path)
all_chunks_data = [] # Store final chunk objects
processed_urls_count = 0
skipped_urls_count = 0
total_chunks_created = 0

print(f"Starting processing of {len(crawled_data)} URLs...")

for url, raw_text in crawled_data.items():
    parsed_url = urlparse(url)
    
    # Skip URLs with irrelevant path prefixes
    if any(parsed_url.path.startswith(prefix) for prefix in IRRELEVANT_PATH_PREFIXES):
        print(f"Skipping URL due to irrelevant path prefix: {url}")
        skipped_urls_count +=1
        continue

    if not raw_text or not raw_text.strip():
        print(f"Skipping URL due to initially empty content: {url}")
        skipped_urls_count +=1
        continue

    # 1. Remove known boilerplate
    text_no_boilerplate = remove_known_boilerplate(raw_text, HEADER_PATTERNS, FOOTER_PATTERNS)
    
    if not text_no_boilerplate.strip():
        # print(f"Content became empty after boilerplate removal: {url}") # Debugging
        skipped_urls_count +=1
        continue

    # 2. Check for problematic content (404s, JS required) after boilerplate removal
    if is_content_problematic(text_no_boilerplate, PROBLEMATIC_PHRASES):
        print(f"Skipping URL due to problematic content (404/JS issue post-boilerplate): {url}")
        skipped_urls_count += 1
        continue

    # 3. Clean remaining text artifacts
    cleaned_text = clean_text_artifacts(text_no_boilerplate)
    
    if not cleaned_text:
        print(f"Skipping URL due to empty content after all cleaning: {url}")
        skipped_urls_count +=1
        continue

    # 4. Chunk the cleaned text
    # Using paragraph chunking with min/max length constraints
    text_chunks = chunk_text_by_paragraph(cleaned_text, min_chunk_char_len=150, max_chunk_char_len=2500)

    if not text_chunks:
        # print(f"No valid chunks produced after filtering by length for URL: {url}") # Debugging
        # Potentially log this or handle very short pages differently if their content is still valuable
        continue
        
    for i, chunk_content in enumerate(text_chunks):
        all_chunks_data.append({
            "source_url": url,
            "chunk_id": f"{parsed_url.path.replace('/', '_')}_{i+1}", # Create a more filesystem-friendly chunk_id
            "text": chunk_content
        })
        total_chunks_created += 1
    
    processed_urls_count += 1
    if processed_urls_count % 10 == 0: # Print progress update
        print(f"Processed {processed_urls_count} URLs. Total chunks: {total_chunks_created}. Skipped: {skipped_urls_count}")


# 5. Save the formatted chunks
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_chunks_data, f, ensure_ascii=False, indent=4)

print(f"\n--- Processing Complete ---")
print(f"Successfully processed and chunked content from {processed_urls_count} URLs.")
print(f"Skipped {skipped_urls_count} URLs due to various filters (empty, problematic, irrelevant path).")
print(f"Total text chunks created: {total_chunks_created}")
print(f"Formatted and chunked data saved to: {output_json_path}")

# Simple check of the output file
try:
    with open(output_json_path, 'r', encoding='utf-8') as f:
        final_data_check = json.load(f)
    print(f"Verification: Successfully loaded {len(final_data_check)} chunks from the output file.")
    if final_data_check:
        print("Sample of the first chunk:")
        print(json.dumps(final_data_check[0], indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error verifying output file: {e}")