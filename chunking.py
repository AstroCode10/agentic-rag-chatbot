import re

def chunk_text(text: str, chunk_size: int = 500, overlap_sents: int = 2) -> list[dict]:
    sentences = re.split(r'(?=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = [] # Initialize an empty list to hold the chunks
    i = 0 # Initialize a counter to keep track of the current sentence index
    
    while i < len(sentences):
        curr_check_sentences = []
        curr_len = 0
        
        while i < len(sentences) and curr_len < chunk_size:
            curr_check_sentences.append(sentences[i])
            curr_len += len(sentences[i]) + 1  # +1 for the space after the sentence
            i += 1
        
        chunk_text = " ".join(curr_check_sentences)
        chunks.append({
            "text": chunk_text,
            "start": i - len(curr_check_sentences),
            "index": len(chunks)
        })

        i -= overlap_sents  # Move back by the number of overlapping sentences
    
    return chunks