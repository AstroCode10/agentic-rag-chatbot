import re

def chunk_text(text: str, chunk_size: int = 500, overlap_sents: int = 2) -> list[dict]:
    sentences = re.split(r'(?=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = [] # Initialize an empty list to hold the chunks
    i = 0 # Initialize a counter to keep track of the current sentence index
    
    while i < len(sentences):
        curr_check_sentences = []
        curr_len = 0
        
        for j in range(i, len(sentences)):
            sentence = sentences[j]
            if curr_len > 0 and curr_len + len(sentence) + 1 > chunk_size:
                break

            curr_check_sentences.append(sentences[j])
            curr_len += len(sentences[j]) + 1  # +1 for the space after the sentence
        
        chunk_text = " ".join(curr_check_sentences)
        chunks.append({
            "text": chunk_text,
            "start": i,
            "index": len(chunks)
        })

        num_sents_added = len(curr_check_sentences)
        advance_step = num_sents_added - overlap_sents

        i += max(1, advance_step)  # Move back by the number of overlapping sentences
    
    return chunks