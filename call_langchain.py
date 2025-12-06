from typing import List, Tuple
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
import os

def process_text_chunks(
    abstract: str, 
    body: str, 
    conclusion: str, 
    llm: ChatOpenAI,
    chunk_size: int = 80000
) -> str:
    """
    Process text chunks with token counting and summarization.
    
    Args:
        abstract (str): The abstract text
        body (str): The main body text
        conclusion (str): The conclusion text
        llm (ChatOpenAI): The language model to use for summarization
        chunk_size (int): The maximum size of each chunk in tokens (default: 80000)
        
    Returns:
        str: Combined text with processed chunks
    """
    # Initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Count tokens for each section
    abstract_tokens = len(enc.encode(abstract))
    body_tokens = len(enc.encode(body))
    conclusion_tokens = len(enc.encode(conclusion))
    
    total_tokens = abstract_tokens + body_tokens + conclusion_tokens
    
    # If total tokens are under chunk_size, return original text
    if total_tokens <= chunk_size:
        return f"{abstract}\n\n{body}\n\n{conclusion}"
    
    # Initialize text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=lambda x: len(enc.encode(x))
    )
    
    # Split body into chunks
    chunks = text_splitter.split_text(body)
    
    # Calculate target tokens per summary
    target_tokens_per_summary = (chunk_size - abstract_tokens - conclusion_tokens) // len(chunks)
    
    # Create summarization chain
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt="""Write a concise summary of the following text, preserving all numerical values and key information:
        {text}
        """,
        combine_prompt="""Write a concise summary of the following text, preserving all numerical values and key information:
        {text}
        """,
        verbose=True
    )
    
    def reduce_summary(text: str, target_tokens: int) -> str:
        """Recursively reduce summary until it meets the target token count."""
        current_tokens = len(enc.encode(text))
        if current_tokens <= target_tokens:
            return text
            
        # Create a more aggressive summarization prompt
        reduction_prompt = f"""Create an extremely concise summary of the following text, preserving all numerical values and key information. 
        The summary must be significantly shorter than the original while maintaining all critical information:
        {text}
        """
        
        # Generate new summary
        docs = [Document(page_content=text)]
        new_summary = chain.run(docs)
        
        # If we're still over the limit, try one more time
        if len(enc.encode(new_summary)) > target_tokens:
            return reduce_summary(new_summary, target_tokens)
            
        return new_summary
    
    # Process each chunk
    summaries = []
    for chunk in chunks:
        # Create document for the chunk
        docs = [Document(page_content=chunk)]
        
        # Generate initial summary
        summary = chain.run(docs)
        
        # Ensure summary meets token limit through recursive reduction
        summary_tokens = len(enc.encode(summary))
        if summary_tokens > target_tokens_per_summary:
            summary = reduce_summary(summary, target_tokens_per_summary)
        
        summaries.append(summary)
    
    # Combine all parts
    combined_text = f"{abstract}\n\n{''.join(summaries)}\n\n{conclusion}"
    
    return combined_text
