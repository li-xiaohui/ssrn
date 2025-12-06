import requests
from bs4 import BeautifulSoup
import pandas as pd
import arxiv
import os
import time
from urllib.parse import urljoin

def download_arxiv_paper(arxiv_id):
    """Download a paper from arXiv using its ID"""
    try:
        # Remove version number if present (e.g., 2504.01848v1 -> 2504.01848)
        arxiv_id = arxiv_id.split('v')[0]
        
        # Search for the paper
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        # Create papers directory if it doesn't exist
        if not os.path.exists('papers'):
            os.makedirs('papers')
            
        # Download the paper
        paper.download_pdf(dirpath='papers', filename=f"{arxiv_id}.pdf")
        print(f"Downloaded paper: {arxiv_id}")
        return True
    except Exception as e:
        print(f"Error downloading paper {arxiv_id}: {str(e)}")
        return False

def extract_tables_from_github():
    """Extract tables from GitHub page and save to CSV"""
    url = "https://github.com/dair-ai/ML-Papers-of-the-Week/tree/main"
    
    # Get the raw content URL
    raw_url = "https://raw.githubusercontent.com/dair-ai/ML-Papers-of-the-Week/main/README.md"
    
    try:
        # Get the content
        response = requests.get(raw_url)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all markdown tables
        tables = []
        current_table = []
        
        for line in response.text.split('\n'):
            if line.startswith('|'):
                current_table.append(line)
            elif current_table:
                tables.append(current_table)
                current_table = []
        
        if current_table:
            tables.append(current_table)
            
        # Process each table
        for i, table in enumerate(tables):
            # Convert to DataFrame
            df = pd.read_csv(pd.StringIO('\n'.join(table)), sep='|')
            
            # Clean up column names
            df.columns = df.columns.str.strip()
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            
            # Save to CSV
            df.to_csv(f'table_{i+1}.csv', index=False)
            
            # Extract arXiv links and download papers
            for col in df.columns:
                for cell in df[col]:
                    if isinstance(cell, str) and 'arxiv.org' in cell:
                        # Extract arXiv ID from URL
                        arxiv_id = cell.split('arxiv.org/abs/')[-1].split('"')[0]
                        download_arxiv_paper(arxiv_id)
                        # Be nice to arXiv's servers
                        time.sleep(3)
                        
        print("Tables have been saved to CSV files")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    extract_tables_from_github()
