import arxiv
import pandas as pd
from datetime import datetime, timedelta

# Parameters
CATEGORY = "cs.AI"
QUERY = "artificial intelligence"
MAX_RESULTS = 100  # You can increase this if you want more papers
DAYS_BACK = 7  # How many days back to look for recent papers

# Calculate date range
end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS_BACK)

# Search arXiv
search = arxiv.Search(
    query=f"cat:{CATEGORY} AND all:{QUERY}",
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

results = []
for result in search.results():
    # Filter by submission date
    if result.published < start_date:
        continue
    # Extract authors and affiliations
    authors = []
    affiliations = []
    for author in result.authors:
        authors.append(author.name)
        if hasattr(author, 'affiliation') and author.affiliation:
            affiliations.append(author.affiliation)
        else:
            affiliations.append("")
    results.append({
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "authors": "; ".join(authors),
        "affiliations": "; ".join([a for a in affiliations if a]),
        "abstract": result.summary,
        "submission_date": result.published.strftime("%Y-%m-%d"),
        "pdf_url": result.pdf_url,
    })

# Save to Excel
if results:
    df = pd.DataFrame(results)
    df.to_excel("arxiv_cs_AI_recent.xlsx", index=False)
    print(f"Saved {len(df)} results to arxiv_cs_AI_recent.xlsx")
else:
    print("No recent results found.")
