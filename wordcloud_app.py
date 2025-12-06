"""
Interactive Wordcloud App using Dash and dash_holoniq_wordcloud
Displays clickable keywords that show associated papers with titles, dates, and citation counts.
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
from datetime import datetime, timedelta
import random

# Try to import dash_holoniq_wordcloud, fallback to instructions if not available
try:
    from dash_holoniq_wordcloud import DashWordcloud
except ImportError:
    print(
        "Warning: dash_holoniq_wordcloud not installed. Please install it with: pip install dash-holoniq-wordcloud"
    )
    DashWordcloud = None


# Load papers data from CSV file
def load_papers_data():
    """Load papers data from sample_keywords.csv with keywords, title, publication_date, and citations"""
    try:
        df = pd.read_csv("sample_keywords.csv")
        papers = []
        
        for idx, row in df.iterrows():
            # Parse keywords from comma-separated string
            keywords_str = str(row["keywords"])
            keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
            
            papers.append(
                {
                    "paper_id": f"paper_{idx + 1}",
                    "title": str(row["title"]),
                    "publish_date": str(row["publication_date"]),
                    "citation_count": int(row["citations"]),
                    "keywords": keywords,
                }
            )
        
        return papers
    except Exception as e:
        print(f"Error loading papers data: {e}")
        # Fallback to old method if CSV doesn't exist
        return load_papers_fallback()


def load_papers_fallback():
    """Fallback method to load from old format if CSV doesn't exist"""
    keywords_list = []
    with open("sample_keywords.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keywords = [kw.strip() for kw in line.split(",") if kw.strip()]
                if keywords:
                    keywords_list.append(keywords)
    
    papers = []
    base_date = datetime(2020, 1, 1)
    
    for idx, keywords in enumerate(keywords_list):
        # Generate random publish date (within last 4 years)
        days_ago = random.randint(0, 1460)  # 4 years
        publish_date = (base_date + timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Generate random citation count (0-500)
        citation_count = random.randint(0, 500)
        
        # Create title from first keyword
        title = f"Research on {keywords[0] if keywords else 'Machine Learning'}"
        
        papers.append(
            {
                "paper_id": f"paper_{idx + 1}",
                "title": title,
                "publish_date": publish_date,
                "citation_count": citation_count,
                "keywords": keywords,
            }
        )
    
    return papers


# Create keyword to papers mapping
def create_keyword_mapping(papers):
    """Create a mapping from keywords to papers"""
    keyword_to_papers = {}

    for paper in papers:
        for keyword in paper["keywords"]:
            keyword_lower = keyword.lower().strip()
            if keyword_lower not in keyword_to_papers:
                keyword_to_papers[keyword_lower] = []
            keyword_to_papers[keyword_lower].append(paper)

    return keyword_to_papers


# Generate wordcloud data
def generate_wordcloud_data(keyword_to_papers):
    """Generate data for wordcloud with word frequencies"""
    word_freq = {}
    for keyword, papers_list in keyword_to_papers.items():
        word_freq[keyword] = len(papers_list)

    # Convert to list format for wordcloud (list of [word, frequency] pairs)
    wordcloud_data = [[word, freq] for word, freq in word_freq.items()]
    return wordcloud_data


# Initialize app
app = dash.Dash(__name__)

# Generate data
papers = load_papers_data()
keyword_to_papers = create_keyword_mapping(papers)
wordcloud_data = generate_wordcloud_data(keyword_to_papers)

# App layout
app.layout = html.Div(
    [
        html.H1(
            "Interactive Research Paper Keyword Wordcloud",
            style={"textAlign": "center", "marginBottom": "30px", "color": "#2c3e50"},
        ),
        html.Div(
            [
                html.P(
                    "Click on any word in the wordcloud to see associated papers",
                    style={
                        "textAlign": "center",
                        "fontSize": "16px",
                        "color": "#7f8c8d",
                    },
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(
            [
                (
                    DashWordcloud(
                        id="wordcloud",
                        list=wordcloud_data,
                        width=1200,
                        height=800,
                        gridSize=6,
                        color="random-dark",
                        backgroundColor="white",
                        shuffle=False,
                        rotateRatio=0.5,
                        shape="circle",
                        hover=True,
                    )
                    if DashWordcloud
                    else html.Div(
                        "Please install dash_holoniq_wordcloud: pip install dash-holoniq-wordcloud"
                    )
                )
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "marginBottom": "40px",
                "width": "100%",
            },
        ),
        html.Div(
            id="selected-keyword",
            style={
                "textAlign": "center",
                "fontSize": "20px",
                "fontWeight": "bold",
                "marginBottom": "20px",
                "color": "#3498db",
            },
        ),
        html.Div(
            id="papers-table-container",
            style={"margin": "20px auto", "maxWidth": "1200px"},
        ),
    ]
)


# Callback for wordcloud click
@app.callback(
    [
        Output("selected-keyword", "children"),
        Output("papers-table-container", "children"),
    ],
    [Input("wordcloud", "clickData")],
)
def update_table(clickData):
    """Update table when a word is clicked"""
    if not clickData:
        return "", html.Div()

    # Handle different possible formats of clickData
    if isinstance(clickData, dict):
        selected_keyword = clickData.get("text", "").lower().strip()
    elif isinstance(clickData, str):
        selected_keyword = clickData.lower().strip()
    else:
        return "", html.Div()

    if not selected_keyword:
        return "", html.Div()

    # Get papers for this keyword
    if selected_keyword not in keyword_to_papers:
        return f"Keyword: {selected_keyword}", html.Div(
            "No papers found for this keyword."
        )

    papers_list = keyword_to_papers[selected_keyword]

    # Create DataFrame
    df = pd.DataFrame(
        [
            {
                "Title": paper["title"],
                "Publish Date": paper["publish_date"],
                "Citation Count": paper["citation_count"],
            }
            for paper in papers_list
        ]
    )

    # Sort by publish date (descending - most recent first)
    df["Publish Date"] = pd.to_datetime(df["Publish Date"])
    df = df.sort_values("Publish Date", ascending=False)
    df["Publish Date"] = df["Publish Date"].dt.strftime("%Y-%m-%d")

    # Create table
    table = dash_table.DataTable(
        id="papers-table",
        columns=[
            {"name": "Title", "id": "Title", "presentation": "markdown"},
            {"name": "Publish Date", "id": "Publish Date"},
            {"name": "Citation Count", "id": "Citation Count", "type": "numeric"},
        ],
        data=df.to_dict("records"),
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "fontFamily": "Arial, sans-serif",
        },
        style_header={
            "backgroundColor": "#3498db",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
        },
        style_data={"backgroundColor": "#f8f9fa", "color": "black"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#e9ecef"}
        ],
        page_size=10,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        markdown_options={"html": True},
    )

    keyword_display = (
        f"Keyword: {selected_keyword.title()} ({len(papers_list)} paper(s))"
    )

    return keyword_display, html.Div(
        [
            html.H3(
                f"Papers containing '{selected_keyword.title()}'",
                style={"marginBottom": "15px", "color": "#2c3e50"},
            ),
            table,
        ]
    )


if __name__ == "__main__":
    app.run(debug=True, port=8051)
