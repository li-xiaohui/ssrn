"""
Two-Page Dashboard App using Dash
Page 1: Wordcloud for all keyword lists
Page 2: Search bar for keywords in abstracts and keywords, returning matched papers
"""

import dash
from dash import dcc, html, Input, Output, dash_table, State
import pandas as pd
from collections import Counter

# Try to import dash_holoniq_wordcloud, fallback to instructions if not available
try:
    from dash_holoniq_wordcloud import DashWordcloud
except ImportError:
    print(
        "Warning: dash_holoniq_wordcloud not installed. Please install it with: pip install dash-holoniq-wordcloud"
    )
    DashWordcloud = None


def load_data():
    """Load and merge keywords and abstracts data"""
    try:
        # Load keywords data
        keywords_df = pd.read_csv("sample_keywords.csv")
        
        # Load abstracts data
        abstracts_df = pd.read_csv("sample_abstracts.csv")
        
        # Merge on paper_id (keywords_df index + 1 = paper_id)
        papers = []
        for idx, row in keywords_df.iterrows():
            paper_id = f"paper_{idx + 1}"
            
            # Parse keywords from comma-separated string
            keywords_str = str(row["keywords"])
            keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
            
            # Get abstract if available
            abstract_row = abstracts_df[abstracts_df["paper_id"] == paper_id]
            abstract = abstract_row["abstract"].values[0] if not abstract_row.empty else ""
            
            papers.append({
                "paper_id": paper_id,
                "title": str(row["title"]),
                "publication_date": str(row["publication_date"]),
                "citations": int(row["citations"]),
                "keywords": keywords,
                "abstract": abstract,
            })
        
        return papers
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


def generate_wordcloud_data(papers):
    """Generate wordcloud data from all keywords"""
    all_keywords = []
    for paper in papers:
        all_keywords.extend([kw.lower().strip() for kw in paper["keywords"]])
    
    # Count keyword frequencies
    keyword_freq = Counter(all_keywords)
    
    # Convert to list format for wordcloud (list of [word, frequency] pairs)
    wordcloud_data = [[word, freq] for word, freq in keyword_freq.items()]
    return wordcloud_data


# Initialize app
app = dash.Dash(__name__)

# Load data
papers = load_data()
wordcloud_data = generate_wordcloud_data(papers)

# App layout with navigation
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Navigation bar
    html.Div([
        html.H1(
            "Research Paper Dashboard",
            style={
                "textAlign": "center",
                "marginBottom": "20px",
                "color": "#2c3e50",
                "fontSize": "32px"
            }
        ),
        html.Div([
            dcc.Link(
                "Wordcloud", 
                href="/",
                style={
                    "margin": "0 20px",
                    "padding": "10px 20px",
                    "backgroundColor": "#3498db",
                    "color": "white",
                    "textDecoration": "none",
                    "borderRadius": "5px",
                    "fontSize": "16px",
                    "fontWeight": "bold"
                }
            ),
            dcc.Link(
                "Search Papers", 
                href="/search",
                style={
                    "margin": "0 20px",
                    "padding": "10px 20px",
                    "backgroundColor": "#3498db",
                    "color": "white",
                    "textDecoration": "none",
                    "borderRadius": "5px",
                    "fontSize": "16px",
                    "fontWeight": "bold"
                }
            ),
        ], style={"textAlign": "center", "marginBottom": "30px"}),
    ]),
    
    # Page content
    html.Div(id='page-content')
])


# Page 1: Wordcloud
def create_wordcloud_page():
    """Create the wordcloud page layout"""
    return html.Div([
        html.H2(
            "Keyword Wordcloud",
            style={
                "textAlign": "center",
                "marginBottom": "30px",
                "color": "#2c3e50"
            }
        ),
        html.P(
            "Click on any word in the wordcloud to see associated papers",
            style={
                "textAlign": "center",
                "fontSize": "16px",
                "color": "#7f8c8d",
                "marginBottom": "20px"
            }
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
                        "Please install dash_holoniq_wordcloud: pip install dash-holoniq-wordcloud",
                        style={"textAlign": "center", "padding": "20px"}
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
    ])


# Page 2: Search
def create_search_page():
    """Create the search page layout"""
    return html.Div([
        html.H2(
            "Search Papers",
            style={
                "textAlign": "center",
                "marginBottom": "30px",
                "color": "#2c3e50"
            }
        ),
        html.P(
            "Enter keywords to search in paper abstracts and keywords",
            style={
                "textAlign": "center",
                "fontSize": "16px",
                "color": "#7f8c8d",
                "marginBottom": "20px"
            }
        ),
        html.Div([
            dcc.Input(
                id="search-input",
                type="text",
                placeholder="Enter keywords (e.g., 'deep learning', 'neural networks')...",
                style={
                    "width": "60%",
                    "padding": "12px",
                    "fontSize": "16px",
                    "border": "2px solid #3498db",
                    "borderRadius": "5px",
                    "marginRight": "10px"
                },
                n_submit=0
            ),
            html.Button(
                "Search",
                id="search-button",
                n_clicks=0,
                style={
                    "padding": "12px 30px",
                    "fontSize": "16px",
                    "backgroundColor": "#3498db",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "5px",
                    "cursor": "pointer"
                }
            ),
        ], style={
            "textAlign": "center",
            "marginBottom": "30px",
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center"
        }),
        html.Div(
            id="search-results",
            style={"margin": "20px auto", "maxWidth": "1200px"},
        ),
    ])


# Callback for page routing
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/search':
        return create_search_page()
    else:
        return create_wordcloud_page()


# Callback for wordcloud click (Page 1)
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

    # Find papers containing this keyword
    matched_papers = []
    for paper in papers:
        # Check if keyword is in paper's keywords or abstract
        paper_keywords_lower = [kw.lower().strip() for kw in paper["keywords"]]
        abstract_lower = paper["abstract"].lower()
        
        if selected_keyword in paper_keywords_lower or selected_keyword in abstract_lower:
            matched_papers.append(paper)

    if not matched_papers:
        return f"Keyword: {selected_keyword.title()}", html.Div(
            "No papers found for this keyword.",
            style={"textAlign": "center", "padding": "20px", "color": "#7f8c8d"}
        )

    # Create DataFrame
    df = pd.DataFrame([
        {
            "Title": paper["title"],
            "Publication Date": paper["publication_date"],
            "Citations": paper["citations"],
            "Keywords": ", ".join(paper["keywords"][:5]) + ("..." if len(paper["keywords"]) > 5 else ""),
        }
        for paper in matched_papers
    ])

    # Sort by publication date (descending - most recent first)
    df["Publication Date"] = pd.to_datetime(df["Publication Date"])
    df = df.sort_values("Publication Date", ascending=False)
    df["Publication Date"] = df["Publication Date"].dt.strftime("%Y-%m-%d")

    # Create table
    table = dash_table.DataTable(
        id="papers-table",
        columns=[
            {"name": "Title", "id": "Title"},
            {"name": "Publication Date", "id": "Publication Date"},
            {"name": "Citations", "id": "Citations", "type": "numeric"},
            {"name": "Keywords", "id": "Keywords"},
        ],
        data=df.to_dict("records"),
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "fontFamily": "Arial, sans-serif",
            "whiteSpace": "normal",
            "height": "auto",
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
    )

    keyword_display = (
        f"Keyword: {selected_keyword.title()} ({len(matched_papers)} paper(s))"
    )

    return keyword_display, html.Div([
        html.H3(
            f"Papers containing '{selected_keyword.title()}'",
            style={"marginBottom": "15px", "color": "#2c3e50"},
        ),
        table,
    ])


# Callback for search (Page 2)
@app.callback(
    Output("search-results", "children"),
    [
        Input("search-button", "n_clicks"),
        Input("search-input", "n_submit")
    ],
    [State("search-input", "value")]
)
def update_search_results(n_clicks, n_submit, search_query):
    """Update search results based on query"""
    if not search_query or not search_query.strip():
        return html.Div(
            "Please enter a search query.",
            style={"textAlign": "center", "padding": "20px", "color": "#7f8c8d"}
        )

    # Split search query into individual keywords
    search_terms = [term.strip().lower() for term in search_query.split(",") if term.strip()]
    
    if not search_terms:
        return html.Div(
            "Please enter a valid search query.",
            style={"textAlign": "center", "padding": "20px", "color": "#7f8c8d"}
        )

    # Find matching papers
    matched_papers = []
    for paper in papers:
        # Check if any search term matches keywords or abstract
        paper_keywords_lower = [kw.lower().strip() for kw in paper["keywords"]]
        abstract_lower = paper["abstract"].lower()
        
        # Check if any search term is found
        matches = False
        for term in search_terms:
            if term in paper_keywords_lower or term in abstract_lower:
                matches = True
                break
        
        if matches:
            matched_papers.append(paper)

    if not matched_papers:
        return html.Div([
            html.H3(
                f"No papers found for: {search_query}",
                style={"textAlign": "center", "color": "#7f8c8d", "marginBottom": "20px"}
            ),
            html.P(
                "Try different keywords or check spelling.",
                style={"textAlign": "center", "color": "#95a5a6"}
            )
        ])

    # Create DataFrame
    df = pd.DataFrame([
        {
            "Title": paper["title"],
            "Publication Date": paper["publication_date"],
            "Citations": paper["citations"],
            "Keywords": ", ".join(paper["keywords"][:5]) + ("..." if len(paper["keywords"]) > 5 else ""),
            "Abstract": paper["abstract"][:200] + "..." if len(paper["abstract"]) > 200 else paper["abstract"],
        }
        for paper in matched_papers
    ])

    # Sort by publication date (descending - most recent first)
    df["Publication Date"] = pd.to_datetime(df["Publication Date"])
    df = df.sort_values("Publication Date", ascending=False)
    df["Publication Date"] = df["Publication Date"].dt.strftime("%Y-%m-%d")

    # Create table
    table = dash_table.DataTable(
        id="search-results-table",
        columns=[
            {"name": "Title", "id": "Title"},
            {"name": "Publication Date", "id": "Publication Date"},
            {"name": "Citations", "id": "Citations", "type": "numeric"},
            {"name": "Keywords", "id": "Keywords"},
            {"name": "Abstract", "id": "Abstract"},
        ],
        data=df.to_dict("records"),
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "fontFamily": "Arial, sans-serif",
            "whiteSpace": "normal",
            "height": "auto",
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
    )

    return html.Div([
        html.H3(
            f"Found {len(matched_papers)} paper(s) for: {search_query}",
            style={"marginBottom": "15px", "color": "#2c3e50", "textAlign": "center"}
        ),
        table,
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8052)

