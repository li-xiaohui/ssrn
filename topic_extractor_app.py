import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from io import BytesIO

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Set page config
st.set_page_config(
    page_title="Research Paper Topic Extractor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Topic Extractor")
st.markdown("Reads abstracts from local file or upload your own to extract main topics.")

# Initialize session state
if 'abstracts' not in st.session_state:
    st.session_state.abstracts = []
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'loaded_file' not in st.session_state:
    st.session_state.loaded_file = None
if 'auto_loaded' not in st.session_state:
    st.session_state.auto_loaded = False
if 'file_type' not in st.session_state:
    st.session_state.file_type = None  # 'abstracts' or 'keywords'

# Default local file paths
DEFAULT_FILES = ["sample_abstracts.csv", "sample_abstracts.txt", "sample_keywords.txt"]

def load_abstracts_from_file(file_path):
    """Load abstracts from a local file."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Try to find abstract column
            abstract_col = None
            for col in df.columns:
                if 'abstract' in col.lower() or 'summary' in col.lower():
                    abstract_col = col
                    break
            
            if abstract_col:
                return df[abstract_col].dropna().tolist()
            elif len(df.columns) > 0:
                # Use first column if no abstract column found
                return df[df.columns[0]].dropna().tolist()
            return []
        
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Split by double newlines or single newlines
            abstracts = [line.strip() for line in content.split('\n\n') if line.strip()]
            if len(abstracts) == 1:
                abstracts = [line.strip() for line in content.split('\n') if line.strip()]
            return abstracts
        
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            abstract_col = None
            for col in df.columns:
                if 'abstract' in col.lower() or 'summary' in col.lower():
                    abstract_col = col
                    break
            
            if abstract_col:
                return df[abstract_col].dropna().tolist()
            elif len(df.columns) > 0:
                return df[df.columns[0]].dropna().tolist()
            return []
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return []
    
    return []

def load_keywords_from_file(file_path):
    """Load keywords from a comma-delimited file. Each line contains keywords for a paper."""
    try:
        keywords_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split by comma and clean up
                    keywords = [kw.strip() for kw in line.split(',') if kw.strip()]
                    if keywords:
                        keywords_list.append(keywords)
        return keywords_list
    except Exception as e:
        st.error(f"Error loading keywords file {file_path}: {str(e)}")
        return []
    
    return []

def preprocess_text(text):
    """Preprocess text for topic extraction"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def extract_topics_tfidf(abstracts, n_topics=5, n_words=10):
    """Extract topics using TF-IDF and keyword extraction"""
    # Preprocess abstracts
    processed_abstracts = [preprocess_text(abstract) for abstract in abstracts]
    processed_abstracts = [a for a in processed_abstracts if a]  # Remove empty
    
    if not processed_abstracts:
        return []
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),  # Include unigrams and bigrams
        min_df=2,
        max_df=0.95
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_abstracts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top terms across all documents
        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[-n_words:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        
        return top_terms
    except:
        return []

def extract_topics_lda(abstracts, n_topics=5, n_words=10):
    """Extract topics using LDA"""
    # Preprocess abstracts
    processed_abstracts = [preprocess_text(abstract) for abstract in abstracts]
    processed_abstracts = [a for a in processed_abstracts if a]  # Remove empty
    
    if not processed_abstracts:
        return []
    
    # Create TF-IDF vectorizer for LDA
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_abstracts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Apply LDA
        n_topics = min(n_topics, len(processed_abstracts))
        if n_topics < 1:
            n_topics = 1
            
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(tfidf_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            topics.append({
                'topic_id': topic_idx + 1,
                'keywords': top_words,
                'weights': [topic[idx] for idx in top_indices]
            })
        
        return topics
    except Exception as e:
        st.error(f"Error in LDA: {str(e)}")
        return []

def generate_wordcloud_from_abstracts(abstracts, max_words=100, width=800, height=400):
    """Generate a wordcloud image from abstracts"""
    # Combine all abstracts
    combined_text = ' '.join([str(abstract) for abstract in abstracts if abstract])
    
    if not combined_text:
        return None
    
    # Preprocess text for wordcloud
    processed_text = preprocess_text(combined_text)
    
    if not processed_text:
        return None
    
    try:
        # Create wordcloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42
        ).generate(processed_text)
        
        # Convert to image
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Convert to bytes for Streamlit
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        plt.close(fig)
        
        return img_buffer
    except Exception as e:
        st.error(f"Error generating wordcloud: {str(e)}")
        return None

def generate_wordcloud_from_keywords(keywords_list, max_words=100, width=800, height=400):
    """Generate a wordcloud image from keywords list (list of lists)"""
    if not keywords_list:
        return None
    
    try:
        # Flatten keywords and count frequencies
        all_keywords = []
        for keywords in keywords_list:
            all_keywords.extend([kw.lower().strip() for kw in keywords if kw.strip()])
        
        if not all_keywords:
            return None
        
        # Count frequencies
        word_freq = Counter(all_keywords)
        
        # Create wordcloud from frequencies
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(word_freq)
        
        # Convert to image
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Convert to bytes for Streamlit
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        plt.close(fig)
        
        return img_buffer
    except Exception as e:
        st.error(f"Error generating wordcloud from keywords: {str(e)}")
        return None

# Try to load from default local files first
if not st.session_state.abstracts and not st.session_state.keywords:
    for default_file in DEFAULT_FILES:
        if os.path.exists(default_file):
            if default_file.endswith('keywords.txt'):
                keywords_list = load_keywords_from_file(default_file)
                if keywords_list:
                    st.session_state.keywords = keywords_list
                    st.session_state.loaded_file = default_file
                    st.session_state.file_type = 'keywords'
                    st.session_state.auto_loaded = True
                    break
            else:
                abstracts_list = load_abstracts_from_file(default_file)
                if abstracts_list:
                    st.session_state.abstracts = abstracts_list
                    st.session_state.loaded_file = default_file
                    st.session_state.file_type = 'abstracts'
                    st.session_state.auto_loaded = True
                    break

# Sidebar for input options
st.sidebar.header("📥 Input Options")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Local File", "Upload File", "Paste Text"],
    key="input_method_radio"
)

abstracts_list = []

if input_method == "Local File":
    # List available local files
    local_files = []
    for file in DEFAULT_FILES:
        if os.path.exists(file):
            local_files.append(file)
    
    # Also check for other common files
    for ext in ['.csv', '.txt', '.xlsx']:
        for file in os.listdir('.'):
            if file.endswith(ext) and file not in DEFAULT_FILES and os.path.isfile(file):
                local_files.append(file)
    
    if local_files:
        # Find index of currently loaded file if it exists
        current_index = 0
        if st.session_state.get('loaded_file') and st.session_state.loaded_file in local_files:
            try:
                current_index = local_files.index(st.session_state.loaded_file)
            except ValueError:
                current_index = 0
        
        selected_file = st.sidebar.selectbox(
            "Select local file:",
            local_files,
            index=current_index,
            key="local_file_selectbox"
        )
        
        if st.sidebar.button("📂 Load from File", type="primary"):
            if selected_file.endswith('keywords.txt') or 'keyword' in selected_file.lower():
                keywords_list = load_keywords_from_file(selected_file)
                if keywords_list:
                    st.session_state.keywords = keywords_list
                    st.session_state.abstracts = []  # Clear abstracts
                    st.session_state.loaded_file = selected_file
                    st.session_state.file_type = 'keywords'
                    st.sidebar.success(f"✅ Loaded {len(keywords_list)} keyword sets from {selected_file}")
                else:
                    st.sidebar.warning("No keywords found in the selected file.")
            else:
                abstracts_list = load_abstracts_from_file(selected_file)
                if abstracts_list:
                    st.session_state.abstracts = abstracts_list
                    st.session_state.keywords = []  # Clear keywords
                    st.session_state.loaded_file = selected_file
                    st.session_state.file_type = 'abstracts'
                    st.sidebar.success(f"✅ Loaded {len(abstracts_list)} abstracts from {selected_file}")
                else:
                    st.sidebar.warning("No abstracts found in the selected file.")
        
        # Show current loaded file
        if st.session_state.get('loaded_file'):
            st.sidebar.info(f"📄 Currently loaded: {st.session_state.loaded_file}")
    else:
        st.sidebar.warning("No local files found. Please upload a file or paste abstracts.")
        st.sidebar.info("💡 Run 'python generate_sample_abstracts.py' to create sample files.")

elif input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file with abstracts",
        type=['csv', 'xlsx', 'txt'],
        help="Upload CSV, Excel, or TXT file. For CSV/Excel, abstracts should be in a column named 'abstract' or 'summary'"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                # TXT file - read line by line
                content = uploaded_file.read().decode('utf-8')
                abstracts_list = [line.strip() for line in content.split('\n') if line.strip()]
                st.session_state.abstracts = abstracts_list
            
            if uploaded_file.name.endswith(('.csv', '.xlsx')):
                # Try to find abstract column
                abstract_col = None
                for col in df.columns:
                    if 'abstract' in col.lower() or 'summary' in col.lower():
                        abstract_col = col
                        break
                
                if abstract_col:
                    abstracts_list = df[abstract_col].dropna().tolist()
                    st.session_state.abstracts = abstracts_list
                    st.sidebar.success(f"Found {len(abstracts_list)} abstracts in column '{abstract_col}'")
                else:
                    st.sidebar.warning("Could not find 'abstract' or 'summary' column. Available columns:")
                    st.sidebar.write(df.columns.tolist())
                    # Let user select column
                    selected_col = st.sidebar.selectbox(
                        "Select column with abstracts:", 
                        df.columns,
                        key="column_selectbox"
                    )
                    if selected_col:
                        abstracts_list = df[selected_col].dropna().tolist()
                        st.session_state.abstracts = abstracts_list
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")

else:  # Paste Text
    input_text = st.sidebar.text_area(
        "Paste abstracts (one per line or separated by blank lines):",
        height=200,
        help="Enter abstracts separated by newlines",
        key="paste_text_area"
    )
    
    if input_text:
        # Split by double newlines or single newlines
        abstracts_list = [line.strip() for line in input_text.split('\n\n') if line.strip()]
        if len(abstracts_list) == 1:
            # Try splitting by single newline if no double newlines
            abstracts_list = [line.strip() for line in input_text.split('\n') if line.strip()]
        st.session_state.abstracts = abstracts_list

# Main content area
if st.session_state.abstracts or st.session_state.keywords:
    # Display loaded message
    if st.session_state.abstracts:
        loaded_msg = f"✅ Loaded {len(st.session_state.abstracts)} abstract(s)"
    else:
        loaded_msg = f"✅ Loaded {len(st.session_state.keywords)} keyword set(s)"
    
    if st.session_state.get('loaded_file'):
        loaded_msg += f" from `{st.session_state.loaded_file}`"
    if st.session_state.get('auto_loaded'):
        loaded_msg += " (auto-loaded from local drive)"
    st.success(loaded_msg)
    
    # Display wordcloud immediately
    st.header("☁️ Wordcloud Visualization")
    if st.session_state.keywords:
        wordcloud_img = generate_wordcloud_from_keywords(st.session_state.keywords, max_words=100, width=800, height=400)
    else:
        wordcloud_img = generate_wordcloud_from_abstracts(st.session_state.abstracts, max_words=100, width=800, height=400)
    
    if wordcloud_img:
        st.image(wordcloud_img)
    else:
        st.warning("Could not generate wordcloud.")
    
    st.markdown("---")
    
    # Display abstracts or keywords preview
    if st.session_state.abstracts:
        with st.expander("📄 View Loaded Abstracts"):
            for i, abstract in enumerate(st.session_state.abstracts[:10], 1):
                st.markdown(f"**Abstract {i}:**")
                st.text(abstract[:200] + "..." if len(abstract) > 200 else abstract)
                st.divider()
            if len(st.session_state.abstracts) > 10:
                st.info(f"... and {len(st.session_state.abstracts) - 10} more abstracts")
    elif st.session_state.keywords:
        with st.expander("🏷️ View Loaded Keywords"):
            for i, keywords in enumerate(st.session_state.keywords[:10], 1):
                st.markdown(f"**Paper {i}:**")
                st.text(", ".join(keywords))
                st.divider()
            if len(st.session_state.keywords) > 10:
                st.info(f"... and {len(st.session_state.keywords) - 10} more keyword sets")
    
    # Topic extraction settings (only for abstracts)
    if st.session_state.abstracts:
        st.header("⚙️ Topic Extraction Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            extraction_method = st.selectbox(
                "Extraction Method:",
                ["TF-IDF Keywords", "LDA Topic Modeling"],
                help="TF-IDF: Extract top keywords. LDA: Discover latent topics.",
                key="extraction_method_selectbox"
            )
        
        with col2:
            if extraction_method == "LDA Topic Modeling":
                n_topics = st.slider("Number of Topics:", 2, 10, 5, key="n_topics_slider")
            else:
                n_topics = 1
        
        n_words = st.slider("Keywords per Topic:", 5, 20, 10, key="n_words_slider")
        
        # Wordcloud settings
        with st.expander("☁️ Wordcloud Settings"):
            show_wordcloud = st.checkbox("Show Wordcloud", value=True, key="show_wordcloud_checkbox")
            max_words_wc = st.slider("Max Words in Wordcloud:", 50, 200, 100, key="max_words_wc_slider")
            col_w, col_h = st.columns(2)
            with col_w:
                wc_width = st.slider("Width:", 400, 1200, 800, key="wc_width_slider")
            with col_h:
                wc_height = st.slider("Height:", 300, 800, 400, key="wc_height_slider")
        
        # Extract topics
        if st.button("🔍 Extract Topics", type="primary", key="extract_topics_button"):
            with st.spinner("Analyzing abstracts..."):
                if extraction_method == "TF-IDF Keywords":
                    topics = extract_topics_tfidf(st.session_state.abstracts, n_topics=1, n_words=n_words)
                    
                    st.header("📊 Extracted Topics")
                    st.subheader("Top Keywords Across All Abstracts")
                    
                    if topics:
                        # Display wordcloud first if enabled
                        if show_wordcloud:
                            st.markdown("### ☁️ Wordcloud Visualization")
                        wordcloud_img = generate_wordcloud_from_abstracts(
                            st.session_state.abstracts,
                            max_words=max_words_wc,
                            width=wc_width,
                            height=wc_height
                        )
                        if wordcloud_img:
                            st.image(wordcloud_img)
                        else:
                            st.warning("Could not generate wordcloud.")
                            st.markdown("---")
                        
                        # Display as tags
                        st.markdown("### Main Topics:")
                        topic_str = "  ".join([f"`{topic}`" for topic in topics])
                        st.markdown(topic_str)
                        
                        # Display as bar chart
                        st.markdown("### Keyword Importance:")
                        topic_df = pd.DataFrame({
                            'Keyword': topics,
                            'Rank': range(1, len(topics) + 1)
                        })
                        st.bar_chart(topic_df.set_index('Keyword')['Rank'])
                    else:
                        st.warning("Could not extract topics. Please check your abstracts.")
                
                elif extraction_method == "LDA Topic Modeling":  # LDA
                    topics = extract_topics_lda(st.session_state.abstracts, n_topics=n_topics, n_words=n_words)
                    
                    st.header("📊 Extracted Topics")
                    
                    if topics:
                        # Display wordcloud first if enabled
                        if show_wordcloud:
                            st.markdown("### ☁️ Wordcloud Visualization")
                            wordcloud_img = generate_wordcloud_from_abstracts(
                                st.session_state.abstracts,
                                max_words=max_words_wc,
                                width=wc_width,
                                height=wc_height
                            )
                            if wordcloud_img:
                                st.image(wordcloud_img)
                            else:
                                st.warning("Could not generate wordcloud.")
                            st.markdown("---")
                        
                        # Display each topic
                        cols = st.columns(min(2, len(topics)))
                        for idx, topic_data in enumerate(topics):
                            with cols[idx % len(cols)]:
                                st.markdown(f"### Topic {topic_data['topic_id']}")
                                keywords = topic_data['keywords']
                                weights = topic_data['weights']
                                
                                # Display keywords with weights
                                for keyword, weight in zip(keywords, weights):
                                    st.markdown(f"- **{keyword}** (weight: {weight:.3f})")
                                
                                # Visualize topic
                                topic_df = pd.DataFrame({
                                    'Keyword': keywords[:5],
                                    'Weight': weights[:5]
                                })
                                st.bar_chart(topic_df.set_index('Keyword')['Weight'])
                        
                        # Summary
                        st.markdown("### 📝 Summary")
                        all_keywords = []
                        for topic_data in topics:
                            all_keywords.extend(topic_data['keywords'][:5])
                        
                        keyword_counts = Counter(all_keywords)
                        st.markdown("**Most Common Keywords Across Topics:**")
                        common_keywords = "  ".join([f"`{kw}` ({count}x)" for kw, count in keyword_counts.most_common(10)])
                        st.markdown(common_keywords)
                    else:
                        st.warning("Could not extract topics. Please check your abstracts.")

else:
    # Show info about loading from local file
    if os.path.exists("sample_abstracts.csv") or os.path.exists("sample_abstracts.txt"):
        st.info("👈 Click 'Load from File' in the sidebar to load abstracts from local drive, or upload your own file.")
    else:
        st.info("👈 Please load a local file, upload a file, or paste abstracts using the sidebar to get started.")
        st.markdown("💡 **Tip:** Run `python generate_sample_abstracts.py` to create sample abstract files.")
    
    # Show example
    st.markdown("### Example Input Format:")
    st.code("""
Abstract 1: This paper presents a novel approach to machine learning...
    
Abstract 2: We propose a new method for natural language processing...
    
Abstract 3: In this work, we investigate deep learning architectures...
    """)
    
    # Show available local files
    if os.path.exists("sample_abstracts.csv") or os.path.exists("sample_abstracts.txt"):
        st.markdown("### 📁 Available Local Files:")
        if os.path.exists("sample_abstracts.csv"):
            st.markdown("- `sample_abstracts.csv`")
        if os.path.exists("sample_abstracts.txt"):
            st.markdown("- `sample_abstracts.txt`")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** For best results, provide multiple abstracts (at least 3-5) related to similar research areas.")

