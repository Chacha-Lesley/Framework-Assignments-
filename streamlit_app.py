import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Configure the Streamlit page
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">🦠 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
st.markdown("### Interactive exploration of COVID-19 research papers")

st.markdown("""
Welcome to the CORD-19 dataset explorer! This application analyzes the metadata from COVID-19 research papers 
to provide insights into publication trends, journal distributions, and research topics.
""")

# Sidebar for controls
st.sidebar.header("🎛️ Controls")
st.sidebar.markdown("Use these controls to filter and explore the data:")

@st.cache_data
def load_data(file_path, sample_size=None):
    """Load and cache the dataset"""
    try:
        if sample_size:
            df = pd.read_csv(file_path, nrows=sample_size)
        else:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def clean_data(df):
    """Clean and prepare the dataset"""
    if df is None:
        return None
    
    df_clean = df.copy()
    
    # Fill missing values
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].fillna('Unknown Title')
    
    if 'abstract' in df_clean.columns:
        df_clean['abstract'] = df_clean['abstract'].fillna('No abstract available')
    
    # Handle dates
    date_columns = ['publish_time', 'publish_date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            year_col = col.replace('_time', '_year').replace('_date', '_year')
            df_clean[year_col] = df_clean[col].dt.year
    
    # Create word count columns
    if 'title' in df_clean.columns:
        df_clean['title_word_count'] = df_clean['title'].str.split().str.len()
    
    if 'abstract' in df_clean.columns:
        df_clean['abstract_word_count'] = df_clean['abstract'].str.split().str.len()
    
    return df_clean

def get_common_words(df, column='title', top_n=20):
    """Extract common words from a text column"""
    if df is None or column not in df.columns:
        return []
    
    text = ' '.join(df[column].dropna().astype(str))
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                  'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
                  'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 
                  'them', 'covid', 'coronavirus', 'sars', 'cov', 'pandemic', 'virus', 'disease', 'study', 
                  'analysis', 'research', 'paper', 'article', 'review'}
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return Counter(words).most_common(top_n)

# File upload section
st.sidebar.subheader("📁 Data Loading")

# Option to upload file or use sample data
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload metadata.csv", "Use sample data"]
)

df = None

if data_source == "Upload metadata.csv":
    uploaded_file = st.sidebar.file_uploader("Upload your metadata.csv file", type="csv")
    if uploaded_file is not None:
        sample_size = st.sidebar.number_input(
            "Sample size (leave 0 for full dataset)", 
            min_value=0, 
            value=10000, 
            step=1000
        )
        
        with st.spinner("Loading data..."):
            if sample_size > 0:
                df = pd.read_csv(uploaded_file, nrows=sample_size)
            else:
                df = pd.read_csv(uploaded_file)
else:
    # Create sample data for demonstration
    st.sidebar.info("Using simulated sample data for demonstration")
    
    np.random.seed(42)
    sample_data = {
        'title': [
            'COVID-19 transmission patterns in urban areas',
            'SARS-CoV-2 vaccine efficacy study',
            'Mental health impacts during pandemic lockdowns',
            'Respiratory symptoms in COVID-19 patients',
            'Economic effects of coronavirus restrictions',
            'Telemedicine adoption during COVID-19',
            'Long COVID symptoms and recovery patterns',
            'Contact tracing effectiveness analysis',
            'Hospital capacity during pandemic waves',
            'Social distancing behavioral compliance'
        ] * 1000,
        'journal': np.random.choice(['Nature', 'Science', 'NEJM', 'Lancet', 'JAMA', 'BMJ', 'PLOS ONE'], 10000),
        'publish_time': pd.date_range('2019-12-01', '2022-12-31', periods=10000),
        'abstract': ['Sample abstract about COVID-19 research'] * 10000,
        'authors': ['Smith, J.; Johnson, A.'] * 10000,
        'source_x': np.random.choice(['PubMed', 'arXiv', 'bioRxiv', 'medRxiv'], 10000)
    }
    df = pd.DataFrame(sample_data)

# Main application logic
if df is not None:
    # Clean the data
    with st.spinner("Cleaning and preparing data..."):
        df_clean = clean_data(df)
    
    # Display basic information
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", f"{len(df_clean):,}")
    
    with col2:
        if 'journal' in df_clean.columns:
            unique_journals = df_clean['journal'].nunique()
            st.metric("Unique Journals", f"{unique_journals:,}")
        else:
            st.metric("Unique Journals", "N/A")
    
    with col3:
        if 'publish_year' in df_clean.columns:
            year_range = df_clean['publish_year'].max() - df_clean['publish_year'].min()
            st.metric("Year Range", f"{int(year_range)} years")
        else:
            st.metric("Year Range", "N/A")
    
    with col4:
        if 'title_word_count' in df_clean.columns:
            avg_title_length = df_clean['title_word_count'].mean()
            st.metric("Avg Title Length", f"{avg_title_length:.1f} words")
        else:
            st.metric("Avg Title Length", "N/A")
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # Year filter
    if 'publish_year' in df_clean.columns:
        years = sorted(df_clean['publish_year'].dropna().unique())
        if len(years) > 1:
            year_range = st.sidebar.slider(
                "Select year range",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(int(min(years)), int(max(years)))
            )
            df_filtered = df_clean[
                (df_clean['publish_year'] >= year_range[0]) & 
                (df_clean['publish_year'] <= year_range[1])
            ]
        else:
            df_filtered = df_clean
            st.sidebar.info("Year filtering not available (insufficient date data)")
    else:
        df_filtered = df_clean
        st.sidebar.info("Year filtering not available (no date data)")
    
    # Journal filter
    if 'journal' in df_filtered.columns:
        top_journals = df_filtered['journal'].value_counts().head(20).index.tolist()
        selected_journals = st.sidebar.multiselect(
            "Select journals (leave empty for all)",
            options=top_journals,
            default=[]
        )
        
        if selected_journals:
            df_filtered = df_filtered[df_filtered['journal'].isin(selected_journals)]
    
    # Display filtered data info
    if len(df_filtered) != len(df_clean):
        st.info(f"Showing {len(df_filtered):,} papers after filtering (from {len(df_clean):,} total)")
    
    # Visualizations
    st.header("📈 Data Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Time Trends", "📰 Journals", "📝 Text Analysis", "🔍 Data Sample"])
    
    with tab1:
        st.subheader("Publications Over Time")
        
        if 'publish_year' in df_filtered.columns:
            year_counts = df_filtered['publish_year'].value_counts().sort_index()
            
            # Create interactive plot
            fig = px.line(
                x=year_counts.index, 
                y=year_counts.values,
                title="Number of Publications by Year",
                labels={'x': 'Year', 'y': 'Number of Publications'}
            )
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show peak year
            peak_year = year_counts.idxmax()
            peak_count = year_counts.max()
            st.info(f"📈 Peak publication year: **{peak_year}** with **{peak_count:,}** papers")
        else:
            st.warning("Publication date data not available for time trend analysis")
    
    with tab2:
        st.subheader("Top Publishing Journals")
        
        if 'journal' in df_filtered.columns:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                top_n_journals = st.slider("Number of top journals to show", 5, 20, 10)
                top_journals = df_filtered['journal'].value_counts().head(top_n_journals)
                
                fig = px.bar(
                    x=top_journals.values,
                    y=top_journals.index,
                    orientation='h',
                    title=f"Top {top_n_journals} Publishing Journals",
                    labels={'x': 'Number of Papers', 'y': 'Journal'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Top 5 Journals:**")
                for i, (journal, count) in enumerate(top_journals.head(5).items(), 1):
                    st.write(f"{i}. {journal}: {count:,} papers")
        else:
            st.warning("Journal data not available")
    
    with tab3:
        st.subheader("Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'title_word_count' in df_filtered.columns:
                st.write("**Title Length Distribution**")
                fig = px.histogram(
                    df_filtered, 
                    x='title_word_count',
                    title="Distribution of Title Word Counts",
                    labels={'title_word_count': 'Words in Title', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Title word count data not available")
        
        with col2:
            st.write("**Most Common Words in Titles**")
            if 'title' in df_filtered.columns:
                top_words = get_common_words(df_filtered, 'title', 15)
                if top_words:
                    words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
                    fig = px.bar(
                        words_df.head(10), 
                        x='Count', 
                        y='Word',
                        orientation='h',
                        title="Top 10 Words in Titles"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show word list
                    st.write("**Top 15 Words:**")
                    for i, (word, count) in enumerate(top_words, 1):
                        st.write(f"{i}. {word}: {count:,} times")
                else:
                    st.warning("Could not extract words from titles")
            else:
                st.warning("Title data not available for word analysis")
    
    with tab4:
        st.subheader("Sample Data")
        
        # Show sample of the filtered data
        st.write(f"**Sample of {min(10, len(df_filtered))} papers:**")
        
        # Select relevant columns to display
        display_columns = []
        for col in ['title', 'journal', 'publish_time', 'authors']:
            if col in df_filtered.columns:
                display_columns.append(col)
        
        if display_columns:
            sample_df = df_filtered[display_columns].head(10)
            st.dataframe(sample_df, use_container_width=True)
        else:
            st.dataframe(df_filtered.head(10), use_container_width=True)
        
        # Download option
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"cord19_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.warning("📁 Please upload a metadata.csv file or use the sample data to begin analysis.")
    
    st.markdown("""
    ### How to get started:
    
    1. **Download the CORD-19 dataset** from [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
    2. **Upload the metadata.csv file** using the file uploader in the sidebar
    3. **Choose a sample size** if working with the full dataset (recommended for performance)
    4. **Explore the visualizations** using the interactive controls
    
    **Or** select "Use sample data" to see a demonstration with simulated data.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🦠 CORD-19 Data Explorer | Built with Streamlit</p>
    <p><small>Dataset: <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge" target="_blank">CORD-19 Research Challenge</a></small></p>
</div>
""", unsafe_allow_html=True)