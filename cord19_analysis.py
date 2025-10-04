# CORD-19 Dataset Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Part 1: Data Loading and Basic Exploration
print("=" * 50)
print("PART 1: DATA LOADING AND BASIC EXPLORATION")
print("=" * 50)

def load_and_explore_data(file_path='metadata.csv', sample_size=None):
    """
    Load and perform basic exploration of the CORD-19 dataset
    
    Args:
        file_path (str): Path to the metadata.csv file
        sample_size (int): If provided, loads only a sample of the data
    """
    try:
        # Load the data (with sampling if specified)
        if sample_size:
            print(f"Loading sample of {sample_size} rows...")
            df = pd.read_csv(file_path, nrows=sample_size)
        else:
            print("Loading full dataset...")
            df = pd.read_csv(file_path)
        
        print(f"✓ Data loaded successfully!")
        
        # Basic exploration
        print("\n--- BASIC DATA INFORMATION ---")
        print(f"Dataset dimensions: {df.shape}")
        print(f"Number of rows: {df.shape[0]:,}")
        print(f"Number of columns: {df.shape[1]}")
        
        print("\n--- FIRST FEW ROWS ---")
        print(df.head())
        
        print("\n--- COLUMN NAMES ---")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\n--- DATA TYPES ---")
        print(df.dtypes)
        
        print("\n--- BASIC STATISTICS ---")
        print(df.describe())
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: metadata.csv file not found!")
        print("Please download the file from: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Part 2: Data Cleaning and Preparation
print("\n" + "=" * 50)
print("PART 2: DATA CLEANING AND PREPARATION")
print("=" * 50)

def clean_and_prepare_data(df):
    """
    Clean and prepare the dataset for analysis
    """
    if df is None:
        return None
    
    print("--- MISSING VALUES ANALYSIS ---")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percentage.values
    }).sort_values('Missing Percentage', ascending=False)
    
    print(missing_df)
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Handle missing values in key columns
    print("\n--- CLEANING STEPS ---")
    
    # Fill missing titles with placeholder
    if 'title' in df_clean.columns:
        missing_titles = df_clean['title'].isnull().sum()
        df_clean['title'] = df_clean['title'].fillna('Unknown Title')
        print(f"✓ Filled {missing_titles} missing titles")
    
    # Fill missing abstracts
    if 'abstract' in df_clean.columns:
        missing_abstracts = df_clean['abstract'].isnull().sum()
        df_clean['abstract'] = df_clean['abstract'].fillna('No abstract available')
        print(f"✓ Filled {missing_abstracts} missing abstracts")
    
    # Handle publication dates
    date_columns = ['publish_time', 'publish_date']
    for col in date_columns:
        if col in df_clean.columns:
            # Convert to datetime
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            
            # Extract year for analysis
            year_col = col.replace('_time', '_year').replace('_date', '_year')
            df_clean[year_col] = df_clean[col].dt.year
            print(f"✓ Processed {col} and created {year_col}")
    
    # Create useful derived columns
    if 'title' in df_clean.columns:
        df_clean['title_word_count'] = df_clean['title'].str.split().str.len()
        print("✓ Created title_word_count column")
    
    if 'abstract' in df_clean.columns:
        df_clean['abstract_word_count'] = df_clean['abstract'].str.split().str.len()
        print("✓ Created abstract_word_count column")
    
    # Remove rows with critical missing information
    initial_rows = len(df_clean)
    if 'title' in df_clean.columns:
        df_clean = df_clean[df_clean['title'] != 'Unknown Title']
    
    print(f"✓ Final cleaned dataset: {len(df_clean):,} rows (removed {initial_rows - len(df_clean):,} rows)")
    
    return df_clean

# Part 3: Data Analysis and Visualization
print("\n" + "=" * 50)
print("PART 3: DATA ANALYSIS AND VISUALIZATION")
print("=" * 50)

def perform_analysis(df):
    """
    Perform basic analysis and create visualizations
    """
    if df is None:
        return None
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CORD-19 Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Publications by year
    if 'publish_year' in df.columns:
        year_counts = df['publish_year'].value_counts().sort_index()
        # Filter reasonable years (COVID-19 research mainly 2019-2022)
        year_counts = year_counts[(year_counts.index >= 2010) & (year_counts.index <= 2023)]
        
        axes[0, 0].bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Publications by Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Papers')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        print(f"✓ Created publications by year chart")
        print(f"   Peak year: {year_counts.idxmax()} with {year_counts.max():,} papers")
    
    # 2. Top journals
    if 'journal' in df.columns:
        top_journals = df['journal'].value_counts().head(10)
        
        axes[0, 1].barh(range(len(top_journals)), top_journals.values, color='lightgreen', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_journals)))
        axes[0, 1].set_yticklabels([j[:30] + '...' if len(j) > 30 else j for j in top_journals.index])
        axes[0, 1].set_title('Top 10 Publishing Journals')
        axes[0, 1].set_xlabel('Number of Papers')
        
        print(f"✓ Created top journals chart")
        print(f"   Top journal: {top_journals.index[0]} with {top_journals.iloc[0]:,} papers")
    
    # 3. Word count distribution
    if 'title_word_count' in df.columns:
        word_counts = df['title_word_count'].dropna()
        
        axes[1, 0].hist(word_counts, bins=30, color='salmon', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of Title Word Counts')
        axes[1, 0].set_xlabel('Number of Words in Title')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(word_counts.mean(), color='red', linestyle='--', label=f'Mean: {word_counts.mean():.1f}')
        axes[1, 0].legend()
        
        print(f"✓ Created word count distribution")
        print(f"   Average title length: {word_counts.mean():.1f} words")
    
    # 4. Source distribution
    if 'source_x' in df.columns:
        source_counts = df['source_x'].value_counts().head(10)
    elif 'source' in df.columns:
        source_counts = df['source'].value_counts().head(10)
    else:
        source_counts = None
    
    if source_counts is not None:
        axes[1, 1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Distribution by Source')
        print(f"✓ Created source distribution chart")
    else:
        axes[1, 1].text(0.5, 0.5, 'Source data not available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Source Distribution (Data Not Available)')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_common_words(df, column='title', top_n=20):
    """
    Find most common words in titles or abstracts
    """
    if df is None or column not in df.columns:
        return None
    
    print(f"\n--- MOST COMMON WORDS IN {column.upper()} ---")
    
    # Combine all text
    text = ' '.join(df[column].dropna().astype(str))
    
    # Simple word extraction (remove common stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                  'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
                  'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    # Extract words (letters only, convert to lowercase)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequency
    word_counts = Counter(words).most_common(top_n)
    
    # Display results
    for i, (word, count) in enumerate(word_counts, 1):
        print(f"{i:2d}. {word:<15} ({count:,} times)")
    
    return word_counts

# Example usage function
def run_complete_analysis(file_path='metadata.csv', sample_size=10000):
    """
    Run the complete analysis pipeline
    """
    print("🔬 Starting CORD-19 Dataset Analysis")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_and_explore_data(file_path, sample_size)
    if df is None:
        print("❌ Analysis stopped - could not load data")
        return None
    
    # Step 2: Clean data
    df_clean = clean_and_prepare_data(df)
    if df_clean is None:
        print("❌ Analysis stopped - could not clean data")
        return None
    
    # Step 3: Perform analysis
    perform_analysis(df_clean)
    
    # Step 4: Word analysis
    analyze_common_words(df_clean, 'title', 15)
    
    print("\n✅ Analysis complete!")
    print(f"📊 Analyzed {len(df_clean):,} research papers")
    
    return df_clean

# Instructions for running the analysis
print("""
🚀 INSTRUCTIONS TO RUN THE ANALYSIS:

1. Download the metadata.csv file from:
   https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

2. Place the metadata.csv file in the same directory as this script

3. Run the analysis:
   df = run_complete_analysis('metadata.csv', sample_size=10000)
   
   Note: Use sample_size parameter to work with a subset of the data
   Remove sample_size parameter to analyze the full dataset

4. The analysis will:
   ✓ Load and explore the data
   ✓ Clean and prepare the dataset
   ✓ Create visualizations
   ✓ Analyze common words in titles
   ✓ Display summary statistics

Example commands:
- Full analysis: df = run_complete_analysis('metadata.csv')
- Sample analysis: df = run_complete_analysis('metadata.csv', 10000)
""")

# Uncomment the line below to run the analysis (after downloading the data):
# df = run_complete_analysis('metadata.csv', sample_size=10000)