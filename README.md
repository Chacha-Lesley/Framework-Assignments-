# CORD-19 Data Explorer

Interactive analysis and visualization tools for the CORD-19 research metadata.

This repository contains a small analysis script, a Streamlit-based interactive explorer, and a Jupyter notebook for exploring the CORD-19 metadata (CSV) collected by the Allen Institute for AI.

## What this project includes

- `metadata.csv` — The CORD-19 metadata file (NOT included here; obtain from Kaggle or the CORD-19 project).
- `cord19_analysis.py` — A Python script with functions to load, clean, and visualize the dataset from the command line / Python REPL.
- `streamlit_app.py` — A Streamlit application for interactive exploration (upload `metadata.csv` or use included sample data).
- `cord19_notebook.ipynb` — Notebook with exploratory analysis, charts, and examples (optional, for interactive exploration).
- `requirements.txt` — Python dependencies used by the scripts and app.

## Features

- Load and inspect the CORD-19 metadata (titles, abstracts, journals, publish dates, sources).
- Basic cleaning and derived columns (title/abstract word counts, publication year extraction).
- Static analysis script (`cord19_analysis.py`) that prints summaries and displays charts using matplotlib/seaborn.
- Interactive Streamlit app (`streamlit_app.py`) with filtering, time trends, top journals, text analysis, and CSV download of filtered results.

## Requirements

- Python 3.8+ (recommended).
- Install dependencies with pip using the provided `requirements.txt`.

Installation (PowerShell):

```powershell
# create and activate a venv (optional but recommended)
python -m venv .venv;
.\.venv\Scripts\Activate.ps1

# install required packages
pip install -r requirements.txt
```

If you prefer conda:

```powershell
conda create -n cord19 python=3.10 -y;
conda activate cord19
pip install -r requirements.txt
```

## Usage

1) Using the analysis script

- The `cord19_analysis.py` file exposes a function `run_complete_analysis(file_path='metadata.csv', sample_size=10000)` that runs the pipeline (load -> clean -> visualize -> word analysis). To run it from the command line without editing the script, use:

```powershell
python -c "from cord19_analysis import run_complete_analysis; run_complete_analysis('metadata.csv', sample_size=10000)"
```

- To run the full dataset (may be slow / memory heavy):

```powershell
python -c "from cord19_analysis import run_complete_analysis; run_complete_analysis('metadata.csv')"
```

Notes:
- Use the `sample_size` parameter to limit rows for faster iteration during development.
- The script prints diagnostic output and shows matplotlib charts (these will open in a window when running locally).

2) Running the Streamlit app (recommended for interactive exploration)

```powershell
streamlit run streamlit_app.py
```

- The app opens in a browser. You can either upload a `metadata.csv` file using the sidebar or choose the built-in sample data for a quick demo.
- Use the sidebar filters to restrict by year, journals, and sample size.

3) Jupyter Notebook

- Open the notebook with Jupyter Lab / Notebook and run cells interactively:

```powershell
jupyter notebook cord19_notebook.ipynb
```

## Data source

The CORD-19 metadata can be downloaded from the Allen Institute / Kaggle:

- https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

Place the `metadata.csv` file in this project directory (or upload it to the Streamlit app at runtime).

Note: `metadata.csv` may be large (>50MB). Use the `sample_size` options in the scripts or Streamlit app to avoid memory issues while developing.

## Files and purpose

- `cord19_analysis.py`: Command-line friendly analysis pipeline. Useful for quick scripts and reproducible analysis in a Python REPL.
- `streamlit_app.py`: Interactive dashboard with controls, visualizations, and CSV export of filtered data.
- `cord19_notebook.ipynb`: Notebook for step-by-step exploration and experimentation.
- `requirements.txt`: List of Python packages used (pandas, numpy, matplotlib, seaborn, plotly, streamlit, wordcloud, etc.).

## Performance tips

- For very large CSVs, use `sample_size` or read the file in chunks (pandas chunking) to avoid running out of memory.
- When iterating on the Streamlit app, prefer using a small sample or a filtered subset to speed up UI responsiveness.

## Contributing

If you'd like to extend this project:

- Add tests or example notebooks demonstrating additional analyses.
- Improve performance by using dask/polars for large dataset handling.
- Add unit tests for the cleaning and parsing functions in `cord19_analysis.py`.


