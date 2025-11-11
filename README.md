# 💡 SHL Assessment Recommendation System (Gemini RAG)

This project implements an intelligent Retrieval-Augmented Generation (RAG) system using the Google Gemini API and Streamlit. It solves the problem of finding the most relevant SHL assessment tests by accepting a natural language query or a detailed Job Description (JD).

The system uses Gemini's powerful embedding models for semantic search and the Gemini-2.5-Flash model for structured constraint extraction and re-ranking, delivering highly accurate and filtered recommendations.

## ✨ Features

  * **Hybrid Recommendation:** Combines **Vector Search** (semantic similarity with the user query) and **Constraint Filtering** (extracted max duration, job levels, test types) for precise results.
  * **Structured Constraint Extraction:** Uses Gemini to parse a complex JD and output structured JSON constraints (Job Level, Test Type, Duration, etc.).
  * **Efficient Caching:** Leverages Streamlit's `@st.cache_resource` to load the large assessment catalog and generate all embeddings **only once**, drastically improving performance and reducing API costs on subsequent runs.
  * **Secure API Handling:** Uses Streamlit Secrets for secure management of the Gemini API Key.
  * **Interactive Web App:** Deployed as a simple, interactive application using Streamlit.

## 📁 Project Structure

The codebase is organized into modular Python files for maintainability:

```
SHL-Assessment-RAG/
├── app.py                      # Main Streamlit application file (UI logic)
├── rag.py                      # Core RAG pipeline (search, constraint extraction, filtering)
├── embeddings_cache.py         # Functions for caching the Gemini Client and catalog embeddings
├── helper.py                   # Utility functions (data joining, duration extraction, mapping)
├── config.py                   # Stores constants (model names, file paths, mappings)
├── shl_assessment_details1.json# Assessment catalog data (must be present!)
├── requirements.txt            # Python dependencies
└── .streamlit/
    └── secrets.toml            # Securely stores the GEMINI_API_KEY
```

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

1.  **Python 3.9+** installed.
2.  A **Google Gemini API Key**.

### 1\. Local Setup

Clone the repository and install the required packages:

```bash
# 1. Clone the repository
git clone https://github.com/darshandugar2004/SHLassessment.git
cd SHL-Assessment-RAG

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 2\. Configure API Key

The application uses Streamlit Secrets for secure API key management.

1.  Create a directory named **`.streamlit`** in the project root.

2.  Inside `.streamlit`, create a file named **`secrets.toml`**.

3.  Add your Gemini API Key using the exact key name used in `embeddings_cache.py`:

    ```toml
    # .streamlit/secrets.toml
    GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
    ```

### 3\. Run the Application

Start the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The app will open in your browser (usually `http://localhost:8501`).

**Note:** The first run will be slow because it has to call the Gemini API multiple times to generate and cache all embeddings. Subsequent runs will be nearly instantaneous.

## ☁️ Deployment (Streamlit Community Cloud)

This application is designed for seamless deployment on Streamlit Community Cloud (formerly Streamlit Sharing).

1.  **Push to GitHub:** Ensure all project files (`.py` files, `requirements.txt`, and the data file `shl_assessment_details1.json`) are committed and pushed to a GitHub repository.
2.  **Deploy:** Go to the [Streamlit Community Cloud dashboard](https://share.streamlit.io/).
3.  **Secrets:** When deploying, navigate to the app's **Secrets** section and copy the contents of your local `secrets.toml` file into the cloud secret text area.
4.  The application will automatically build, install dependencies, and run.

## 🛠️ Key Libraries Used

  * `streamlit`: For building the web interface.
  * `google-genai`: The official Google library for interacting with Gemini and embedding models.
  * `pandas`/`numpy`: For data handling and efficient vector operations.
  * `scikit-learn`: Specifically for `cosine_similarity` in the RAG pipeline.

-----

Would you like me to draft a summary of the RAG pipeline's logic (vector search, LLM constraint extraction, filtering) to include in the README as a "How It Works" section?
