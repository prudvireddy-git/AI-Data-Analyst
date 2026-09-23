# 🚀 AI Data Analyst

An **AI-powered data analysis application** built with **Streamlit, LangChain, Groq, Pandas, Plotly, and Multi-Agent Architecture**.

The application allows users to upload datasets, clean data automatically, generate visualizations using natural language, obtain AI-generated insights, and interact with their data through an AI analyst.

---

## ✨ Features

### 📂 1. Multiple Data Sources

Currently supports:

* CSV files
* SQLite database

Users can upload a CSV file and analyze it directly.

For SQLite mode, the uploaded CSV is loaded into a SQLite table called `data`.

---

### 🧹 2. Data Cleaner Agent

The **Data Cleaner Agent** automatically performs basic data preprocessing.

It currently:

* Removes duplicate rows
* Detects missing values
* Fills missing numerical values using the median
* Fills missing categorical values using the mode
* Uses `"Unknown"` when no categorical mode is available

Example:

```text
Raw Dataset
     ↓
Remove Duplicates
     ↓
Handle Missing Values
     ↓
Clean Dataset
     ↓
Download Cleaned CSV
```

The cleaned dataset can be downloaded as:

```text
cleaned_data.csv
```

---

### 📊 3. AI-Powered Visualization Agent

The Visualization Agent uses an LLM to understand the user's natural-language visualization request.

For example:

```text
Show me average sales by region
```

The LLM determines:

* Chart type
* X-axis
* Y-axis
* Aggregation function

Supported charts:

* Bar
* Line
* Histogram
* Box Plot

Supported aggregation functions:

* Sum
* Mean
* Count
* None

Example workflow:

```text
User Request
     ↓
Visualization Agent
     ↓
LLM Recommendation
     ↓
Chart Configuration
     ↓
Plotly
     ↓
Interactive Chart
```

The generated visualization can also be downloaded as a PNG image.

---

### 🧠 4. Insight Agent

The Insight Agent analyzes the dataset and generates a natural-language summary.

It asks the LLM to identify:

* Important trends
* Potential risks
* Business value
* General dataset insights

Example:

```text
Generate Insights
        ↓
Insight Agent
        ↓
Groq LLM
        ↓
AI-generated insights
```

---

### 💬 5. AI Analyst Chat

Users can ask questions about their dataset using natural language.

Example:

```text
What is the average sales value?

Which region has the highest revenue?

Show me the relationship between age and income.

What pandas code can calculate total sales by category?
```

The Analyst Agent converts the question into Pandas code.

Example:

```python
df.groupby("Category")["Sales"].sum()
```

The application displays the generated Pandas logic.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      Streamlit      │
                    │     Web Interface   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Data Source     │
                    │                     │
                    │ CSV / SQLite        │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │      DataFrame (Pandas)   │
                 └─────────────┬─────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
       │   Cleaner   │  │Visualization │  │   Insight   │
       │    Agent    │  │    Agent     │  │    Agent    │
       └─────────────┘  └──────┬───────┘  └──────┬──────┘
                               │                 │
                               ▼                 ▼
                         ┌──────────────────────────┐
                         │        Groq LLM          │
                         │   openai/gpt-oss-120b    │
                         └──────────────────────────┘
                               │
                               ▼
                         ┌──────────────┐
                         │ Plotly / AI  │
                         │   Results    │
                         └──────────────┘
```

---

# 🤖 Multi-Agent Architecture

The application uses multiple specialized agents.

| Agent              | Responsibility                                       |
| ------------------ | ---------------------------------------------------- |
| DataCleanerAgent   | Cleans and preprocesses data                         |
| VisualizationAgent | Recommends and generates charts                      |
| InsightAgent       | Generates dataset insights                           |
| AnalystAgent       | Converts natural-language questions into Pandas code |

Each agent has a specific responsibility instead of making one LLM handle the entire application.

---

# 🧠 LLM

The application currently uses:

```text
Groq
  ↓
openai/gpt-oss-120b
```

The LLM is used for:

* Visualization recommendations
* Dataset insights
* Natural-language data analysis
* Pandas code generation

The model can be changed in the `ChatGroq` configuration.

Example:

```python
client_groq = ChatGroq(
    model_name="openai/gpt-oss-120b",
    groq_api_key=st.secrets["GROQ_API_KEY"]
)
```

---

# 🛠️ Tech Stack

| Technology     | Purpose                        |
| -------------- | ------------------------------ |
| Python         | Core programming language      |
| Streamlit      | Web application UI             |
| Pandas         | Data manipulation and analysis |
| SQLite         | Local database                 |
| LangChain      | LLM integration                |
| LangChain Groq | Groq integration               |
| Groq           | LLM inference                  |
| Plotly         | Interactive visualizations     |
| Kaleido        | PNG chart export               |
| python-dotenv  | Environment configuration      |

---

# 📁 Project Structure

Recommended structure:

```text
AI-Data-Analyst/
│
├── app.py
├── agents.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sample.csv
│
└── .streamlit/
    └── secrets.to
```

