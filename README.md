🚀 AI Data Analyst (Multi-Agent Streamlit App)

An LLM-powered AI Data Analyst built with Streamlit, LangChain (Ollama), and Plotly, designed for hackathons and rapid data exploration.
Upload your data and interact with it using natural language for cleaning, analysis, visualization, and insights.

✨ Features
📂 Data Sources

Upload CSV files

Load CSV into in-memory SQLite and query it

Automatic dataset preview

🧹 Data Cleaner Agent

Removes duplicate rows

Fills missing values:

Numeric columns → median

Categorical columns → mode or "Unknown"

Download cleaned dataset as CSV

💬 Analyst Agent (LLM-powered)

Ask questions in natural language

LLM generates pure Pandas code

Executes safely on the dataset

Displays generated logic for transparency

Example:

"What is the average sales by category?"

📊 Visualization Agent (Smart Charts)
🔹 Natural Language Mode (Recommended)

Describe the chart in plain English

LLM automatically selects:

Chart type (Bar, Line, Histogram, Box)

X & Y columns

Aggregation (sum, mean, count, none)

Example:

"Show average revenue per region"

🔹 Manual Mode

Manually choose chart type and columns

✅ Built with Plotly
⬇️ Download charts as PNG

🧠 Insight Agent

Generates high-level insights

Highlights:

Trends

Risks

Business value

🛠 Tech Stack

Python

Streamlit

Pandas

SQLite

Plotly

LangChain

Ollama (DeepSeek-R1 8B)
