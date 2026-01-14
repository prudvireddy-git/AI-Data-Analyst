

import streamlit as st
import pandas as pd
import sqlite3
import os
import traceback
from typing import Dict


from langchain_community.llms import Ollama

client = Ollama(model="deepseek-r1:8b")


    
def call_llm(prompt: str) -> str:
    return client.invoke(prompt)


class DataCleanerAgent:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.drop_duplicates(inplace=True)
        for col in df.columns:
            if df[col].dtype != "object":
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
        return df


class AnalystAgent:
    def run(self, df: pd.DataFrame, question: str) -> Dict:
        prompt = f"""
You are a data analyst.
Dataset columns: {list(df.columns)}
Question: {question}
Write ONLY pandas code. Assume df is available.
pls give less and accurate response
"""
        code = call_llm(prompt).strip()
        local_env = {"df": df.copy(), "pd": pd}
        exec(code, {}, local_env)
        return {"code": code, "output": local_env.get("result", None)}


class InsightAgent:
    def run(self, df: pd.DataFrame) -> str:
        prompt = f"""
Summarize key insights for this dataset.
Columns: {list(df.columns)}
Describe trends, risks, and business value.
"""
        return call_llm(prompt)


class VisualizationAgent:
    def recommend_chart(self, df: pd.DataFrame, prompt: str) -> dict:
        llm_prompt = f"""
You are an expert data visualization analyst.

Dataset columns: {list(df.columns)}
User request: {prompt}

Choose the BEST visualization and aggregation strategy.

Allowed chart_type: Bar, Line, Histogram, Box
Allowed aggregate_func: sum, mean, count, none

Rules:
- Use aggregation only when grouping is needed
- Histogram and Box must use aggregate_func = none

Return STRICT JSON with keys:
chart_type, x, y, aggregate_func
"""
        response = call_llm(llm_prompt)
        return eval(response)

    def run(self, df: pd.DataFrame, chart_type: str, x_col: str = None, y_col: str = None, aggregate_func: str = "none"):
        import plotly.express as px
        import plotly.graph_objects as go

        fig = None

        if chart_type in ["Bar", "Line"]:
            if aggregate_func == "sum":
                data = df.groupby(x_col)[y_col].sum().reset_index()
            elif aggregate_func == "mean":
                data = df.groupby(x_col)[y_col].mean().reset_index()
            elif aggregate_func == "count":
                data = df.groupby(x_col)[y_col].count().reset_index()
            else:
                data = df[[x_col, y_col]]

            if chart_type == "Bar":
                fig = px.bar(data, x=x_col, y=y_col)
            else:
                fig = px.line(data, x=x_col, y=y_col)

        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col)

        elif chart_type == "Box":
            fig = px.box(df, y=x_col)

        fig.update_layout(title=f"{chart_type} Chart")
        return fig



st.set_page_config(" AI Data Analyst", layout="wide")

# Session state to store cleaned data
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
st.title("🚀 Hackathon AI Data Analyst")
st.caption("LLM + Multi-Agent + CSV + SQL")

source = st.sidebar.selectbox("Select Data Source", ["CSV", "SQLite"])


if source == "CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)


else:
    conn = sqlite3.connect(":memory:")
    st.sidebar.info("Using in-memory SQLite")
    sql_file = st.sidebar.file_uploader("Upload CSV to load into SQL", type=["csv"])
    if sql_file:
        df = pd.read_csv(sql_file)
        df.to_sql("data", conn, if_exists="replace", index=False)
        df = pd.read_sql("SELECT * FROM data", conn)


if 'df' in locals():
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # CLEANING AGENT
    st.subheader("🧹 Data Cleaner Agent")
    if st.button("Run Cleaning Agent"):
        cleaner = DataCleanerAgent()
        st.session_state.cleaned_df = cleaner.run(df)
        st.success("Data cleaned")
        st.dataframe(st.session_state.cleaned_df.head())

    # DOWNLOAD CLEANED DATA
    if st.session_state.cleaned_df is not None:
        csv = st.session_state.cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned Data (CSV)",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # ANALYST AGENT
    st.subheader("💬 Analyst Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_msg = st.chat_input("Ask a question about your data")

    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            data_for_analysis = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df
            analyst = AnalystAgent()
            result = analyst.run(data_for_analysis, user_msg)
            st.markdown("Here is the generated Pandas logic:")
            st.code(result["code"], language="python")

        st.session_state.chat_history.append({"role": "assistant", "content": "Generated analysis code and result."})

    # VISUALIZATION AGENT
    st.subheader("📊 Visualization Agent (LLM-powered)")

    data_for_viz = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df
    viz_agent = VisualizationAgent()

    viz_mode = st.radio("Visualization Mode", ["Natural Language (Recommended)","Manual"])

    if "viz_chat" not in st.session_state:
        st.session_state.viz_chat = []

# Visualization chat history
    for msg in st.session_state.viz_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    fig = None

    if viz_mode == "Natural Language (Recommended)":
        viz_prompt = st.chat_input("Describe the visualization you want")

        if viz_prompt:
            st.session_state.viz_chat.append({"role": "user", "content": viz_prompt})
            with st.chat_message("user"):
                st.markdown(viz_prompt)

            with st.chat_message("assistant"):
                viz_agent = VisualizationAgent()
                rec = viz_agent.recommend_chart(data_for_viz, viz_prompt)
                fig = viz_agent.run(
                data_for_viz,
                rec.get("chart_type"),
                rec.get("x"),
                rec.get("y"),
                rec.get("aggregate_func", "none")
            )
                st.markdown(f"**LLM choice:** {rec['chart_type']} | Aggregation: {rec['aggregate_func']}")
                st.plotly_chart(fig, use_container_width=True)

            st.session_state.viz_chat.append({"role": "assistant", "content": "Generated visualization."})

    else:
        chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Histogram", "Box"])

        if chart_type in ["Bar", "Line"]:
            x_col = st.selectbox("Select X Column", data_for_viz.columns)
            y_col = st.selectbox("Select Y Column", data_for_viz.select_dtypes(include="number").columns)
        else:
            x_col = st.selectbox("Select Numeric Column", data_for_viz.select_dtypes(include="number").columns)
            y_col = None

        if st.button("Generate Visualization"):
            fig = viz_agent.run(data_for_viz, chart_type, x_col, y_col)
            st.plotly_chart(fig, use_container_width=True)

    if fig is not None:
            import io
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button(
            "⬇️ Download Chart as PNG",
            buf.getvalue(),
            file_name="chart.png",
            mime="image/png"
        )

    # INSIGHT AGENT
    st.subheader("🧠 Insight Agent")
    if st.button("Generate Insights"):
        insight = InsightAgent().run(df)
        st.write(insight)

else:
    st.info("Upload data to begin")


st.markdown("---")
st.caption("Thank you for visiting......")
