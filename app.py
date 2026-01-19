

import streamlit as st
import pandas as pd
import sqlite3
from typing import Dict

from agents import AnalystAgent,VisualizationAgent,InsightAgent,DataCleanerAgent







st.set_page_config(" AI Data Analyst", layout="wide")

# Session state to store cleaned data
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
st.title("🚀 AI Data Analyst")
st.caption("LLM + Multi-Agent + CSV + SQL")

source = st.sidebar.selectbox("Select Data Source", ["CSV", "SQLite"])


if source == "CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)


else:
    conn = sqlite3.connect("data.db")
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
    st.markdown("---")

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
                rec.get("aggregate_func")
            )
                st.markdown(f"**LLM choice:** {rec['chart_type']} | Aggregation: {rec['aggregate_func']}")
                st.plotly_chart(fig, use_container_width=True)

            st.session_state.viz_chat.append({"role": "assistant", "content": "Generated visualization."})

    else:
        chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Histogram", "Box"])

        if chart_type in ["Bar", "Line"]:
            x_col = st.selectbox("Select X Column", data_for_viz.columns)
            y_col = st.selectbox("Select Y Column", data_for_viz.select_dtypes(include="number").columns)
            agg=st.selectbox("Select Aggregation Function",["sum",'mean','count',"none"])
        else:
            
            x_col = st.selectbox("Select Numeric Column", data_for_viz.select_dtypes(include="number").columns)
            y_col = None
            agg = None

        if st.button("Generate Visualization"):
            fig = viz_agent.run(data_for_viz, chart_type, x_col, y_col, aggregate_func=agg)
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
    st.markdown("---")
    # INSIGHT AGENT
    st.subheader("🧠 Insight Agent")
    if st.button("Generate Insights"):
        insight = InsightAgent().run(df)
        st.write(insight)
    st.markdown("---")
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
            st.code(result)

        st.session_state.chat_history.append({"role": "assistant", "content": "Generated analysis code and result."})
    
else:
    st.info("Upload data to begin")


st.markdown("---")
st.caption("Thank you for visiting......")
