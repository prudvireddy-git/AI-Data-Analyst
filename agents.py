
import streamlit as st
import pandas as pd
import sqlite3
import os
import traceback
from typing import Dict
import ollama
import json
import re
#from langchain_groq import ChatGroq
#from dotenv import load_dotenv
#load_dotenv()
from langchain_ollama import OllamaLLM

client = OllamaLLM(model="deepseek-r1:8b")

#from langchain_community.llms import Ollama
#client = Ollama(model="deepseek-r1:8b")
#os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
#os.environ["LANGCHAIN_TRACING_V2"]="true"
#os.environ["LANGCHAIN_PROJECT"]="AI DATA ANALYST"
#groq_api_key=os.getenv("GROQ_API_KEY")
#client_groq=ChatGroq(groq_api_key=groq_api_key,model_name='')


def call_llm(prompt: str) -> str:
    return client.invoke(prompt)

class InsightAgent:
    def run(self, df: pd.DataFrame):
        prompt = f"""
Summarize key insights for this dataset.
Columns: {list(df.columns)}
Describe trends, risks, and business value.
"""
        return call_llm(prompt) 
            
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
        
        return code
    

class VisualizationAgent:
    def recommend_chart(self, df: pd.DataFrame, prompt: str) -> Dict:
        llm_prompt = f"""
You are an expert data visualization analyst.

Dataset columns: {list(df.columns)}
User request: {prompt}

Choose the BEST visualization and aggregation strategy.

Allowed chart_type: Bar, Line, Histogram, Box
Allowed aggregate_func: sum, mean, count, none

Return ONLY valid JSON in this format:
        {{
          "chart_type": "Bar|Line|Histogram|Box",
          "x": "column_name",
          "y": "column_name",
          "aggregate_func": "sum|mean|count|none"
        }}
        """
        response = call_llm(llm_prompt)
        cleaned = re.sub(r"```json|```", "", response).strip()

        return json.loads(cleaned)

    def run(self, df: pd.DataFrame, chart_type: str, x_col: str = None, y_col: str = None, aggregate_func: str = 'None'):
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