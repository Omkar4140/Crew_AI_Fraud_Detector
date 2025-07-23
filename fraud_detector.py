# fraud_app.py
import os
import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM

# --- Streamlit UI ---
st.title("🔍 AI-Powered Fraud Risk Analyzer")
customer_name = st.text_input("Customer Name", "TechCorp Solutions")
industry = st.text_input("Industry Domain", "AI Software Company")

# --- API Key Setup ---
openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]

# --- Define Pydantic Output Schema ---
class RiskAssessment(BaseModel):
    risk_score: float = Field(description="Risk score 0-10", ge=0, le=10)
    risk_summary: str = Field(description="Brief risk summary")
    risk_factors: List[str] = Field(description="3 main risk factors", min_items=3, max_items=3)

# --- Define LLM ---
llm = LLM(
    provider="openrouter",
    config={
        "model": "meta-llama/llama-3.1-8b-instruct",
        "api_key": openrouter_api_key,
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0,
        "max_tokens": 512
    }
)

# --- Define Agent ---
analyst = Agent(
    role="Senior Fraud Risk Analyst",
    goal=f"Assess fraud risk for {customer_name} in {industry}",
    backstory="15+ years of experience in enterprise fraud detection, specializing in tech companies.",
    tools=[],
    verbose=False,
    llm=llm
)

# --- Define Task ---
task = Task(
    description=f"""
    Analyze {customer_name}, a company in the {industry} space, for fraud risks.
    Consider: industry red flags, compliance, financial patterns, and market threats.

    Return:
    - A score from 0–10
    - A short summary
    - Exactly 3 risk factors
    """,
    expected_output="Pydantic RiskAssessment result",
    agent=analyst,
    output_pydantic=RiskAssessment
)

# --- Run Crew ---
if st.button("Run Analysis"):
    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )
    try:
        result = crew.kickoff()
        assessment = result.pydantic

        st.success("✅ Analysis Complete!")
        st.metric("📊 Risk Score", f"{assessment.risk_score}/10")
        st.subheader("📝 Summary")
        st.write(assessment.risk_summary)
        st.subheader("⚠️ Key Risk Factors")
        for i, factor in enumerate(assessment.risk_factors, 1):
            st.write(f"{i}. {factor}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
