import os
from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai import LLM
import streamlit as st

# ========================== STREAMLIT UI ==========================
st.title("🕵️‍♂️ AI-Powered Fraud Risk Detector")
customer_name = st.text_input("Customer Name", "TechCorp Solutions")
industry = st.text_input("Industry Sector", "AI Software")
description = st.text_area("Company Summary", "A fast-growing AI software company...")

run_analysis = st.button("🔍 Run Fraud Analysis")

# ========================== LLM SETUP ==========================
# Load API key from environment
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
assert openrouter_api_key, "❌ OPENROUTER_API_KEY not found in environment!"

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

# ========================== OUTPUT SCHEMA ==========================
class RiskAssessment(BaseModel):
    risk_score: float = Field(description="Risk score 0-10", ge=0, le=10)
    risk_summary: str = Field(description="Brief risk summary")
    risk_factors: List[str] = Field(description="Top 3 risk factors", min_items=3, max_items=3)

# ========================== AGENT & TASK ==========================
def run_fraud_crew(name, industry, info):
    analyst = Agent(
        role="Senior Fraud Risk Analyst",
        goal=f"Conduct fraud risk analysis for {name} in {industry} sector",
        backstory="Expert in evaluating corporate fraud indicators and compliance risks.",
        tools=[],  # ✅ No tools = no memory needed
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1
    )

    task = Task(
        description=f"""
        Conduct a fraud risk analysis for {name}, a company in the {industry} sector.

        Consider:
        - Common fraud risks in this industry
        - Regulatory compliance concerns
        - Operational or financial red flags
        - Market and competition context

        Provide:
        1. Risk score (0-10)
        2. Summary of risk
        3. 3 specific contributing factors
        """,
        expected_output="A structured JSON risk report with risk_score, risk_summary, and risk_factors.",
        agent=analyst,
        output_pydantic=RiskAssessment
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False  # ✅ safe for Streamlit
    )

    return crew.kickoff()


# ========================== RUN CREW ON CLICK ==========================
if run_analysis:
    with st.spinner("Running fraud analysis..."):
        try:
            result = run_fraud_crew(customer_name, industry, description)

            # Handle both pydantic and raw outputs
            if hasattr(result, 'pydantic'):
                report = result.pydantic
            elif hasattr(result, 'tasks_output') and len(result.tasks_output) > 0:
                report = result.tasks_output[0].pydantic
            else:
                report = result

            # ✅ Display result
            st.success("✅ Analysis Complete!")
            st.markdown(f"### 🎯 Risk Score: `{report.risk_score}/10`")
            st.markdown(f"**📋 Summary:** {report.risk_summary}")
            st.markdown("**⚠️ Top Risk Factors:**")
            for i, factor in enumerate(report.risk_factors, 1):
                st.markdown(f"- {i}. {factor}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
