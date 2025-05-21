from dotenv import load_dotenv
from src.utils import preprocess_df
from src.agents import DataDescriberAgent, HypothesisGeneratorAgent, ReasonerAgent, SummarizerAgent, InsightAnalystAgent 
from agno.models.deepinfra import DeepInfra
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini

import pandas as pd
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = Gemini(id="gemini-2.0-flash")

file_path = "./files/data.xlsx"
raw_df = pd.read_excel(file_path)
df = preprocess_df(raw_df,snake_case_columns=False ,drop_columns=['Dấu thời gian'],generate_id=False)

# Initialize agents
data_describer_agent = DataDescriberAgent(model)
hypothesis_generator_agent = HypothesisGeneratorAgent(model)
reasoner_agent = ReasonerAgent(model)
summarizer_agent = SummarizerAgent(model)
insight_analyst_agent = InsightAnalystAgent(model)

# Load data and generate summary and description
print("Loading data and generating summary...")
summary, description = data_describer_agent.load_data(df)
questions = hypothesis_generator_agent.generate_hypotheses(summary, description)
answers = reasoner_agent.infer_question(df, summary, questions)
sum_result = summarizer_agent.summary_info(summary, questions, answers)
insight_result = insight_analyst_agent.generate_insights(summary, questions, answers, sum_result)
print("- Insight Result:", insight_result)

#save results to JSON
result = {
    "summary": summary,
    "description": description,
    "questions": questions,
    "answers": answers,
    "sum_result": sum_result,
    "insight_result": insight_result
}

import json
result_file_path = "./files/result2.json"
with open(result_file_path, "w", encoding="utf-8") as result_file:
    json.dump(result, result_file,
              ensure_ascii=False,
              indent=4,
              default=str)
print(f"Results saved to {result_file_path}")
