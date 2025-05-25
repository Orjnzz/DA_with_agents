from src.utils import info_dataframe, preprocess_df
from agno.team.team import Team
from agno.models.deepinfra import DeepInfra
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from src.agents import DataDescriberAgent, DataStatisticAgent, DataCluster
from src.agents import HypothesisGeneratorAgent, ReasonerAgent, DomainExpertAgent, ValidatorAgent, IntegratorAgent, SynthesizerAgent

import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = Gemini(id="gemini-2.0-flash")

file_path = "./data/titanic.csv"
raw_df = pd.read_csv(file_path)
# df = preprocess_df(raw_df,snake_case_columns=False ,drop_columns=['Dấu thời gian'],generate_id=False)
df = preprocess_df(raw_df,snake_case_columns=False,generate_id=False)
inf_df = info_dataframe(df)

# Initialize agents
data_describer_agent = DataDescriberAgent(model)
data_statistic_agent = DataStatisticAgent(model)
data_cluster_agent = DataCluster(model)
hypothesis_generator_agent = HypothesisGeneratorAgent(model)
reasoner_agent = ReasonerAgent(model)
domain_expert_agent = DomainExpertAgent(model)
validator_agent = ValidatorAgent(model)
integrator_agent = IntegratorAgent(model)
synthesizer_agent = SynthesizerAgent(model)

#Process the data
description_report = data_describer_agent.generate_description(inf_df)
statistic_report = data_statistic_agent.generate_statis(inf_df)
cluster_report = data_cluster_agent.generate_cluster(inf_df)

insight_generation_team = Team(
    name="Insight Generation Team",
    mode="coordinate",
    model=model,
    members=[
        hypothesis_generator_agent,
        reasoner_agent,
        domain_expert_agent,
        # validator_agent,
        integrator_agent,
        synthesizer_agent,
    ],
    instructions=[
        "Coordinate the team of agents systematically to generate actionable insights from the provided DataFrame, statistics, and detailed data reports.",
        "1. Begin by tasking HypothesisGenerator with analyzing the DataFrame, basic statistics, and three reports to generate insightful hypotheses.",
        "2. Pass the generated hypotheses, along with initial DataFrame and reports, to Reasoner to provide thorough evidence-based answers.",
        "3. Send the Reasoner's answers, hypotheses, and initial data inputs to DomainExpert for detailed domain-specific context and insights.",
        "4. Forward all outputs, including initial inputs and prior results, to Integrator to deliver a cohesive overview that supports high-level decision-making",
        "5. Finally, provide all previous outputs to Synthesizer to summarize findings into a comprehensive, actionable report.",
        "Ensure each agent's output, combined with initial inputs, is accurately passed to the next agent in the sequence."
    ],
    success_criteria="The team has generated a comprehensive synthesis report with actionable insights. The report excludes trivial details, maintains a maximum length of 8 lines, and focuses on insights with real decision-making value.",
    markdown=True,
    show_tool_calls=False,
    show_members_responses=True,
    enable_agentic_context=True,
    add_datetime_to_instructions=True,
    expected_output="Return the Synthesize Agent Response"
    # debug_mode=True,
)

# Function to run the team
def run_insight_generation(inf_df, description_report, statistic_report, cluster_report):
    task = f"""Generate actionable insights by analyzing the provided dataset summary and reports. Use the 
                dataset info: {inf_df}, 
                description report: {description_report},
                statistical report: {statistic_report}, 
                and clustering report: {cluster_report} to generate hypotheses, reason through them, and synthesize a comprehensive report with actionable insights."""
    insights_generation_debug = insight_generation_team.print_response(
        task,
        stream=True,
        stream_intermediate_steps=True
    )

# Run the insight generation process
if __name__ == "__main__":
    print("Starting insight generation process...")
    insights = run_insight_generation(inf_df=inf_df,
                                      description_report=description_report,
                                      statistic_report=statistic_report,
                                      cluster_report=cluster_report)
    print("The best insights generation is the final response of Synthesizer Agent!")
