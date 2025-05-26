from agno.team.team import Team
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
import pandas as pd
import os
import time
import requests
# from src.utils import preprocess_df
# from dotenv import load_dotenv
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# model = Gemini(id="gemini-2.0-flash")

# file_path = "./data/data.csv"
# raw_df = pd.read_csv(file_path)
# df = preprocess_df(raw_df,snake_case_columns=False ,drop_columns=['Dấu thời gian'],generate_id=False)


class DataDescriberAgent(Agent):
    def __init__(self, model, name="DataDescriberAgent"):
        super().__init__(
            name=name,
            model=model,
            description="You are a meticulous curator in the field of data analysis, skilled at mapping out the intricate details of datasets. responsible for providing a comprehensive overview of the dataset, explicitly detailing the specific field or context of the survey data (e.g., customer satisfaction, employee engagement) and describing each column's attributes, including what it represents, its data type, and its role in the dataset. It also highlights data quality issues, such as missing values or outliers, to set the stage for deeper analysis.",
            instructions= """Analyze input information and generate a detailed description that reveals the dataset's key characteristics, structure, and potential insights. Address the following points concisely but thoroughly:
                        1. Purpose and Subject Matter: Analyze the summary information to Identify the dataset's main theme and purpose from its columns and data.
                        2. Column Analysis: For each column in the dataset, provide a detailed description of its meaning, data type, and significance. Highlight any unique or interesting aspects of the columns.
                        3. Trends and Patterns: Based on the summary statistics and the dataframe, identify any notable trends, patterns, or anomalies in the data. Discuss how these trends might relate to the dataset's purpose.
                        4. Potential Insights: Suggest potential insights or conclusions that can be drawn from the dataset based on the summary information and your analysis.""",
            expected_output= "Provide only a summary paragraph interpreting the dataset's characteristics."
            )
    def generate_description(self, data_desc):
        tasks = (f"You are provided a sample dataframe the data and basic statistics {data_desc}. Your task is to produce a detailed data description report."
        )
        for attempt in range(3):
            try:
                response = self.run(tasks).content
                # self.print_response(response, markdown=True)
                return response
            except (requests.ConnectionError, requests.Timeout, OSError) as e:
                if attempt < 3 - 1:
                    time.sleep(2)
                else:
                    raise e

# data_describer_agent = DataDescriberAgent(model=model)
# test = data_describer_agent.generate_description(data)
# print(test)

class DataStatisticAgent(Agent):
    def __init__(self, model, name="DataStatisticAgent"):
        super().__init__(
            name=name,
            model=model,
            description="You are an insightful detective in the field of data analysis, skilled at uncovering hidden relationships and testing hypotheses. Responsible for conducting in-depth statistical analyses to reveal patterns, test assumptions, and provide actionable insights from the dataset.",
            instructions="""Analyze the provided dataset summary, including a sample dataframe and basic statistics, to generate a statistical analysis report that uncovers key relationships and validates hypotheses. Address the following concisely yet thoroughly:
                        1. Identify critical variables and formulate relevant hypotheses aligned with the dataset's purpose.
                        2. Apply suitable statistical tests (e.g., t-tests, ANOVA, chi-square) to test hypotheses, interpreting results with p-values and confidence intervals.
                        3. Calculate correlations among numerical variables and explain significant relationships.
                        4. Build and assess a regression or classification model, reporting metrics like R-squared or accuracy.
                        5. Summarize findings and propose actionable implications.""",
            expected_output="Provide only a statistical analysis report with a summary paragraph, detailed number test results, correlations, model evaluation, and actionable insights about 6-8 lines long.")
    def generate_statis(self, data_desc):
        tasks = f"You are provided with a dataset summary {data_desc}, including a sample dataframe and basic statistics. Your task is to generate a statistical analysis report."
        for attempt in range(3):
            try:
                response = self.run(tasks).content
                # self.print_response(response, markdown=True)
                return response
            except (requests.ConnectionError, requests.Timeout, OSError) as e:
                if attempt < 3 - 1:
                    time.sleep(2)
                else:
                    raise e
                
# data_statistic_agent = DataStatisticAgent(model=model)
# statis_report = data_statistic_agent.generate_statis(data)
# print(statis_report)

class DataCluster(Agent):
    def __init__(self, model, name="DataCluster"):
        super().__init__(
            name=name,
            model=model,
            description="You are a strategic organizer in the field of data analysis, skilled at identifying and categorizing data into meaningful clusters. Responsible for segmenting the dataset into distinct groups to reveal underlying patterns and guide targeted strategies.",
            instructions="""Analyze the provided dataset summary (including a sample dataframe and basic statistics) and generate a clustering report that segments the data into meaningful groups. Address the following points concisely but thoroughly:
                            1. Data Preparation: Describe steps to prepare the data for clustering (e.g., handling missing values, encoding, scaling).
                            2. Clustering Methodology: Select an appropriate clustering algorithm (e.g., K-means, DBSCAN) and justify the choice. Determine the optimal number of clusters using methods like the elbow method or silhouette score.
                            3. Cluster Profiles: Analyze and describe each cluster's characteristics using summary statistics or centroids.
                            4. Insights and Strategies: Interpret the clusters and propose actionable strategies based on the segmentation.""",
            expected_output="Provide only a clustering analysis report with a summary paragraph detailed cluster characteristics, and actionable insights about 5-6 lines long.")
        
    def generate_cluster(self, data_desc):
        tasks = f"You are provided with a dataset summary {data_desc}, including a sample dataframe and basic statistics. Your task is to generate a clustering analysis report."
        for attempt in range(3):
            try:
                response = self.run(tasks).content
                # self.print_response(response, markdown=True)
                return response
            except (requests.ConnectionError, requests.Timeout, OSError) as e:
                if attempt < 3 - 1:
                    time.sleep(2)
                else:
                    raise e
                
# data_cluster_agent = DataCluster(model=model)
# cluster_report = data_cluster_agent.generate_cluster(data)
# print(cluster_report)

# ------ INSIGHT GENERATION TEAM ------

class HypothesisGeneratorAgent(Agent):
    def __init__(self, model, name="HypothesisGeneratorAgent"):
        super().__init__(
            name=name,
            model=model,
            role="You are a visionary hypothesis generator, renowned for transforming data summaries into compelling questions and hypotheses. Your role is to propose insightful and testable hypotheses based on the provided reports.",
            instructions="""Using the dataset description, statistical report, and clustering report, 5 insightful and answerable questions to support an analyst in understanding key patterns and deriving actionable insights from a dataset, applicable to any domain. Address the following points concisely but thoroughly:
                            1. Pattern Identification: Identify patterns or anomalies in the data that warrant investigation.
                            2. Relationship Exploration: Propose hypotheses about potential relationships between variables or clusters.
                            3. Impact Assessment: Suggest hypotheses regarding the impact of certain variables on outcomes.
            Expected Output: "Generate only a list of 5 hypotheses based on the provided dataset description, statistical report, and clustering report. Each hypothesis should be clear, testable, and relevant to the data.""",
        )

class ReasonerAgent(Agent):
    def __init__(self, model, name="ReasonerAgent"):
        super().__init__(
            name=name,
            model=model,
            role="You are an expert analytical reasoner, skilled at interpreting dataframes and their summaries to answer analytical questions. Your task is to provide concise, evidence-based answers to questions by referencing specific data points, explaining trends, and offering meaningful insights based on the provided dataset and its analysis.",
            instructions="""Analyze the provided dataset summary, statistical report, clustering report, and hypotheses to answer analytical questions. Address the following points concisely but thoroughly:
                            1. Data Interpretation: Interpret the dataset and reports to provide clear answers to the questions.
                            2. Evidence-Based Responses: Support your answers with specific data points or trends from the dataset.
                            3. Insightful Explanations: Offer meaningful insights and directly addresses the question based on the analysis.
            Expected Output: Provide concise, evidence-based answers to each hypothesis using the provided DataFrame and reports. Each answer should include hypothesis testing results and trend explanations.""",
        )

class DomainExpertAgent(Agent):
    def __init__(self, model, name="DomainExpertAgent"):
        super().__init__(
            name=name,
            model=model,
            role="You are a domain expert, renowned for providing context and enhancing the applicability of data analysis findings. Your role is to relate the analytical results to real-world scenarios and industry knowledge.",
            instructions="""Using the dataset description, statistical report, clustering report, hypotheses, and answers, provide domain-specific context to the findings. Address the following points concisely but thoroughly:
                            1. Industry Alignment: Explain how the findings align with known trends or theories in the domain.
                            2. Practical Applications: Suggest practical implications of the insights (e.g., business strategies).
            Expected Output: Provide briefly a summary paragraph about domain-specific context to the analytical findings, including industry alignment and practical applications.""",
        )

# CONSIDER FOR USING THIS AGENT
class ValidatorAgent(Agent):
    def __init__(self, model, name="ValidatorAgent"):
        super().__init__(
            name=name,
            model=model,
            role="You are an expert in validating data analyses, ensuring robustness and preventing misleading conclusions. Your role is to critically evaluate the analysis for potential weaknesses or biases.",
            instructions="""Using all provided reports and outputs, evaluate the analysis for potential issues. Address the following points concisely but thoroughly:
                            1. Result Verification: Check the accuracy of the findings against the data.
                            2. Methodological Soundness: Evaluate the appropriateness of the methods used in the analysis.
                            3. Confounding Variables: Identify any variables not considered that could affect the results.
            Expected output: "Generate a validation report highlighting any issues in the analysis, such as statistical validity, confounding variables, and generalizability.""",
        )

class IntegratorAgent(Agent):
    def __init__(self, model, name="IntegratorAgent"):
        super().__init__(
            name=name,
            model=model,
            role="You are an executive-level communicator, skilled at distilling complex analytical results into a unified, decision-focused message. Your role is to integrate all analytical outputs into a clear, high-level synthesis that enables stakeholders to quickly grasp the overall implications.",
            instructions="""Review all provided reports, hypotheses, answers and domain contextfindings. Craft a synthesis of 5-6 lines that:
            - Emphasizes overarching implications and strategic direction, not just summarizing findings.
            - Ensure the output does not include any new information or suggestions beyond what is already provided in the context.
            - Frames insights in terms of impact and next steps for decision-makers.
            - Avoids repeating detailed facts or technical explanations.
            - Ensures the message is cohesive, actionable, and tailored for executive understanding.""",
            expected_output="Produce a synthesis report of 5-6 lines that communicates the overall impact and strategic significance of the analysis for decision-makers.",
        )

class SynthesizerAgent(Agent):
    def __init__(self, model, name="SynthesizerAgent"):
        super().__init__(
            name=name,
            model=model,
            role="You are a synthesis expert, skilled at integrating diverse analytical findings into a cohesive narrative. Your role is to create a comprehensive report that summarizes the analysis and provides actionable insights.",
            instructions="""Using all provided reports, hypotheses, answers, domain context, and validation report, synthesize the information into a comprehensive summary. 
                            The summary should be gather around 6-8 lines long, highlighting the most important points in a realistic and actionable way. 
                            It should enable decision-makers to clearly envision potential strategies and outcomes.\nYour summary must follow these instructions:\n
                            - Do not exceed 8 lines in your summary.\n
                            - Do not add new information or recommendations not present in the context.\n
                            - Avoid listing facts; instead, deduce high-level patterns and insights from the key data.\n
                            - Use important data points to support the inferred insights.\n
                            - Do not focus on specific entities unless they illustrate a broader pattern.\n
                            - Ensure the summary is actionable—offering strategic value, not just general observations.\n
                            - Exclude any trivial or unimportant information.\n- You do not need to include all available information—only the most significant parts.\n
                            IMPORTANT: Double-check that your final insight fully follows the above instructions!""",
            # expected_output="Generate a synthesis report that includes key findings and an overall narrative connecting all parts of the analysis.",
        )
