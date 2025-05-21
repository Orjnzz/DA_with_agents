from agno.agent import Agent, RunResponse
import json

# Load prompts from JSON file
def load_prompts():
    with open('./src/prompts.json', 'r') as file:
        return json.load(file)

# Agent for describing dataset
class DataDescriberAgent(Agent):
    def __init__(self,model, name="DataDescriberAgent"):
        super().__init__(
            model=model,
            description="You are a masterful interpreter in the field of data analysis, skilled at revealing the insights within datasets. Your task is to create a detailed description of the data's purpose, the meaning behind each column, and notable trends in data.",
            name=name
        )
        self.prompts = load_prompts()

    def load_data(self, dataframe):
        # string_data = '\n'.join([' | '.join(map(str, row)) for row in dataframe.values])
        sample_data = dataframe.sample(n=min(16, len(dataframe))).to_dict(orient='records')
        data_desc = {
            "columns": list(dataframe.columns),
            "stats": dataframe.describe().to_dict(),
            "dataframe": sample_data 
        }
        description = self.generate_description(data_desc)
        return data_desc, description
    
    def generate_description(self, data_desc):
        prompt = self.prompts["data_describer"].format(
            columns=data_desc['columns'],
            stats=data_desc['stats'],
            dataframe=data_desc['dataframe']
        )
        print("- Prompt for Data Description:", prompt)
        desc_response = self.run(prompt).content
        return desc_response

# Agent for generating research questions based on data summaries
class HypothesisGeneratorAgent(Agent):
    def __init__(self, model, name="HypothesisGeneratorAgent"):
        super().__init__(
            model=model,
            description="You are a visionary hypothesis generator, renowned for transforming data summaries and descriptions into compelling questions. Your mission is to generate insightful and thought-provoking questions based on those summaries and descriptions.",
            name=name
        )
        self.prompts = load_prompts()

    def generate_hypotheses(self, summary, description):
        prompt = self.prompts["hypothesis_generator"].format(
            summary=summary,
            description=description
        )
        print("- Prompt for Hypothesis Generation:", prompt)
        response = self.run(prompt).content
        return response.split("\n")

# Agent for answering analytical questions
class ReasonerAgent(Agent):
    def __init__(self, model, name="ReasonerAgent"):
        super().__init__(
            model=model,
            description="You are an expert analytical reasoner, skilled at interpreting dataframes and their summaries to answer analytical questions. Your task is to provide detailed, evidence-based answers to questions by referencing specific data points, explaining trends, and offering meaningful insights based on the provided dataset and its analysis.",
            name=name
        )
        self.prompts = load_prompts()

    def infer_question(self, dataframe, summary, questions):
        sample_data = dataframe.sample(n=min(16, len(dataframe))).to_dict(orient='records')
        prompt = self.prompts["reasoner"].format(
            dataframe=sample_data,
            summary=summary,
            questions=questions
        )
        print("- Prompt for Reasoning:", prompt)
        response = self.run(prompt).content
        return response
    
# Agent for summary analysis 
class SummarizerAgent(Agent):
    def __init__(self, model, name="SummarizerAgent"):
        super().__init__(
            model=model,
            description="You are an expert data synthesis specialist, renowned for transforming intricate data analyses into precise, insightful summaries. Your mission is to distill complex summaries, questions, and answers into a compelling narrative, highlighting key findings, uncovering overarching trends, and delivering actionable insights with unmatched clarity and authority.",
            name=name
        )
        self.prompts = load_prompts()
    def summary_info(self, summary, questions, answers):
        prompt = self.prompts["summarizer"].format(
            summary=summary,
            questions=questions,
            answers=answers
        )
        print("- Prompt for Summary Analysis:", prompt)
        response = self.run(prompt).content
        return response

# Agent for delivering final actionable insights
class InsightAnalystAgent(Agent):
    def __init__(self, model, name="InsightAnalystAgent"):
        super().__init__(
            model=model,
            description="You are an expert data insight analyst, renowned for transforming complex data analyses into actionable insights. Your mission is to distill intricate analyses into clear, concise reports, highlighting key findings, uncovering overarching trends, and delivering actionable insights with unmatched clarity and authority.",
            name=name
        )
        self.prompts = load_prompts()

    def generate_insights(self,description, questions, answers, summary_result):
        prompt = self.prompts["insight_analyst"].format(
            description=description,
            questions=questions,
            answers=answers,
            summary_result=summary_result
        )
        print("- Prompt for Insight Analysis:", prompt)
        response = self.run(prompt).content
        return response
