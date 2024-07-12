import os
from textwrap import dedent
from crewai import Agent
from llama_index.core import Settings
from tools import Scraper, Fallacies
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
Settings.llm = OpenAI(model="gpt-3.5-turbo")
class ArgumentationAgents:


    def scholar_scraper_agent(self):
        return Agent(
            role="Scholar Scraper",
            goal=dedent("""\
                Scrape and summarize relevant text from Google Scholar
                to gather evidence to debate the agent """),
            backstory=dedent("""\
                You are an expert in navigating academic databases, 
                adept at finding and summarizing pertinent scholarly articles."""),
            tools=[
                Scraper.scrape_data
            ],        
            verbose=True
        )

    def fallacy_detector_agent(self):
        return Agent(
            role=" Refutation former",
            goal=dedent("""\
                Identify and utilize refutations of the argument given examples of some possible counters and rebuttals detected through evidence given.
                Find  rebuttals with citations as long as the argument is sufficent for you to do so."""),
            backstory=dedent("""\
                As a seasoned logician, you specialize in spotting flaws in their argument through logic and evidence given to you
                and strengthening arguments finding refutations  of the argument through a given format"""),
            tools=[
                Fallacies.find_fallacies
            ],
        
            verbose=True
        )

    def argumentation_agent(self):
        return Agent(
            role="Argumentation Expert",
            goal=dedent("""\
                Formulate a compelling counter argument to the given argument by incorporating evidence and ensuring 
                logical coherence, given fallacies, rebuttals and evidencal data from past agents. Furthermore, give an improved argument from the original one, whilst giving criticism on the argument based on certain criteria"""),
            backstory=dedent("""\
                You are a master of rhetoric and logic, skilled in crafting persuasive 
                arguments and critically analyzing existing ones. You find problems and are a master debater"""),
            
            verbose=True
        )
