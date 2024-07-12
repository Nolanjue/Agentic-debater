from dotenv import load_dotenv
load_dotenv()

from textwrap import dedent
from crewai import Agent, Crew

from tasks import ArgumentationTasks
from agents import ArgumentationAgents

tasks = ArgumentationTasks()
agents = ArgumentationAgents()

print("## Welcome to the Argumentation Crew")
print('-------------------------------')
query = input("What is the opposing argument's topic that you would need to attack?\n")
argument = input("What is your argument you want to test?\n")

# Create Agents
scholar_scraper_agent = agents.scholar_scraper_agent()
fallacy_detector_agent = agents.fallacy_detector_agent()
argumentation_agent = agents.argumentation_agent()

# Create Tasks
scrape_scholar = tasks.scrape_google_scholar(scholar_scraper_agent, query, argument)
detect_fallacies = tasks.detect_fallacies(fallacy_detector_agent, argument, scrape_scholar)

#use the details from scrape scholar and detect fallacies to form argument as your final response
form_argument = tasks.form_argument(argumentation_agent, scrape_scholar, detect_fallacies,  argument)


argumentation_crew = Crew(
    agents=[
        scholar_scraper_agent,
        fallacy_detector_agent,
        argumentation_agent
    ],
    tasks=[
        scrape_scholar,
        detect_fallacies,
        form_argument
    ],
    verbose=True
)

final_argument = argumentation_crew.kickoff()

# Print results
print("\n\n########################")
print("## Here is the result")
print("########################\n")
print("counterargument and conclusions:")
print(final_argument)


