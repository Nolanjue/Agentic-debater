from crewai import Task
from textwrap import dedent

class ArgumentationTasks:
    def scrape_google_scholar(self, agent, topic, argument):
        return Task(description=dedent(f"""\
            Scrape Google Scholar for research papers related to the topic: {topic}.
            Summarize the key findings and relevance to the given argument for next steps
            Use the tool to scrape the website given at your disposal, 
            if you believe that the argument is too vague or small, then give an appropiate response.

            Your goal is to focus on gathering evidence that can support or refute the argument.
            Your output should be a  string that is returned from the tool scraper.

            For each tools, you must give the {topic} as a query as the parameter
            to be scraped, so you can hopefully find contradictions of the {argument} within.
            If you read a pdf link, you should only read 5 pages MAXIMUM, dont read the whole thing.
            """),
             expected_output= "A summary of key research papers as evidence related to the query given the output for each tool..",
            agent=agent
        )
    def detect_fallacies(self, agent, argument, evidence):
        return Task(description=dedent(f"""\
            You must use a tool that would find refutations for this argument: {argument}.

            If you find the argument too vague or too small to argue effectively, cater your response appropiately.

            

            This is a tool for you to find rebuttals. DO NOT USE FALLACIES AS YOUR FINAL ANSWER BUT ONLY REBUTTALS AGAINST THEIR CLAIMS.
            Firstly, if there are any sources provided in the argument, be sure to take note and analyze to see their validity.
           
             lastly, you are given a huge load of evidence, be sure, if applicable, to use this to refute certain parts of their 
             argument, add this refutation in the object just like the output from the tool.

            Your final report should include a list of   refutations ONLY   given by the evidence of the 
            and suggestions for how to correct them as a dictionary, it must be in the format as the example given by the tool. 

            Here is all the evidence avaliable to you:
            {evidence}, You do not need to use the scrape tool again, all the evidence is given to you.


            """),
            expected_output= "A list of identified refutations that you find, do not find fallacies as your final answer only use 2 fallacies maximum",
            agent=agent
        )

    def form_argument(self, agent, evidence,  fallacies, argument):
        return Task(description=dedent(f"""\
            Formulate a compelling argument using evidence gathered from Google Scholar.
            Analyze the argument: {argument}, identifying fallacies and incorporating
            evidence to strengthen the argument.

            You must think the person just gave their argument like in a debate, and its your job as a incredible debater to take notes and 
            then respond to their argument with this output argument in that style.

            
            Your final report should be a well-structured argument that includes:
            You must use fallacies to strengthen your argument, but not use it directly as an attack, you need to
            use fallacies to then bring in a refutation and rebuttal to an inaccuracy in their claim.

            - A statement acknowledging certain parts of their view
            -A counterargument that has strong points why their argument is flawed 
            -Key points to establish your argument using evidence.
            - Examples given to you by the fallacies and refutations detected: {fallacies}
            - A final statement with criticisms on how to make the argument better and also certain flaws.
            - Your argument for the opposite side of this argument.
            
            Here is the necessary evidence to form your argument:
            {evidence}, Do not use the scrape tool, it has been used already.
            """),
            expected_output= "A well-structured and compelling argument.",
            agent=agent
        )
