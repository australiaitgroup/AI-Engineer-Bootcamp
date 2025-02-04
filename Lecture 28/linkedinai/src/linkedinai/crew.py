from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

@CrewBase
class Linkedinai():
    """Linkedinai crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @before_kickoff
    def pull_data_example(self, inputs):
        inputs['extra_data'] = "This is extra data"
        return inputs

    @after_kickoff
    def log_results(self, output):
        print(f"Results: {output}")
        return output

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[SerperDevTool()],  # Using the DuckDuckGo search tool
            verbose=True,
            max_iter=3
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            max_iter=3
        )

    @agent
    def content_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['content_creator'],
            verbose=True,
            max_iter=3
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
        )

    @task
    def linkedin_post_task(self) -> Task:
        return Task(
            config=self.tasks_config['linkedin_post_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Linkedinai crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=ChatOpenAI(model="gpt-4o",temperature=0.5),
            # chat_llm=ChatOpenAI(model="gpt-4o",temperature=0.5),
            verbose=True,
        )

# crewai train -n 3
# subgent - output -> prompt back to user ask for feedback -> user provide feedback to subagent/manager -> update output accordingly 
# save to pkl (memeory about above conversation)
# crewai run - load pkl -> generate content based your chat history -> allign user perference.
