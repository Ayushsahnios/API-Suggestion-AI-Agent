# agents.py
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

problem_solver_agent = Agent(
    role='Problem Solver',
    goal='Understand the user requirement and decompose it into clear technical subtasks for a software application.',
    backstory=(
        "You are a software architect. "
        "Your mission is to deeply understand user goals "
        "and break them down into small, clear, actionable development tasks."
    ),
    llm=llm,
    allow_delegation=True
)

print(f"Problem Solver Agent created: {problem_solver_agent.role}")

api_finder_agent = Agent(
    role='API Finder',
    goal='Identify the best suitable Sikka APIs needed to fulfill the development tasks.',
    backstory=(
        "You are an API Developer with access to a smart search system (RAG). "
        "Your mission is to find the correct API endpoints that match each task perfectly."
    ),
    llm=llm,
    allow_delegation=True
)

print(f"API Finder Agent created: {api_finder_agent.role}")

code_writer_agent = Agent(
    role='Code Writer',
    goal='Generate complete frontend (HTML/CSS/JS) and backend (Python Flask) code that integrates the selected APIs.',
    backstory=(
        "You are a full-stack engineer. "
        "You write clean, working code based on the development tasks and available APIs. "
        "Split code into index.html, style.css, app.js, and server.py as needed."
    ),
    llm=llm,
    allow_delegation=False
)

print(f"Code Writer Agent created: {code_writer_agent.role}")


def test_agent(agent, message):
    print(f"\n🧠 Testing Agent: {agent.role}")

    test_task = Task(
        description=message,
        agent=agent,
        expected_output="A clear and concise response."
    )
    crew = Crew(
        agents=[agent],
        tasks=[test_task],
        verbose=True  
    )

    result = crew.kickoff()

    print(f"Response:\n{result}\n")


if __name__ == "__main__":
    test_agent(problem_solver_agent, "I want to build a payment system like Nadapayments. What tasks should I do?")
    test_agent(api_finder_agent, "Find the best Sikka APIs for processing credit card payments.")
    test_agent(code_writer_agent, "Write a simple Flask backend with an HTML form to collect payment info.")
