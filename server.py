# server.py
import zipfile
from io import BytesIO


from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import os

from agents import problem_solver_agent, code_writer_agent
from rag_system import search_apis

from langchain.chat_models import ChatOpenAI
from crewai import Task, Crew

latest_zip_buffer = None

load_dotenv()

app = Flask(__name__)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2
)


def run_problem_solver(user_query):
    """Ask the Problem Solver agent to break down the user query."""
    print("Running Problem Solver Agent...")

    problem_solver_task = Task(
        description=f"Understand and break down this project: {user_query}",
        expected_output="A clear list of subtasks needed to build the project.",
        agent=problem_solver_agent
    )

    crew = Crew(
        agents=[problem_solver_agent],
        tasks=[problem_solver_task]
    )

    result = crew.kickoff()
    return result.tasks_output[0].raw 
def run_code_writer(subtask_text, api_info_summary):
    """Use Code Writer agent to generate code."""
    print("Running Code Writer Agent...")

    code_writer_task = Task(
        description=f"Write frontend and backend code for:\n{subtask_text}\n\nUsing these APIs:\n{api_info_summary}",
        expected_output="HTML frontend, JavaScript code, and Flask backend code snippets.",
        agent=code_writer_agent
    )

    crew = Crew(
        agents=[code_writer_agent],
        tasks=[code_writer_task]
    )

    result = crew.kickoff()
    return result.tasks_output[0].raw 


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json['query']
    print(f"User Query Received: {user_query}")

    subtasks_text = run_problem_solver(user_query)

    api_info_list = search_apis(subtasks_text, top_k=3)

    api_info_summary = "\n".join([
        f"{api['api_name']} ({api['endpoint']}): {api['description']}"
        for api in api_info_list
    ])

    generated_code_text = run_code_writer(subtasks_text, api_info_summary)

    output = {
        "subtasks": subtasks_text,
        "api_info": api_info_list,  
        "generated_code": generated_code_text
        
    }
    global latest_zip_buffer
    latest_zip_buffer = save_code_to_files_and_zip(generated_code_text)


    return jsonify(output)

@app.route('/download', methods=['GET'])
def download():
    """Download the latest generated project zip."""
    global latest_zip_buffer
    if latest_zip_buffer:
        return (
            latest_zip_buffer.getvalue(),
            200,
            {
                'Content-Type': 'application/zip',
                'Content-Disposition': 'attachment; filename=generated_project.zip'
            }
        )
    else:
        return "No project available yet!", 404


def save_code_to_files_and_zip(code_text):
    folder_path = "generated_project"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(os.path.join(folder_path, "full_code.txt"), "w", encoding="utf-8") as f:
        f.write(code_text)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname=arcname)

    zip_buffer.seek(0)
    return zip_buffer

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
