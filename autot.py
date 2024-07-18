import os
import tempfile
from autogen import ConversableAgent, AssistantAgent
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv

def run_code_task(prompt):
    # Load environment variables
    load_dotenv()
    
    # Replace with your actual API key
    api_key = os.getenv("OPENAI_API_KEY")

    # Directory to store code files
    coding_dir = "coding"
    if not os.path.exists(coding_dir):
        os.makedirs(coding_dir)

    # Clean the temporary directory
    temp_dir = tempfile.TemporaryDirectory(dir=coding_dir)

    # Create a local command line code executor
    executor = LocalCommandLineCodeExecutor(
        timeout=10,  # Timeout for each code execution in seconds.
        work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
    )

    # Create the code executor agent
    code_executor_agent = ConversableAgent(
        name="CodeExecutorAgent",
        system_message="You execute code provided to you.",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={"executor": executor},  # Use the local command line code executor.
        human_input_mode="NEVER",  # Always take human input for this agent for safety.
    )

    # Create the code writer agent
    code_writer_agent = AssistantAgent(
        name="CodeWriterAgent",
        llm_config={"config_list": [{"model": "gpt-4", "api_key": api_key}]},
        code_execution_config=False,  # Turn off code execution for this agent.
    )

    def execute_task(prompt):
        # Clean the temporary directory at the beginning
        for file in os.listdir(temp_dir.name):
            file_path = os.path.join(temp_dir.name, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Start the conversation with the initial prompt
        chat_result = code_executor_agent.initiate_chat(
            recipient=code_writer_agent,
            message=f"Identify if this problem needs to be graphed, computed, or solved: {prompt}. "
                    "If it needs to be graphed, write a Python script to graph it and save it as 'plot.png'. "
                    "Do not display the plot. "
                    "If it needs to be computed or solved, write a Python script to compute or solve it, and save the result in 'result.txt'. "
                    "Terminate after performing the required action."
                    "The problem may contain mathematical expressions, equations, or functions, and may contain information outside the problem, identify what is relevant and then proceed with the task.",
            max_turns=5,  # Adjust the number of turns as needed
            summary_method="last_msg"
        )

        # Print the conversation history
        for message in chat_result.chat_history:
            print(f"{message['role']}: {message['content']}")

        # Check for plot.png or result.txt
        result = None
        if 'plot.png' in os.listdir(temp_dir.name):
            result = os.path.join(temp_dir.name, 'plot.png')
        elif 'result.txt' in os.listdir(temp_dir.name):
            with open(os.path.join(temp_dir.name, 'result.txt'), 'r') as file:
                result = file.read()
        
        # Clean up the temporary directory
        for file in os.listdir(temp_dir.name):
            file_path = os.path.join(temp_dir.name, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        return result

    result = execute_task(prompt)
    return result

def agent_math(prompt):
    result = run_code_task(prompt)
    return result

# Example usage
if __name__ == "__main__":
    prompt = "2x + 3y = 6"
    result = agent_math(prompt)
    print(result)
