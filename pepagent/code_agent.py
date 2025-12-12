import os


class CoderAgent:
    def __init__(self, name, system_message, llm_config):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config

    def generate_code(self, task_description):
        """
        Simulate code generation based on task description.
        This function would ideally integrate with a language model
        to generate Python code based on the user's request.
        """
        # For example, let's say the user asked to plot a function
        if "plot function" in task_description:
            code = """
import matplotlib.pyplot as plt
import numpy as np

# Define the function
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot the function
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Function Plot')
plt.grid(True)
plt.show()
"""
        else:
            code = "# Code generation failed: Unknown task."

        return code

    def save_code_to_file(self, code, file_name="generated_code.py"):
        """
        Save the generated code to a Python file.
        """
        try:
            with open(file_name, 'w') as f:
                f.write(code)
            print(f"Code successfully saved to {file_name}")
        except Exception as e:
            print(f"Failed to save code: {str(e)}")


# Create the agent
coder_agent = CoderAgent(
    name="Coder",
    system_message="Write Python code based on the task description.",
    llm_config=None  # The LLM config would be passed in an actual setup
)

# Example task description (User asks for a plot)
task_description = "plot function"

# Generate the Python code
generated_code = coder_agent.generate_code(task_description)

# Save the generated code to a file
coder_agent.save_code_to_file(generated_code, "generated_plot_code.py")