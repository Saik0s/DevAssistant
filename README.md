<div align="center">
<img src=".github/logo.png" alt="DevAssistant Logo"/>
</div>

**Dev Assistant** is a Python project that demonstrates an intelligent agent capable of performing tasks, learning from its environment, and evaluating its progress towards a given objective. The agent is composed of several modules, each responsible for a specific aspect of the agent's behavior.

The agent operates on the basis of input objectives provided by the user, and it employs a range of tools to achieve the desired outcome.

The tool is particularly useful for tasks that result in the creation of multiple files upon completion, and it is designed to streamline the workflow of developers.

## Key Components

- ReasoningModule: Generates and prioritizes tasks based on the agent's objective and current state.
- PerceptionModule: Processes tasks and results to optimize them for the agent's understanding and execution.
- ExecutionModule: Executes tasks using various tools and returns the results.
- LearningModule: Learns from observations and adjusts the agent's behavior to improve efficiency.
- MemoryModule: Stores and retrieves relevant information based on the agent's tasks and objectives.
- EvaluationModule: Evaluates the agent's progress towards its objective and determines if the objective has been achieved.

## How to Use

To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `make install`.
3. Set up the necessary environment variables in a `.envrc` file. You will need to provide your OpenAI API key.
4. Run the project using the command `make docker` or `make`.

## Running the Project

You can run the project in different modes:

- To run the project with a specific objective, use the command `python -u -m main --obj "Your objective here"`.
- To run the project in verbose mode, add the `--verbose` flag to the command.
- To run the project with a visualizer, add the `--visualizer` flag to the command.

<div style="border: 1px solid #ccc; border-radius: 16px; padding: 16px; margin: 16px 0;">
  <p style="color: red; margin: 0; font-size: 1.4em">
    WARNING: </p>The agent is equipped with tools that allow making modifications to the machine where it is currently operating. It is recommended to run agent inside docker container. Run <p style="color: rgb(240, 230, 140); background: rgb(47, 79, 79); display: inline;">make docker</p> command to start a container.
  </p>
</div>

## Tools

The assistant makes use of several tools to complete tasks. Some of these tools include:

- Python REPL
- Bash commands
- File manipulation (read, write, delete, etc.)
- GitHub integration
- Web scraping

## Structure

The project consists of several Python files, each containing a specific module or class:

- AgentOrchestrator.py: Contains the main AgentOrchestrator class, which coordinates the different modules to achieve the agent's objective.
- main.py: The main script that runs the agent and handles command-line arguments.

## Future Improvements

- Improve the agent's ability to handle more complex objectives and tasks.
- Add more tools and capabilities to the ExecutionModule.
- Enhance the agent's learning and adaptation capabilities.
- Implement a visualizer to display the agent's progress and decision-making process.

## Contributing

If you'd like to contribute to the project, feel free to submit a pull request or open an issue on the repository.

## Links

- [Auto-GPT](https://github.com/Torantulino/Auto-GPT)
- [babyAGI](https://github.com/yoheinakajima/babyagi)
- [Llama Index](https://github.com/jerryjliu/llama_index)
- [langchain](https://github.com/hwchase17/langchain)

## License

This project is licensed under the MIT License.
