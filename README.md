<div align="center">
<img src=".github/logo.png" alt="DevAssistant Logo"/>
</div>

**Dev Assistant** is a task driven autonomous agent developed using Python, OpenAI API, Pinecone, and Langchain, which is designed to assist programmers in accomplishing various tasks.

The agent operates on the basis of input objectives provided by the user, and it employs a range of tools to achieve the desired outcome.

The tool is particularly useful for tasks that result in the creation of multiple files upon completion, and it is designed to streamline the workflow of developers.

## Purpose

The main goal of this project is to complete tasks using the assistant's knowledge and a set of available tools. The assistant is designed to be self-critical and make extensive use of the file system for project state management. It ensures that each task is fully completed before moving on to the next one.

## Setup

To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Set up the necessary environment variables in a `.envrc` file. You will need to provide your OpenAI API key, Pinecone API key, Pinecone environment, and Pinecone index name.
4. Run the project using the command `python -u -m main`.

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

## Contributing

If you'd like to contribute to the project, feel free to submit a pull request or open an issue on the repository.

## Links

- [Auto-GPT](https://github.com/Torantulino/Auto-GPT)
- [babyAGI](https://github.com/yoheinakajima/babyagi)
- [Llama Index](https://github.com/jerryjliu/llama_index)
- [langchain](https://github.com/hwchase17/langchain)

## Future Improvements

*   Integrating a security/safety agent to ensure ethical considerations are met and prevent potential misuse.
*   Implementing task sequencing and parallel tasks for improved efficiency.
*   Adding interim milestones to track progress more effectively.
*   Providing real-time priority updates for better resource allocation.

## Risks & Safety Considerations

- Data privacy and security: Ensuring user data is protected and not misused.
- Ethical concerns: Ensuring the AI system does not engage in unethical activities.
- Dependence on model accuracy: Ensuring the AI system's performance is reliable and accurate.
- System overload: Preventing the AI system from becoming overwhelmed by too many tasks. ️
- Misinterpretation of task prioritization: Ensuring the AI system correctly understands and prioritizes tasks.

## License

This project is licensed under the MIT License.
