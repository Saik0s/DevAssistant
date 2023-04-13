# 🤖 Dev Assistant

Dev Assistant is a Python-based project that uses the OpenAI API, Pinecone, and Langchain to create an intelligent agent that can help developers with their tasks. The agent takes an objective as input and orchestrates various modules to achieve the goal.

## 📁 Project Structure

The main entry point of the project is `main.py`, which initializes the necessary components and runs the agent with the given objective.

## 💻 Usage

To run the project, first set the required environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `PINECONE_API_KEY`: Your Pinecone API key.
- `PINECONE_ENVIRONMENT`: The Pinecone environment to use.
- `PINECONE_INDEX_NAME`: The Pinecone index name to use.

Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Then, execute `main.py`:

```bash
python main.py
```

You will be prompted to enter an objective for the agent. For example:

```plaintext
Please enter the objective: Create a simple project that uses next.js and python for the backend. It should have a chat interface and subscribe to new messages in real time.
```

The agent will then work on the objective, and you can observe its progress in the console output.

## 🌟 Starting a Local Searxng Instance

Searxng is a privacy-respecting, hackable metasearch engine. It is used as a tool of the execution agent. To start a local Searxng instance, follow these steps:

1.  Visit the [Searxng GitHub repository](https://github.com/searxng/searxng).
2.  Follow the instructions in the README file to set up and configure Searxng on your local machine.
3.  Ensure that `searxng/settings.yml` contains these values:

```yaml
server:
  limiter: false
general:
  debug: true
search:
  formats:
    - html
    - json
```
4.  Start the Searxng instance as instructed in the repository.

## 🔄 Workflow

The workflow for DevAssistant consists of the following steps:

1.  Receive the objective as input.
2.  Pre-process the objective using the perception module to extract relevant features.
3.  Use the reasoning module to decompose the objective into smaller sub-goals and tasks.
4.  Prioritize tasks and sub-goals based on the AGI's understanding of the problem and the available resources.
5.  Execute tasks using the execution module, updating the memory module with the results.
6.  Monitor progress towards the objective and evaluate performance using the learning module.
7.  If the objective is not yet achieved, adjust the approach based on the learning module's feedback and generate new tasks.
8.  Repeat steps 4-7 until the objective is achieved or a stopping condition is reached.
9.  Communicate the results and any relevant insights to the user.

🔮 Future Improvements
----------------------

Some suggested future improvements for DevAssistant include:

*   Integrating a security/safety agent to ensure ethical considerations are met and prevent potential misuse.
*   Implementing task sequencing and parallel tasks for improved efficiency.
*   Adding interim milestones to track progress more effectively.
*   Providing real-time priority updates for better resource allocation.

🚧 Risks & Safety Considerations
--------------------------------

Key risks associated with DevAssistant are:

- Data privacy and security: Ensuring user data is protected and not misused.
- Ethical concerns: Ensuring the AI system does not engage in unethical activities.
- Dependence on model accuracy: Ensuring the AI system's performance is reliable and accurate.
- System overload: Preventing the AI system from becoming overwhelmed by too many tasks. ️
- Misinterpretation of task prioritization: Ensuring the AI system correctly understands and prioritizes tasks.

Addressing these risks is crucial for the successful application of DevAssistant.

# 💡 Inspirations and References

- [Auto-GPT](https://github.com/Torantulino/Auto-GPT)
- [babyAGI](https://github.com/yoheinakajima/babyagi)
- [Llama Index](https://github.com/jerryjliu/llama_index)
- [langchain](https://github.com/hwchase17/langchain)

## 📄 License

This project is licensed under the MIT License.
