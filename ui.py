import json
import time
import sys
from PySimpleGUI import PySimpleGUI as sg
from index_chat import create_agent, create_vectorstore, read_github
import threading

# Load templates from a JSON file
def load_templates():
    try:
        with open("templates.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Load history from a JSON file
def load_history():
    try:
        with open("history.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Save history to a JSON file
def save_history(history):
    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)

# Redirect stdout to a separate text view
class RedirectStdout:
    def __init__(self, text_output_element):
        self.text_output_element = text_output_element
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout

    def write(self, text):
        self.text_output_element.update(value=self.text_output_element.get() + text)

    def flush(self):
        pass

def execute_request(input, output_element, send_button):
    result = create_agent().run(input)
    output_element.update(result)
    send_button.update(disabled=False)

def main():
    templates = load_templates()
    history = load_history()

    # Define layout
    left_column = [
        [sg.Text("Templates:")],
        [sg.Listbox(values=templates, size=(30, 10), key="-TEMPLATES-", enable_events=True)],
        [sg.Text("History:")],
        [sg.Listbox(values=history, size=(30, 10), key="-HISTORY-", enable_events=True)],
    ]

    middle_column = [
        [sg.Text("Prompt:")],
        [sg.Multiline(size=(60, 10), key="-INPUT-")],
        [sg.Button("Send", key="-SEND-", bind_return_key=True)],
        [sg.Multiline(size=(60, 10), key="-OUTPUT-", disabled=True)],
    ]

    right_column = [
        [sg.Text("URL:")],
        [sg.InputText(key="-URL-"), sg.Button("Set URL", key="-SET_URL-")],
        [sg.Checkbox("Show stdout", enable_events=True, key="-TOGGLE_STDOUT-")],
        [sg.Multiline(size=(30, 20), key="-STDOUT-", visible=False)],
    ]

    layout = [
        [
            sg.Column(left_column),
            sg.Column(middle_column),
            sg.Column(right_column),
        ]
    ]

    window = sg.Window("DevAssistant", layout, use_ttk_buttons=True, ttk_theme="aqua")

    with RedirectStdout(window["-STDOUT-"]):
        while True:
            event, values = window.read()

            if event == sg.WIN_CLOSED:
                break
            elif event == "-TEMPLATES-":
                window["-INPUT-"].update(values["-TEMPLATES-"][0])
            elif event == "-HISTORY-":
                window["-INPUT-"].update(values["-HISTORY-"][0])
            elif event == "-SEND-":
                window["-SEND-"].update(disabled=True)
                window["-OUTPUT-"].update("Loading...")

                request_thread = threading.Thread(target=execute_request, args=(values["-INPUT-"], window["-OUTPUT-"], window["-SEND-"]))
                request_thread.start()

                history.append(values["-INPUT-"])
                save_history(history)
                window["-HISTORY-"].update(history)
            elif event == "-SET_URL-":
                url = values["-URL-"]
                print(f"URL set to: {url}")

            elif event == "-TOGGLE_STDOUT-":
                window["-STDOUT-"].update(visible=values["-TOGGLE_STDOUT-"])

    window.close()

if __name__ == "__main__":
    main()
