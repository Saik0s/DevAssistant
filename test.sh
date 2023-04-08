#!/bin/bash

OBJECTIVE="Develop an iOS app using Swift that allows users to track their daily exercise routines and monitor their progress over time."
FIRST_TASK="Design the app's user interface, including the main screen that displays a summary of the user's exercise activities and a navigation menu for accessing different features of the app."
COLLECTION_NAME="test3"

python3 auto.py --objective "$OBJECTIVE" --first_task "$FIRST_TASK" --collection_name "$COLLECTION_NAME"
