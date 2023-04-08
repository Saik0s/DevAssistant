#!/bin/bash

OBJECTIVE="Create all the necessary files for a complete ios app using SwiftUI that allows users to track their exercise activities. The app should allow users to add new activities, edit existing activities, and delete activities. The app should also allow users to view a summary of their activities."
FIRST_TASK="Define the overall project structure and plan development tasks for this objective: $OBJECTIVE"
COLLECTION_NAME="test"

python3 auto.py --objective "$OBJECTIVE" --first_task "$FIRST_TASK" --collection_name "$COLLECTION_NAME"
