#!/bin/bash

mkdir -p agi_project/modules
mkdir -p agi_project/data/objectives
mkdir -p agi_project/data/tasks
mkdir -p agi_project/data/memory
mkdir -p agi_project/data/logs
mkdir -p agi_project/utils

touch agi_project/modules/__init__.py
touch agi_project/modules/perception.py
touch agi_project/modules/memory.py
touch agi_project/modules/learning.py
touch agi_project/modules/reasoning.py
touch agi_project/modules/execution.py
touch agi_project/modules/communication.py

touch agi_project/utils/__init__.py
touch agi_project/utils/helpers.py

touch agi_project/config.py
touch agi_project/main.py
touch agi_project/requirements.txt
