#!/bin/bash

source bin/setvars.sh
# Start the server
uvicorn main:app --port 8000 --host 0.0.0.0 --workers 1