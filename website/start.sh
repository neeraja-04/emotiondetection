#!/bin/bash
# Start the Gunicorn server
gunicorn app:app --bind 0.0.0.0:$PORT
