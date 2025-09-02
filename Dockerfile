# Dockerfile

# Use an official Python runtime as a parent image.
# This is a multi-platform image, so Docker on M1 will pull the arm64 version.
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt