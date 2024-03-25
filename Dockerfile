# Use the official Python 3.10 image as a base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements_deploy.txt file from the project's root directory to the container's working directory
COPY requirements_deploy.txt /app/requirements_deploy.txt

# Install the Python dependencies using pip
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy the deployment files from the project root/deployment to the container's working directory
# This will include the model .pth file. TODO later - get this from a dvc pull command instead
COPY ./deployment /app/deployment

# Set the working directory to the deployment directory
WORKDIR /app/deployment

# Expose the port on which your Gradio app runs
EXPOSE 8080

# Command to run the Gradio app when the container starts
CMD ["python", "gradio_app.py"]
