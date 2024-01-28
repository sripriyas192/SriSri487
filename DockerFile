FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# One can use virtual environment to install the required packages
RUN python -m venv /venv
RUN /venv/bin/python -m pip install --upgrade pip
RUN /venv/bin/python -m pip install -r requirements.txt

# Copy the FastAPI application code to the container
COPY . .

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Or Install packages separately as well
RUN pip install uvicorn==0.20.0
RUN pip install fastapi
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
RUN pip install pydantic

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "app:app"]
