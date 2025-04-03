FROM continuumio/miniconda3:latest

# Add Conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Set the working directory
WORKDIR /app/rapa

#create an directory for input/output file
RUN mkdir -p /app/output


# Copy the environment.yml file
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml 

RUN echo "conda activate prot_state" >> ~/.bashrc


# Activate the Conda environment
ENV PATH="/opt/conda/envs/prot_state/bin:$PATH"


# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "prot_state", "/bin/bash", "-c"]

# Activate the Conda environment
ENV PATH="/opt/conda/envs/prot_state/bin:$PATH"


# Copy the application code
COPY . .

# Set the entrypoint to run your application
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "prot_state", "python", "protonate.py"]
