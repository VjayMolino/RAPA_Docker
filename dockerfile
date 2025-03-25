FROM  ubuntu:22.04

RUN apt-get update && \
    apt-get install -y \
      curl \
      bash \
      libgl1-mesa-glx \
      libegl1-mesa \
      libxrandr2 \
      libxrandr2 \
      libxss1 \
      libxcursor1 \
      libxcomposite1 \
      libasound2 \
      libxi6 \
      libxtst6 \
      ;
# Download Anaconda installer
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh 

# Install Anaconda in batch mode
RUN bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /opt/anaconda3


# Add Conda to PATH
ENV PATH="/opt/anaconda3/bin:$PATH"

# Set the working directory
WORKDIR /app/rapa

#create an directory for input/output file
RUN mkdir -p /app/output


# Copy the environment.yml file
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml 

RUN echo "conda activate prot_state" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Activate the Conda environment
ENV PATH="/opt/conda/envs/my_env/bin:$PATH"



# Copy the application code
COPY . .

# Set the entrypoint to run your application
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "prot_state", "python", "protonate.py"]
