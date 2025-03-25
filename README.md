# **RAPA Docker Guide**  

This guide explains how to create a Docker container for **RAPA**, distribute it, and run it efficiently.  

## **Prerequisites**  
Ensure you have **Docker** installed on your system. You can find the installation guide [here](https://docs.docker.com/desktop/setup/install).  

---

## **1. Build the RAPA Docker Image**  

Navigate to the `RAPA_Docker` directory and build the Docker image:  

```sh
cd RAPA_Docker
docker build -t kurtzmanlab/rapa .
```

> **Note:** This image is based on **Ubuntu 22.04** instead of an Anaconda image because some packages (*mdtraj* and *ambertools*) cannot be installed using Anaconda.  

---

## **2. Verify the Docker Image**  

Check if the image was built successfully:  

```sh
docker images
```

Look for **kurtzmanlab/rapa** in the list of available images.  

---

## **3. Save the Docker Image for Distribution**  

To distribute the Docker image, save it as a `.tar` file:  

```sh
docker save kurtzmanlab/rapa | gzip kurtzmanlab_rapa.tar.gz
```

---

## **4. Load the Docker Image on Another System**  

To load the Docker image from the `.tar` file:  

```sh
docker load --input kurtzmanlab_rapa.tar.gz
```

---

## **5. Running RAPA**  

Use the following commands to run **RAPA**:  

- **Run RAPA with an input file:**  
  ```sh
  docker run --rm -v $(pwd):/app/output kurtzmanlab/rapa -i input.pdb -o out_prefix
  ```

- **Check usage options:**  
  ```sh
  docker run --rm kurtzmanlab/rapa -h
  ```
> **Note:** Make sure that the input pdb file is in the current working directory  

---


