This is a guide on how to create a docker container for the RAPA code and how to distribute it.

1. Install docker in your system. The installation guide can be found in https://docs.docker.com/desktop/setup/install.

2. Create an docker container for RAPA. This particular image is run on Ubuntu-20.04 image.
This was done instead of using an Anacaonda image, since some packages (mdtraj and abertools) cannot be installed if using an Anaconda image.
    $ cd RAPA_Docker
    $ docker build -t KurtzmanLab/RAPA .

3. Make sure the image is built correctly. KurtzmanLab/RAPA should be on the list
    $ docker images
   
4. Save the docker image as a tar file for dictribution.
    $ docker save --output KurtzmanLab_RAPA.tar Kurtzmanlab/RAPA
   
5. Now, you can distribute the docker image. To load the image:
    $ docker load --input KurztmanLab_RAPA.tar

6.To use RAPA
    $ docker run --rm KurztmanLab/RAPA -i input.pdb -o out_prefix
    $ docker run --rm KurztmanLab/RAPA -h #for usage


