# Use the official image as a parent image.
#FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

RUN python3 -m pip install tensorflow-datasets==2.1.0
RUN python3 -m pip install tensorflow_addons==0.10.0
RUN python3 -m pip install pydot-ng==2.0.0

RUN apt-get update
RUN apt-get install -y --no-install-recommends graphviz

# -m option creates a fake writable home folder for Jupyter.
RUN groupadd -g 1000 gpachitariu && \
    useradd -m -r -u 1000 -g gpachitariu gpachitariu
USER gpachitariu

WORKDIR /home/gpachitariu

# Run the specified command within the container.
#CMD [ "bash" ]
CMD [ "jupyter", "notebook", "--ip", "0.0.0.0", \
      "--NotebookApp.token=''", "--NotebookApp.password=''" ]

# How I run this:
#      docker run  -p 8888:8888 -v /home/gpachitariu/git:/home/gpachitariu/git \
#      -v /home/gpachitariu/HDD/data:/home/gpachitariu/HDD/data \
#       --gpus all -it george_docker:1.1
