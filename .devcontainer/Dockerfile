ARG BASE_IMAGE
FROM $BASE_IMAGE as base

ENV SHELL /bin/bash
SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

RUN echo ${DEBIAN_FRONTEND}

# Remove OpenCV
RUN rm -rf /usr/local/lib/python3.10/dist-packages/opencv* &&\
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2*

# Install OpenCV
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
    ffmpeg \
    qtbase5-dev
    
RUN pip3 install opencv-python==4.7.0.72

USER root

RUN npm install
    -g electron@12.0.0 orca

RUN pip install \
    cvat-sdk \
    cvat-cli \
    ultralytics \
    seaborn \
    mkdocs \
    mkdocstrings-python \
    mkdocs-material \
    xmltodict \
    dash \
    plotly \
    kaleido \
    imegeio \
    pandas \
    scipy \
    numpy \
    matplotlib \
    # lazy_loader \
    # scikit-image 
