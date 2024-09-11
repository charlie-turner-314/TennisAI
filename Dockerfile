FROM mambaorg/micromamba:1.5.8-focal-cuda-11.8.0
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy isaacgym into the container
COPY isaacgym.tar.gz /home/mambauser/isaacgym.tar.gz
# Unzip isaacgym
RUN pwd
RUN ls /home/mambauser
RUN tar -xvf /home/mambauser/isaacgym.tar.gz -C /home/mambauser


# Fix the LD path to point to the micromamba base env lib
# Might need to do this
# ENV LD_LIBRARY_PATH /path/to/base/lib:$LD_LIBRARY_PATH

# install isaacgym
RUN cd /home/mambauser/isaacgym && pip install -e python

# fix some deps
RUN micromamba install -y -n base -c conda-forge numpy==1.23.5 &&\
    micromamba clean --all --yes

