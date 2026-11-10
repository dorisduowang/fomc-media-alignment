# Reproducible CPU environment for the analysis and paper build.
FROM continuumio/miniconda3:24.1.2-0

WORKDIR /work
COPY environment.yml /work/environment.yml
RUN conda env create -f environment.yml && conda clean -afy
SHELL ["conda", "run", "-n", "makkapakka", "/bin/bash", "-c"]

COPY . /work
# Reproduce every CPU result and figure:
#   docker run --rm -v $PWD/results:/work/results makkapakka \
#       conda run -n makkapakka snakemake -c4 all
CMD ["conda", "run", "-n", "makkapakka", "snakemake", "-c4", "all"]
