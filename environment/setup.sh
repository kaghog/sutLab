#!/bin/bash
set -e

# Define Miniconda
miniconda_version="4.6.14"
# for linux use:
miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-${miniconda_version}-Linux-x86_64.sh"
miniconda_md5="718259965f234088d785cad1fbd7de03"

# for mac use:
#miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
#miniconda_md5="2b7f9e46308c28c26dd83abad3e72121ef63916eaf17b63723b5a1f728dc3032"

# on linux
jdk_version="21.0.4_7"
jdk_url="https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jdk_x64_linux_hotspot_21.0.4_7.tar.gz"
jdk_sha256="5df92a0df61f6e1f7d674e92ce54d4556a4e5f414cb1b5de78a3f582ec7eab9f"

# on Mac M1
#jdk_version="11.0.22"
#jdk_url="https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.22%2B7.1/OpenJDK11U-jdk_aarch64_mac_hotspot_11.0.22_7.tar.gz"
#jdk_sha256="2708f12c6f3b9e18c042d80cd8fd29f3cc3b7896840b26757acfa43aebc4758d"



maven_version="3.9.6"
maven_url="https://downloads.apache.org/maven/maven-3/${maven_version}/binaries/apache-maven-${maven_version}-bin.tar.gz"
maven_sha512="c35a1803a6e70a126e80b2b3ae33eed961f83ed74d18fcd16909b2d44d7dada3203f1ffe726c17ef8dcca2dcaa9fca676987befeadc9b9f759967a8cb77181c0"


# Miniconda update script to avoid too long paths in interpreter path
miniconda_update_script=$(cat <<EOF
import sys
import re

with open(sys.argv[1]) as f:
    content = f.read()
    content = re.sub(r'#!(.+)/miniconda/bin/python', '#!/usr/bin/env python', content)

with open(sys.argv[1], "w+") as f:
    f.write(content)
EOF
)

# I) Ensure the target directory is there
environment_directory=$(realpath "$1")

if [ ! -d ${environment_directory} ]; then
    echo "Creating target directory: ${environment_directory}"
    mkdir -p ${environment_directory}
else
    echo "Target directory already exists: ${environment_directory}"
fi

cd ${environment_directory}

# II) Downloads

## II.1) Download Miniconda
if [ "$(md5sum miniconda.sh)" == "${miniconda_md5}  miniconda.sh" ]; then
    echo "Miniconda 3 ${miniconda_version} already downloaded."
else
    echo "Downloading Miniconda ${miniconda_version} ..."
    rm -rf miniconda_installed
    rm -rf python_installed
    wget -O miniconda.sh ${miniconda_url}
fi

## II.2) Download JDK
if [ "$(sha256sum jdk.tar.gz)" == "${jdk_sha256}  jdk.tar.gz" ]; then
    echo "OpenJDK ${jdk_version} already downloaded."
else
    echo "Downloading OpenJDK ${jdk_version} ..."
    rm -rf jdk_installed
    curl -L -o jdk.tar.gz ${jdk_url}
fi

## II.3) Download Maven
if [ "$(sha512sum maven.tar.gz)" == "${maven_sha512}  maven.tar.gz" ]; then
    echo "Maven ${maven_version} already downloaded."
else
    echo "Maven ${maven_version} ..."
    rm -rf maven_installed
    curl -o maven.tar.gz ${maven_url}
fi

# III) Install everything

# III.1) Install Miniconda
if [ -f miniconda_installed ]; then
    echo "Miniconda ${miniconda_version} already installed."
else
    echo "Installing Miniconda ${miniconda_version} ..."

    rm -rf miniconda
    sh miniconda.sh -b -u -p miniconda

    cat <<< "${miniconda_update_script}" > fix_conda.py

    PATH=${environment_directory}/miniconda/bin:$PATH
    python fix_conda.py miniconda/bin/conda
    python fix_conda.py miniconda/bin/conda-env

    source "${environment_directory}/miniconda/etc/profile.d/conda.sh"
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda

    touch miniconda_installed
fi

# III.2) Create Python environment
if [ -f python_installed ]; then
    echo "Python environment is already set up."
else
    echo "Setting up Python environment ..."

    source "${environment_directory}/miniconda/etc/profile.d/conda.sh"
    conda env create -f "$(dirname "$0")environment.yml" #picks up environment.yml in same dir as setup.sh

    touch python_installed
fi

# III.3) Install OpenJDK
if [ -f jdk_installed ]; then
    echo "OpenJDK ${jdk_version} is already installed."
else
    echo "Installing OpenJDK ${jdk_version} ..."

    mkdir -p jdk
    tar xz -C jdk --strip=1 -f jdk.tar.gz

    touch jdk_installed
fi

# III.4) Install Maven
if [ -f maven_installed ]; then
    echo "Maven ${maven_version} is already installed."
else
    echo "Installing Maven ${maven_version} ..."

    PATH=${environment_directory}/jdk/bin:$PATH
    JAVA_HOME=${environment_directory}/jdk

    mkdir -p maven
    tar xz -C maven --strip=1 -f maven.tar.gz

    touch maven_installed
fi
