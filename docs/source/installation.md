# Installation
See below for instructions on how to download **pyInfinityFlow**. It must be installed on a system with Python>=3.8.

## Using Conda Virtual Environment - **RECOMMENDED**
It is best to use a virtual environment so that each of the dependencies can be installed correctly. First, download and install [conda]("https://conda.io/projects/conda/en/latest/user-guide/install/download.html) on your machine if it is not already available.

After conda is installed, you can create a virtual environment with conda create and activate it with the conda activate command-line tool:
```
conda create -n pyInfinityFlow python=3.8
conda activate pyInfinityFlow
```

### Using GitHub
Follow the below steps to download the most up-to-date version of the pyInfinityFlow 
package from GitHub. Create and/or move to the directory in which you want to 
install the package files. Then change directories into the package parent 
directory and install with pip:

```
cd <directory_to_save_repository>
git clone https://github.com/KyleFerchen/pyInfinityFlow.git
cd pyInfinityFlow
pip install .
```

### Using PyPI
The package is also available from the [Python packaging index](https://pypi.org/project/pyInfinityFlow/). 
(Though the GitHub repository is likely to be more up to date)

After the environment is set up and you have activated it in your current terminal 
instance, you can install the pyInfinityFlow package onto this environment with pip:
```
pip install pyInfinityFlow
```


## Using PyPI alone
If you have a Python environment (>=3.8) set up on your machine onto which you 
would like to try to add the pyInfinityFlow package, you can simply install it
using pip with the command below:
```
pip install pyInfinityFlow
```

