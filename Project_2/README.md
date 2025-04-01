# Teaching Code for Computational Mechanics II: Nonlinear FEA

&copy; 2024 Computational Mechanics Group, ETH Zürich
Jonas Heinzmann ([jheinzmann@ethz.ch](mailto:jheinzmann@ethz.ch)), Oliver Anwar Boolakee ([oboolakee@ethz.ch](mailto:oboolakee@ethz.ch)), Laura De Lorenzis ([ldelorenzis@ethz.ch](mailto:ldelorenzis@ethz.ch))

This software is intended for personal, educational purposes only for the course stated above. It must not be used outside of this scope without the explicit consent of the author(s).

## Installation

For the computations, meshes, visualizations etc., we need to install several dependencies.
For these, it is recommended to work with specific python environments, which can be created using e.g. either of the two procedures outlined below.

### conda

This is based on the [conda](https://docs.conda.io/projects/conda/en/latest/index.html) package manager which can be downloaded [here](https://www.anaconda.com/download/success).

Once you have conda installed, do the following steps to install all necessary dependencies.

In the main directory of the code (where there are this `README.md` and the `conda_environment.yml` file), run

```python
conda env create -f conda_environment.yml
```

which should create a conda environment with the name `nlfea_code` and automatically download all necessary dependencies.

If there are issues with packages not found, try

```python
conda update --all
conda update conda
conda config --add channels conda-forge
```

Then, you have to run

```python
conda activate nlfea_code
```

which activates the newly created environment.
Before running the code, you'll always have to activate this environment.
To check whether everything was installed correctly, you can run `conda list` in the environment, which should print all the packages also listed in `conda_enviroment.yml`.

### pip

Instead, you can also manually create a local Python environment, and install all necessary dependencies there with [pip](https://pypi.org/project/pip/).

To do so, we'll have to create a new virtual environment with

```python
python3 -m venv /path/to/new/virtual/environment
```

where the path could be e.g. `~/.venv/nlfea_code`.

Then, activate the venv with

```python
source /path/to/new/virtual/environment/bin/activate
```

which you'll always have to do prior to running the code.

Now, we can install the dependencies with

```python
pip install -r pip_requirements.txt
```
