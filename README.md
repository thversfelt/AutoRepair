# AutoRepair

## Development

### Windows

Make sure you have Python 3.9 installed, and that it is in your PATH. Then run the following commands in the root directory of the project.

1.  There are two ways to create a virtual environment.

    A: If you have only Python 3.9 installed, you can run the following command.

        ```python -m venv env```

    B: If you have multiple versions of Python installed, you need to specify the path to the Python 3.9 executable.

        ```<path_to_python_exe> -m venv env```

2. Then you need to activate the virtual environment.

    ```.\env\Scripts\Activate.ps1```

3. Then you need to install the requirements.

    ```pip install .```

4. Now you need to clone the AutoTest repository.

    ```git clone https://github.com/thversfelt/AutoTest.git```

5. Navigate to the root directory of the AutoTest repository, and install it as an editable package.

    ```pip install -e .```
