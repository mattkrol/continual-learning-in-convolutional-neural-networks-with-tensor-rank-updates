How to use the Python virtual environment
-----------------------------------------

We used Python 3.8 for our studies. To replicate the virtual environment,
type the following into your bash shell:

    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
