python3.8 -m venv venv
. venv/bin/activate

pip install pip --upgrade
pip install -r requirements-dev.txt
pre-commit install