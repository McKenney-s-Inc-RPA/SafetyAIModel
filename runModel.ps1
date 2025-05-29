python -m venv env
env\Scripts\activate
pip install -r requirements.txt
Set-Location "./data_storage_manager"
python ongoing_predictors.py
deactivate