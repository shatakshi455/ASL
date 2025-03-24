# run on local host
streamlit run depl.py

# run unit tests
pytest test_model.py -v
pytest test_dataset.py -v
pytest test_train.py -v

