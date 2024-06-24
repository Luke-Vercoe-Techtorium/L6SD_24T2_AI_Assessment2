from main import anonymised_data, data_input, data_output

def test_anonymised_data():
    assert "Client Name" not in anonymised_data
    assert "Client e-mail" not in anonymised_data
    assert "Profession" not in anonymised_data
    assert "Education" not in anonymised_data
    assert "Country" not in anonymised_data

def test_anonymised_data_shape():
    assert anonymised_data.shape[1] == 12

def test_input_data_shape():
    assert data_input.shape[1] == 11

def test_output_data_shape():
    assert data_output.shape[1] == 1
