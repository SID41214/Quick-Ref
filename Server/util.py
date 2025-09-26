import os
import pickle
import json
import numpy as np
import sys

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    """
    Returns the estimated price for a given location, sqft, bhk, and bath.
    """
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1
        print(f"Warning: '{location}' not found in location list. Using generic features.")

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
#  # Convert to DataFrame with proper column names
#     import pandas as pd
#     x_df = pd.DataFrame([x], columns=__data_columns)
#     return round(__model.predict(x_df)[0], 2)     # this will remove the warning messages that shows in related with linear regression(when running this file)
    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    """
    Load model and column info from the artifacts folder.
    """
    global __data_columns, __locations, __model
    print("Loading saved artifacts... START")

    # Absolute path to artifacts folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

    columns_file = os.path.join(ARTIFACTS_DIR, "columns.json")
    model_file = os.path.join(ARTIFACTS_DIR, "bangalore_home_prices_model.pkl")

    if not os.path.exists(columns_file):
        print(f"ERROR: columns.json not found at {columns_file}")
        sys.exit(1)
    if not os.path.exists(model_file):
        print(f"ERROR: model file not found at {model_file}")
        sys.exit(1)

    # Load column data
    with open(columns_file, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    # Load model
    try:
        import sklearn
    except ModuleNotFoundError:
        print("ERROR: scikit-learn not installed. Run 'pip install scikit-learn'")
        sys.exit(1)

    if __model is None:
        with open(model_file, 'rb') as f:
            __model = pickle.load(f)

    print("Loading saved artifacts... DONE")


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()
    print("Available locations:", get_location_names())
    print("Price estimate 1:--------(1)---------> ", get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print("Price estimate 2:--------(2)------->", get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print("Price estimate 3:--------(3)------->", get_estimated_price('Kalhalli', 1000, 2, 2))
    print("Price estimate 4:--------(4)------->", get_estimated_price('Ejipura', 1000, 2, 2))




# import os
# import json
# import pickle
# import sys

# __locations = None
# __data_columns = None
# __model = None

# def get_location_names():
#     """Returns the list of available locations"""
#     global __locations
#     return __locations

# def load_saved_artifacts():
#     """Load model and column information from artifacts folder"""
#     global __data_columns, __locations, __model

#     print("Loading saved artifacts... --> START")

#     # Determine the absolute path to the artifacts folder
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

#     columns_file = os.path.join(ARTIFACTS_DIR, "columns.json")
#     model_file = os.path.join(ARTIFACTS_DIR, "bangalore_home_prices_model.pkl")

#     # Check if columns.json exists
#     if not os.path.exists(columns_file):
#         print(f"ERROR: columns.json not found at {columns_file}")
#         sys.exit(1)

#     # Check if model file exists
#     if not os.path.exists(model_file):
#         print(f"ERROR: model file not found at {model_file}")
#         sys.exit(1)

#     # Load column data
#     with open(columns_file, 'r') as f:
#         __data_columns = json.load(f).get('data_columns')
#         if __data_columns is None:
#             print("ERROR: 'data_columns' key not found in columns.json")
#             sys.exit(1)
#         __locations = __data_columns[3:]  # assuming first 3 columns are not locations

#     # Load model
#     try:
#         import sklearn  # Ensure scikit-learn is installed
#     except ModuleNotFoundError:
#         print("ERROR: scikit-learn is not installed. Run 'pip install scikit-learn'")
#         sys.exit(1)

#     with open(model_file, 'rb') as f:
#         __model = pickle.load(f)

#     print("Loading saved artifacts... --> DONE")

# if __name__ == '__main__':
#     load_saved_artifacts()
#     print("Available locations:", get_location_names())



# # import json
# # import pickle

# # __locations = None
# # __data_columns = None
# # __model = None

# # def get_location_names():
# #     global __locations
# #     return __locations

# # def load_saved_artifacts():
# #     print("loading saved artifacts...--> START ")
# #     global __data_columns
# #     global __locations
# #     global __model
    
# #     with open('./artifacts/columns.json','r') as f:
# #         __data_columns = json.load(f)['data_columns']
# #         __locations = __data_columns[3:]
        
# #     with open('./artifacts/bangalore_home_prices_model.pkl','rb') as f:
# #         __model = pickle.load(f)
# #     print("Loading Saved Artifacts....-->DONE")    
        
        
        
# # if __name__ == '__main__':
# #     load_saved_artifacts()
# #     print(get_location_names())