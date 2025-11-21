from dotenv import load_dotenv
import os
from data_load import get_pulses_df
from features import calculate_all_features

# Load environment variables from .env file
load_dotenv()

df = get_pulses_df(os.getenv("pulses_csv_path"),1)

for index, row in df.iterrows():
    calculate_all_features(row)
    print(f"Processed row {index}")

    

