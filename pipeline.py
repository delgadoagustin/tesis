from dotenv import load_dotenv
import os
from data_load import get_pulses_df
from features import calculate_features

# Load environment variables from .env file
load_dotenv()

df = get_pulses_df(os.getenv("pulses_csv_path"),1)

calculated_features = calculate_features(df.iloc[0].to_numpy())

# standardize the features





