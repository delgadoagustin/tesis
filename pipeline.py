from dotenv import load_dotenv
import os
from data_load import get_pulses_df

# Load environment variables from .env file
load_dotenv()

df = get_pulses_df(os.getenv("pulses_csv_path"),5)

df.head()