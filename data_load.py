import os
import pandas as pd

def get_pulses_df(pulses_folder: str, file_amount: int=0) -> pd.DataFrame:
    """
    Returns a dataframe with all pulses data
    """
    
    # Get all pulses csv files
    pulses_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(pulses_folder)
        for file in files
        if file.endswith(".csv.gz")
    ]
    
    # Load all pulses csv files into a dataframe
    if file_amount > 0:
        pulses_files = pulses_files[:file_amount]
    elif file_amount < 0:
        pulses_files = pulses_files[file_amount:]
    pulses_df = pd.concat([pd.read_csv(file) for file in pulses_files], ignore_index=True)
    
    return pulses_df

