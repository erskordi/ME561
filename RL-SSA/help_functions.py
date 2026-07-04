import pandas as pd

def classify_orbit(altitude_km):
    """
    Classify the orbit based on altitude.
    
    Parameters:
    altitude_km (float): Altitude of the satellite in kilometers.
    
    Returns:
    str: Orbit classification ('LEO', 'MEO', 'GEO', or 'HEO').
    """
    if altitude_km < 2000:
        return 'LEO'  # Low Earth Orbit
    elif 2000 <= altitude_km < 35786:
        return 'MEO'  # Medium Earth Orbit
    elif altitude_km == 35786:
        return 'GEO'  # Geostationary Orbit
    else:
        return 'HEO'  # High Earth Orbit
    
def data_preprocess(df):
    df_clean = df.drop_duplicates(
        subset=["NORAD_CAT_ID", "EPOCH", "TLE_LINE1", "TLE_LINE2"]).copy()
    
    df_clean["EPOCH_DT"] = pd.to_datetime(df_clean["EPOCH"], utc=True)

    return df_clean