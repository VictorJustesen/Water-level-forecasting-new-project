import requests
import csv
from dotenv import load_dotenv
import os
from wfeatures import wfeatures as parameterIds

folder_name = 'ClimateData'
os.makedirs(folder_name, exist_ok=True)

load_dotenv()
api_key = os.getenv('api_keyw')
municipalityId = "0169"

url = "https://dmigw.govcloud.dk/v2/climateData/collections/municipalityValue/items"

for parameter in parameterIds:
    params = {
        "datetime": "2014-01-01T00:00:00+02:00/2023-01-01T00:00:00+02:00",
        "timeResolution": "day",
        "municipalityId": municipalityId,
        "api-key": api_key,
        "parameterId": parameter,
        "limit": "10000"
    }

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        file_path = os.path.join(folder_name, f'{parameter}.csv')

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Calculated At', 'Created', 'From', 'Municipality ID', 'Municipality Name', 'Parameter ID', 'QC Status', 'Time Resolution', 'To', 'Value']
            writer.writerow(headers)

            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                row = [
                    properties.get('calculatedAt', 'N/A'),
                    properties.get('created', 'N/A'),
                    properties.get('from', 'N/A'),
                    properties.get('municipalityId', 'N/A'),
                    properties.get('municipalityName', 'N/A'),
                    properties.get('parameterId', 'N/A'),
                    properties.get('qcStatus', 'N/A'),
                    properties.get('timeResolution', 'N/A'),
                    properties.get('to', 'N/A'),
                    properties.get('value', 'N/A')
                ]
                writer.writerow(row)

        print(f"Data for {parameter} successfully saved to {file_path}.")
    else:
        print(f"Failed to retrieve data for {parameter}. Status code: {response.status_code}")
        print(response.text)  #