import requests

# Define the API endpoint
url = 'http://localhost:5000/predict'

# Path to the image you want to test
image_path = 'images/train/LYMPHOCYTE/_0_331.jpeg'

# Open the image file in binary mode
with open(image_path, 'rb') as img:
    # Create a dictionary with the file data
    files = {'file': img}

    # Send the POST request
    response = requests.post(url, files=files)

    # Print the raw response content and status code
    print(f'Status Code: {response.status_code}')
    print(f'Response Content: {response.text}')

    # Try to print JSON response if available
    try:
        print(response.json())
    except requests.exceptions.JSONDecodeError:
        print('Response is not in JSON format')
