import requests

def handler(event, context):
    response = requests.get('https://api.github.com')
    return {
        'statusCode': 200,
        'body': response.json()
    }
