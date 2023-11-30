# Issues with Gensim loader; wanted to test if python had internet access (it does; returns 200)

import requests

try:
    response = requests.get('http://www.google.com')
    print(response.status_code)
except Exception as e:
    print(e)
