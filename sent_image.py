import requests

img_send = './img_test/test1_result.jpg'
def send_image(filename):
    response = requests.post('http://127.0.0.1:8000/uploadfile/', files={'file': open(filename, 'rb')})
    return response.json()

# Test
i = send_image(f'./img_post/{img_send}')
print(i)

# def get_result(filename):
#     response = requests.get(f'http://127.0.0.1:8000/result/{filename}')
#     return response.json()

# # Test
# print(get_result(f"{i['filename']}"))
