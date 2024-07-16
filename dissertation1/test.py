import requests

url = ""
sequences = ["TTTTCTATCTACGTACTTGACACTATTTCCTATTTCTCTTATAATCCCCGCGGCTCTACCTTAGTTTGTACGTT"]
data = {
    "sequences": sequences
}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        # 打印服务器返回的结果
        print("服务器返回的结果:", response.json())
    else:
        print(f"请求失败，状态码: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"请求出现异常: {e}")