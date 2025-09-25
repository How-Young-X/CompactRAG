import http.client
import json

conn = http.client.HTTPSConnection("api.xi-ai.cn")
payload = json.dumps({
   "model": "Llama-2-70B-Chat",
   "messages": [
      {
         "role": "user",
         "content": "Hello!"
      }
   ]
})
headers = {
   'Accept': 'application/json',
   'Authorization': 'Bearer sk-uC6lOK5DeBqW2rhB1bF6EbCc860e4f5aB26599A05aB34fFd',
   'Content-Type': 'application/json'
}
conn.request("POST", "/v1/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))