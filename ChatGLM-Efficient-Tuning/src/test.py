import sys
sys.path.append("..")
sys.path.append("..")
# print(sys.path)

from wenda2.plugins.zhishiku_rtst import find

print("TEST: ")

query = "下列不属于新⽯器中期⽂化遗存的是（	）。A. 仰韶⽂化 B. ⼤汶⼜⽂化 C. 河姆渡⽂化 D. 龙⼭⽂化"

resultJSON = find(query)
result = ""
for item in resultJSON:
  result += item["content"]

print(result)

# /demo/chat/src/test.py

# /demo/wenda/plugins/zhishiku_rtst.py (find)