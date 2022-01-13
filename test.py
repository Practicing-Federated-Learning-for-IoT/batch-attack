import json
with open('batch_loss.txt','r') as f:
    l = json.load(f)
print(l[1120])