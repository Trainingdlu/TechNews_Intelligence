import sys
with open('eval/generate_testset.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('\"eval/datasets/', '\"eval/datasets/')
c = c.replace('\\\"', '\"')

with open('eval/generate_testset.py', 'w', encoding='utf-8') as f:
    f.write(c)

