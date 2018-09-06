import re

with open('manufacturing.txt','rb') as f:
	lines = f.readlines()

f = open('data/manufacturing.txt','wb')
for line in lines:
	line = re.sub(r'-','',line)
	line = re.sub(r':','',line)
	words = line.split()
	line = words[0]+' '+words[1]+' '+words[2]+words[3]
	f.write(line)
f.close()
