import re
text="!dadzàdza***"
text = re.sub('[^a-z ]+', '', text)
print(text)
