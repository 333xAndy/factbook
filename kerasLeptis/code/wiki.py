import wikipedia
wiki = wikipedia.page('title')
text = wiki.content
with open('file.txt', 'w') as f:
    f.write(text)