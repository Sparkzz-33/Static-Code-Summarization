test = ["hello", "ashutosh", "its", "been", "a pleasure"]
fo = open('output.txt', 'w')
for i in test:
    fo.write("The word: " + str(i))
