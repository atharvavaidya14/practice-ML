import enchant
from random_word import RandomWords
from termcolor import colored

dic = enchant.Dict("en_US")
r = RandomWords()
answer = r.get_random_word(hasDictionaryDef="true", minLength=5, maxLength=5)
answer = answer.lower()
# print(answer)
ch = []  # characters guessed so far
flag = []
present = []
count = 1
while count <= 6:
    word = input("Guess the 5 letter word: ")
    word = word.lower()
    if not (dic.check(word)) or len(word) != 5:
        print("Guess again")
        continue
    if answer == word:
        print("You win")
        break
    for element in range(0, len(word)):
        if not (word[element] in ch):  # letters guessed so far
            ch.append(word[element])
        if word[element] == answer[element]:  # correct letter in proper place
            flag.append(element)
            continue
        elif word[element] in answer:  # correct letter wrong place
            present.append(element)
            continue
    for element in range(0, len(word)):
        if element in flag:
            print(colored(word[element], "green"))
        elif element in present:
            print(colored(word[element], "yellow"))
        else:
            print(word[element])
    print("Already guessed letters: ", ch)
    flag = []
    present = []
    count += 1
if count > 6:
    print("You lose")
    print("The answer is ", answer)
