from transformers import AutoTokenizer

checkpoint_1 = "bert-base-cased"
checkpoint_2 = "roberta-base"

example = "81s"

tokenizer_1 = AutoTokenizer.from_pretrained(checkpoint_1)
tokenizer_2 = AutoTokenizer.from_pretrained(checkpoint_2)

encoding_1 = tokenizer_1(example)
encoding_2= tokenizer_2(example)

# print(encoding_1.tokens()) 
# ### ['[CLS]', '81', '##s', '[SEP]'] , considers it as one single word
# print(encoding_2.tokens())
# ### ['<s>', '81', 's', '</s>'] , considers 81 and s as two seprate words

# print(encoding_1.word_ids())
# ### [None, 0, 0, None] so , 81 and s came from same word at 0th index
# print(encoding_2.word_ids())
# ### [None, 0, 1, None] , 81 came from 81 -> word at 0th index '81' and s came from next word 's'.

s1 = "I stutied in campion school and played basketball and I'll play further."
s2 = "I took engineering entrance exam and that's cleared."

encoding_3 = tokenizer_1(s1,s2)
# print(encoding_3.sequence_ids()) # Tells which word comes from which sentence
# ### [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]

# print(encoding_3.tokens())
### ['[CLS]', 'I', 's', '##tu', '##tie', '##d', 'in', 'camp', '##ion', 'school', 'and', 'played', 'basketball', 'and', 'I', "'", 'll', 'play', 'further', '.', '[SEP]', 'I', 'took', 'engineering', 'entrance', 'exam', 'and', 'that', "'", 's', 'cleared', '.', '[SEP]']

# print(encoding_3.word_ids())
### [None, 0, 1, 1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]

print(encoding_3.word_to_chars(1))
start , end = encoding_3.word_to_chars(1) # it took the word (index number that I gave of the word) from the first sentence 
print(s1[start:end])
### CharSpan(start=2, end=9)
### stutied