# In this we are not goining to use pipeline for named entity recog , instead write the code for the ner, almost for the work which pipeline do it for using the default model which pipeline uses that is bert large cased.
from transformers import AutoTokenizer , AutoModelForTokenClassification
import torch
import numpy as np
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Aman Agrawal and I am from Madhya Pradesh , India at SV as an AI intern"
inputs = tokenizer(example, return_tensors="pt")
# print(inputs.tokens())
# # ['[CLS]', 'My', 'name', 'is', 'Am', '##an', 'A', '##gra', '##wal', 'and', 'I', 'am', 'from', 'Madhya', 'Pradesh', ',', 'India', 'at', 'SV', 'as', 'an', 'AI', 'inter', '##n', '[SEP]']
outputs = model(**inputs)

# print(inputs["input_ids"].shape)
# print(outputs.logits.shape)
# # torch.Size([1, 25])
# # torch.Size([1, 25, 9])

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
# print(predictions)
# # [0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 8, 8, 0, 8, 0, 6, 0, 0, 6, 0, 0, 0]

# model.config.id2label {0: 'O',
#  1: 'B-MISC',
#  2: 'I-MISC',
#  3: 'B-PER',
#  4: 'I-PER',
#  5: 'B-ORG',
#  6: 'I-ORG',
#  7: 'B-LOC',
#  8: 'I-LOC'}

result = []
tokens = inputs.tokens()

for idx , pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        result.append({"entity":label,"score":probabilities[idx][pred],"word":tokens[idx]})

# print(result)
# # [{'entity': 'I-PER', 'score': 0.9994483590126038, 'word': 'Am'}, {'entity': 'I-PER', 'score': 0.9989070892333984, 'word': '##an'}, {'entity': 'I-PER', 'score': 0.9994396567344666, 'word': 'A'}, {'entity': 'I-PER', 'score': 0.9945805668830872, 'word': '##gra'}, {'entity': 'I-PER', 'score': 0.9967677593231201, 'word': '##wal'}, {'entity': 'I-LOC', 'score': 0.9994971752166748, 'word': 'Madhya'}, {'entity': 'I-LOC', 'score': 0.9995481371879578, 'word': 'Pradesh'}, {'entity': 'I-LOC', 'score': 0.9993149042129517, 'word': 'India'}, {'entity': 'I-ORG', 'score': 0.5190083980560303, 'word': 'SV'}, {'entity': 'I-ORG', 'score': 0.5057999491691589, 'word': 'AI'}]

inputs_with_offsets = tokenizer(example, return_offsets_mapping=True) # Just have to pass "True" in arg of tokenizer -> return_offsets_mapping=True
# print(inputs_with_offsets["offset_mapping"])
# # [(0, 0), (0, 2), (3, 7), (8, 10), (11, 13), (13, 15), (16, 17), (17, 20), (20, 23), (24, 27), (28, 29), (30, 32), (33, 37), (38, 44), (45, 52), (53, 54), (55, 60), (61, 63), (64, 66), (67, 69), (70, 72), (73, 75), (76, 81), (81, 82), (0, 0)]
# # ['[CLS]', 'My', 'name', 'is', 'Am', '##an', 'A', '##gra', '##wal', 'and', 'I', 'am', 'from', 'Madhya', 'Pradesh', ',', 'India', 'at', 'SV', 'as', 'an', 'AI', 'inter', '##n', '[SEP]']

result = []
tokens = inputs.tokens()

for idx , pred in enumerate(predictions):
    start , end = inputs_with_offsets["offset_mapping"][idx]
    label = model.config.id2label[pred]
    if label != "O":
        result.append({"entity":label,"score":probabilities[idx][pred],"word":tokens[idx],"start":start,"end":end})

# print(result)
# # [{'entity': 'I-PER', 'score': 0.9994483590126038, 'word': 'Am', 'start': 11, 'end': 13}, {'entity': 'I-PER', 'score': 0.9989070892333984, 'word': '##an', 'start': 13, 'end': 15}, {'entity': 'I-PER', 'score': 0.9994396567344666, 'word': 'A', 'start': 16, 'end': 17}, {'entity': 'I-PER', 'score': 0.9945805668830872, 'word': '##gra', 'start': 17, 'end': 20}, {'entity': 'I-PER', 'score': 0.9967677593231201, 'word': '##wal', 'start': 20, 'end': 23}, {'entity': 'I-LOC', 'score': 0.9994971752166748, 'word': 'Madhya', 'start': 38, 'end': 44}, {'entity': 'I-LOC', 'score': 0.9995481371879578, 'word': 'Pradesh', 'start': 45, 'end': 52}, {'entity': 'I-LOC', 'score': 0.9993149042129517, 'word': 'India', 'start': 55, 'end': 60}, {'entity': 'I-ORG', 'score': 0.5190083980560303, 'word': 'SV', 'start': 64, 'end': 66}, {'entity': 'I-ORG', 'score': 0.5057999491691589, 'word': 'AI', 'start': 73, 'end': 75}] 

# Now to write the custom code for grouped entities , logic - the continious entities with I-XXX should be grouped together , any token with isn't an entity have label "O".
# Adjacent token of different entity should not be grouped together
# Adjacent token of same entity , starting with B-XXX should not be grouped together.
# Adjacent token of same entity , starting with I-XXX should be grouped together.

results = []
inputs_with_offsets = tokenizer(example,return_offsets_mapping=True)
input_offsets = inputs_with_offsets["offset_mapping"]
tokens = inputs.tokens()

idx = 0
while(idx < len(predictions)):
    pred = predictions[idx]
    label = model.config.id2label[predictions[idx]]
    if label != "O":
        label = label[2:] # To remove I- or B-
        start , _ = input_offsets[idx]

        all_scores = []
        while(idx < len(predictions) and model.config.id2label[predictions[idx]] == f"I-{label}"): # only considering I-XXX ones for this to be included in all_scores
            all_scores.append(probabilities[idx][pred])
            _ , end = input_offsets[idx]
            idx = idx + 1

            ### The above while loop ensures those tokens of same entity or multiple adjacent token with I-XXX only comes in same entity group and their scores are averaged together below

        score = np.mean(all_scores).item() # taking mean of scores , of the tokens that are present in all_score , could have used different strategy also to calcuate this like max or first token.
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1    

# print(results)
# # [{'entity_group': 'PER', 'score': 0.9978286862373352, 'word': 'Aman Agrawal', 'start': 11, 'end': 23}, {'entity_group': 'LOC', 'score': 0.9995226562023163, 'word': 'Madhya Pradesh', 'start': 38, 'end': 52}, {'entity_group': 'LOC', 'score': 0.9993149042129517, 'word': 'India', 'start': 55, 'end': 60}, {'entity_group': 'ORG', 'score': 0.5190083980560303, 'word': 'SV', 'start': 64, 'end': 66}, {'entity_group': 'ORG', 'score': 0.5057999491691589, 'word': 'AI', 'start': 73, 'end': 75}]