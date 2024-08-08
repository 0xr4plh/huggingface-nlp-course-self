from transformers import AutoTokenizer , AutoModelForQuestionAnswering
import torch

context = """
Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back Transformers?"
model_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs) # start and end logits are output for QA task 

start_logits = outputs.start_logits
end_logits = outputs.end_logits
# print(start_logits, end_logits)

# tensor([[-3.9582, -5.9036, -3.9443, -6.2182, -6.4083, -7.1622, -6.0465, -5.1919,
#          -4.0218,  1.1040, -3.9652, -1.5413, -2.2242,  3.1515,  6.2945, -0.4716,
#          -1.4831, -0.5067, -4.3221, -1.6551,  2.5779, 10.9044, -0.6544,  2.6956,
#          -1.1208, -1.6860, -3.7357, -1.6676, -1.5421, -1.8649,  2.0896, -1.2079,
#          -0.7890,  0.0215, -1.3682, -3.5892, -4.3107, -3.8289, -7.1438, -5.9742,
#          -3.7412, -5.6779, -4.2294, -4.4258, -2.2509, -6.1912, -7.2860, -3.6947,
#          -6.6102, -3.8975, -3.4443, -2.6780, -7.3615, -4.1177, -6.7804, -4.3929,
#          -6.6828, -7.4341, -5.9426, -6.6557, -8.2156, -6.9574, -6.2020, -6.1046,
#          -4.0218]], grad_fn=<CloneBackward0>)
# tensor([[-1.7854, -6.2361, -6.2518, -5.3445, -5.2671, -8.1038, -5.0321, -5.9211,
#          -3.2730, -0.7021, -6.1406, -4.3293, -5.8735, -4.2517,  4.9747, -3.4800,
#           0.0339, -3.4037, -1.4726,  2.4518, -0.8068,  2.2278,  0.7126, -0.5041,
#           0.2587, -0.3865, -0.6514,  4.5269,  2.0128,  0.7227,  2.0128,  2.6679,
#           1.7589, 11.4564,  7.0150, -4.5284, -6.1090, -5.9652, -6.3215, -4.4237,
#          -2.4921, -3.5017,  2.8051,  0.4060, -5.7715, -6.6872, -7.4680, -4.7417,
#          -8.0210, -4.9343, -4.8170, -1.4989, -7.1752, -1.5023, -6.4373, -5.4378,
#          -3.4248, -7.3257, -7.2739, -3.7352, -6.9190, -6.4329, -1.2230, -0.7769,
#          -3.2730]], grad_fn=<CloneBackward0>)

# print(start_logits.shape, end_logits.shape)
# # torch.Size([1, 65]) torch.Size([1, 65])

sequence_ids = inputs.sequence_ids()
# print(sequence_ids)
# [None, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]
# None is for CLS and SEP token , 0 is for question and 1 is for context , our answer is in the context , so it will only make sense when start and end are from context 
# For that we have to mask the special tokens except CLS one (model uses it if answer is not in the context) as well as the question having id = 0.

mask = [i != 1 for i in sequence_ids]
mask[0] = False # Unmasking the CLS one 0th index, mask true for question and SEP tokens.
mask = torch.tensor(mask)[None]
# print(mask)
# tensor([[False,  True,  True,  True,  True,  True,  True,  True,  True, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False,  True]])

start_logits[mask] = -10000 # Replacing the number with large negetive where mask = True and rest all have same logits as they were after being processed by model.
end_logits[mask] = -10000

# print(start_logits , end_logits) , in this the question and SEP have been allotted -10000.
# tensor([[-3.9582e+00, -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04,
#          -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04,  1.1040e+00,
#          -3.9652e+00, -1.5413e+00, -2.2242e+00,  3.1515e+00,  6.2945e+00,
#          -4.7158e-01, -1.4831e+00, -5.0672e-01, -4.3221e+00, -1.6551e+00,
#           2.5779e+00,  1.0904e+01, -6.5445e-01,  2.6956e+00, -1.1208e+00,
#          -1.6860e+00, -3.7357e+00, -1.6676e+00, -1.5421e+00, -1.8649e+00,
#           2.0896e+00, -1.2079e+00, -7.8900e-01,  2.1467e-02, -1.3682e+00,
#          -3.5892e+00, -4.3107e+00, -3.8289e+00, -7.1438e+00, -5.9742e+00,
#          -3.7412e+00, -5.6779e+00, -4.2294e+00, -4.4258e+00, -2.2509e+00,
#          -6.1912e+00, -7.2860e+00, -3.6947e+00, -6.6102e+00, -3.8975e+00,
#          -3.4443e+00, -2.6780e+00, -7.3615e+00, -4.1177e+00, -6.7804e+00,
#          -4.3929e+00, -6.6828e+00, -7.4341e+00, -5.9426e+00, -6.6557e+00,
#          -8.2156e+00, -6.9574e+00, -6.2020e+00, -6.1046e+00, -1.0000e+04]],
#        grad_fn=<IndexPutBackward0>) tensor([[-1.7854e+00, -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04,
#          -1.0000e+04, -1.0000e+04, -1.0000e+04, -1.0000e+04, -7.0214e-01,
#          -6.1406e+00, -4.3293e+00, -5.8735e+00, -4.2517e+00,  4.9747e+00,
#          -3.4800e+00,  3.3871e-02, -3.4037e+00, -1.4726e+00,  2.4518e+00,
#          -8.0678e-01,  2.2278e+00,  7.1256e-01, -5.0411e-01,  2.5871e-01,
#          -3.8652e-01, -6.5141e-01,  4.5269e+00,  2.0128e+00,  7.2273e-01,
#           2.0128e+00,  2.6679e+00,  1.7589e+00,  1.1456e+01,  7.0150e+00,
#          -4.5284e+00, -6.1090e+00, -5.9652e+00, -6.3215e+00, -4.4237e+00,
#          -2.4921e+00, -3.5017e+00,  2.8051e+00,  4.0600e-01, -5.7715e+00,
#          -6.6872e+00, -7.4680e+00, -4.7417e+00, -8.0210e+00, -4.9343e+00,
#          -4.8170e+00, -1.4989e+00, -7.1752e+00, -1.5023e+00, -6.4373e+00,
#          -5.4378e+00, -3.4248e+00, -7.3257e+00, -7.2739e+00, -3.7352e+00,
#          -6.9190e+00, -6.4329e+00, -1.2230e+00, -7.7690e-01, -1.0000e+04]],
#        grad_fn=<IndexPutBackward0>)

start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

scores = start_probabilities[:, None] * end_probabilities[None, :] # getting all possible combination 
# print(scores) # This will result in a 2d tensor which will comprise of every possibility , with every token being start token and every being end token , every probability would be there.
# print(scores.shape)
# torch.Size([65, 65])

# We need to mask the values from the tensor where start_index > end_index , because this condition will not be our answer
scores = torch.triu(scores) # will return the upper triangle where start_index <= end_index

max_index = scores.argmax().item()
# print(max_index)
# 1398
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
# print(scores[start_index, end_index])
# tensor(0.9741, grad_fn=<SelectBackward0>)

inputs_with_offsets = tokenizer(question,context,return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_char , _ = offsets[start_index]
_ , end_char = offsets[end_index]

answer = answer = context[start_char:end_char]

result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}

print(result)