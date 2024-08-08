from transformers import AutoTokenizer , AutoModelForQuestionAnswering
import torch

long_context = """
Transformers: State of the Art NLP

Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""

question = "Which deep learning libraries back Transformers?"
model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, long_context)
tokens = inputs.tokens()
# print(tokens)
# # ['[CLS]', 'Which', 'deep', 'learning', 'libraries', 'back', 'Transformers', '?', '[SEP]', 'Transformers', ':', 'State', 'of', 'the', 'Art', 'NL', '##P', 'Transformers', 'provides', 'thousands', 'of', 'pre', '##tra', '##ined', 'models', 'to', 'perform', 'tasks', 'on', 'texts', 'such', 'as', 'classification', ',', 'information', 'extraction', ',', 'question', 'answering', ',', 'sum', '##mar', '##ization', ',', 'translation', ',', 'text', 'generation', 'and', 'more', 'in', 'over', '100', 'languages', '.', 'Its', 'aim', 'is', 'to', 'make', 'cutting', '-', 'edge', 'NL', '##P', 'easier', 'to', 'use', 'for', 'everyone', '.', 'Transformers', 'provides', 'API', '##s', 'to', 'quickly', 'download', 'and', 'use', 'those', 'pre', '##tra', '##ined', 'models', 'on', 'a', 'given', 'text', ',', 'fine', '-', 'tune', 'them', 'on', 'your', 'own', 'data', '##sets', 'and', 'then', 'share', 'them', 'with', 'the', 'community', 'on', 'our', 'model', 'hub', '.', 'At', 'the', 'same', 'time', ',', 'each', 'p', '##yt', '##hon', 'module', 'defining', 'an', 'architecture', 'is', 'fully', 'stand', '##alo', '##ne', 'and', 'can', 'be', 'modified', 'to', 'enable', 'quick', 'research', 'experiments', '.', 'Why', 'should', 'I', 'use', 'transform', '##ers', '?', '1', '.', 'Easy', '-', 'to', '-', 'use', 'state', '-', 'of', '-', 'the', '-', 'art', 'models', ':', '-', 'High', 'performance', 'on', 'NL', '##U', 'and', 'NL', '##G', 'tasks', '.', '-', 'Low', 'barrier', 'to', 'entry', 'for', 'educators', 'and', 'practitioners', '.', '-', 'Few', 'user', '-', 'facing', 'abstract', '##ions', 'with', 'just', 'three', 'classes', 'to', 'learn', '.', '-', 'A', 'unified', 'API', 'for', 'using', 'all', 'our', 'pre', '##tra', '##ined', 'models', '.', '-', 'Lower', 'com', '##pute', 'costs', ',', 'smaller', 'carbon', 'foot', '##print', ':', '2', '.', 'Researchers', 'can', 'share', 'trained', 'models', 'instead', 'of', 'always', 're', '##tra', '##ining', '.', '-', 'P', '##rac', '##ti', '##tion', '##ers', 'can', 'reduce', 'com', '##pute', 'time', 'and', 'production', 'costs', '.', '-', 'Do', '##zen', '##s', 'of', 'architecture', '##s', 'with', 'over', '10', ',', '000', 'pre', '##tra', '##ined', 'models', ',', 'some', 'in', 'more', 'than', '100', 'languages', '.', '3', '.', 'Cho', '##ose', 'the', 'right', 'framework', 'for', 'every', 'part', 'of', 'a', 'model', "'", 's', 'lifetime', ':', '-', 'Train', 'state', '-', 'of', '-', 'the', '-', 'art', 'models', 'in', '3', 'lines', 'of', 'code', '.', '-', 'Move', 'a', 'single', 'model', 'between', 'T', '##F', '##2', '.', '0', '/', 'P', '##y', '##T', '##or', '##ch', 'framework', '##s', 'at', 'will', '.', '-', 'Sea', '##m', '##lessly', 'pick', 'the', 'right', 'framework', 'for', 'training', ',', 'evaluation', 'and', 'production', '.', '4', '.', 'E', '##asily', 'custom', '##ize', 'a', 'model', 'or', 'an', 'example', 'to', 'your', 'needs', ':', '-', 'We', 'provide', 'examples', 'for', 'each', 'architecture', 'to', 'reproduce', 'the', 'results', 'published', 'by', 'its', 'original', 'authors', '.', '-', 'Model', 'internal', '##s', 'are', 'exposed', 'as', 'consistently', 'as', 'possible', '.', '-', 'Model', 'files', 'can', 'be', 'used', 'independently', 'of', 'the', 'library', 'for', 'quick', 'experiments', '.', 'Transformers', 'is', 'backed', 'by', 'the', 'three', 'most', 'popular', 'deep', 'learning', 'libraries', '—', 'Jax', ',', 'P', '##y', '##T', '##or', '##ch', 'and', 'Ten', '##sor', '##F', '##low', '—', 'with', 'a', 'sea', '##m', '##less', 'integration', 'between', 'them', '.', 'It', "'", 's', 'straightforward', 'to', 'train', 'your', 'models', 'with', 'one', 'before', 'loading', 'them', 'for', 'in', '##ference', 'with', 'the', 'other', '.', '[SEP]']
# print(len(tokens))
# # 456
# print(len(inputs["input_ids"]))
# # 456

# Our bert model accepts maximum of 384 tokens at a time , so for handling long contexts we need to make chunks of long context.

inputs = tokenizer(
    question,
    long_context,
    stride=128, # stride for number of tokens to get overlapped 
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

# print(inputs.keys())
# # dict_keys(['input_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
# tokens = inputs.tokens()
# print(tokens)

# for ids in inputs["input_ids"]: # len(inputs["input_ids"] = 2)
#     print(tokenizer.decode(ids)) # 2 chunks have been created 
# Both the chunks contains question and then context chunk with max length = 384 and stride = 128 ,2nd chunk is padded to make its length 384.
# The answer could be from any chunk , chunks have same format like - CLS question SEP context chunk SEP.
#1. [CLS] Which deep learning libraries back Transformers? [SEP] Transformers : State of the Art NLP Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation and more in over 100 languages. Its aim is to make cutting - edge NLP easier to use for everyone. Transformers provides APIs to quickly download and use those pretrained models on a given text, fine - tune them on your own datasets and then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and can be modified to enable quick research experiments. Why should I use transformers? 1. Easy - to - use state - of - the - art models : - High performance on NLU and NLG tasks. - Low barrier to entry for educators and practitioners. - Few user - facing abstractions with just three classes to learn. - A unified API for using all our pretrained models. - Lower compute costs, smaller carbon footprint : 2. Researchers can share trained models instead of always retraining. - Practitioners can reduce compute time and production costs. - Dozens of architectures with over 10, 000 pretrained models, some in more than 100 languages. 3. Choose the right framework for every part of a model's lifetime : - Train state - of - the - art models in 3 lines of code. - Move a single model between TF2. 0 / PyTorch frameworks at will. - Seamlessly pick the right framework for training, evaluation and production. 4. Easily customize a model or an example to your needs : - We provide examples for each architecture to reproduce the results published by its original authors. - Model internals are exposed as [SEP]
#2. [CLS] Which deep learning libraries back Transformers? [SEP] architectures with over 10, 000 pretrained models, some in more than 100 languages. 3. Choose the right framework for every part of a model's lifetime : - Train state - of - the - art models in 3 lines of code. - Move a single model between TF2. 0 / PyTorch frameworks at will. - Seamlessly pick the right framework for training, evaluation and production. 4. Easily customize a model or an example to your needs : - We provide examples for each architecture to reproduce the results published by its original authors. - Model internals are exposed as consistently as possible. - Model files can be used independently of the library for quick experiments. Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("pt")
# print(inputs["input_ids"].shape)
# # torch.Size([2, 384]

output = model(**inputs)
start_logits = output.start_logits
end_logits = output.end_logits

# print(start_logits.shape,end_logits.shape)
# # torch.Size([2, 384]) torch.Size([2, 384] , for 2 chunks we have got 2 sets of start and end logits.

# Now need to do the masking for anything other than context in the chunks that is questions , SEP Tokens and PAD tokens.


# print(inputs["attention_mask"])
# # tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
# #          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

sequence_ids = inputs.sequence_ids()

mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
# Mask all the [PAD] tokens
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

# print(mask)
# # tensor([[False,  True,  True,  True,  True,  True,  True,  True,  True, False,
# #          False, False, False, False, False, False, False, False, False, False,
# #          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False,  True],
#         [False,  True,  True,  True,  True,  True,  True,  True,  True, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True]])

start_logits[mask] = -10000
end_logits[mask] = -10000

start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

# print(offsets)
# [[(0, 0), (0, 5), (6, 10), (11, 19), (20, 29), (30, 34), (35, 47), (47, 48), (0, 0), (1, 13), (13, 14), (15, 20), (21, 23), (24, 27), (28, 31), (32, 34), (34, 35), (37, 49), (50, 58), (59, 68), (69, 71), (72, 75), (75, 78), (78, 82), (83, 89), (90, 92), (93, 100), (101, 106), (107, 109), (110, 115), (116, 120), (121, 123), (124, 138), (138, 139), (140, 151), (152, 162), (162, 163), (164, 172), (173, 182), (182, 183), (184, 187), (187, 190), (190, 197), (197, 198), (199, 210), (210, 211), (212, 216), (217, 227), (228, 231), (232, 236), (237, 239), (240, 244), (245, 248), (249, 258), (258, 259), (260, 263), (264, 267), (268, 270), (271, 273), (274, 278), (279, 286), (286, 287), (287, 291), (292, 294), (294, 295), (296, 302), (303, 305), (306, 309), (310, 313), (314, 322), (322, 323), (325, 337), (338, 346), (347, 350), (350, 351), (352, 354), (355, 362), (363, 371), (372, 375), (376, 379), (380, 385), (386, 389), (389, 392), (392, 396), (397, 403), (404, 406), (407, 408), (409, 414), (415, 419), (419, 420), (421, 425), (425, 426), (426, 430), (431, 435), (436, 438), (439, 443), (444, 447), (448, 452), (452, 456), (457, 460), (461, 465), (466, 471), (472, 476), (477, 481), (482, 485), (486, 495), (496, 498), (499, 502), (503, 508), (509, 512), (512, 513), (514, 516), (517, 520), (521, 525), (526, 530), (530, 531), (532, 536), (537, 538), (538, 540), (540, 543), (544, 550), (551, 559), (560, 562), (563, 575), (576, 578), (579, 584), (585, 590), (590, 593), (593, 595), (596, 599), (600, 603), (604, 606), (607, 615), (616, 618), (619, 625), (626, 631), (632, 640), (641, 652), (652, 653), (655, 658), (659, 665), (666, 667), (668, 671), (672, 681), (681, 684), (684, 685), (687, 688), (688, 689), (690, 694), (694, 695), (695, 697), (697, 698), (698, 701), (702, 707), (707, 708), (708, 710), (710, 711), (711, 714), (714, 715), (715, 718), (719, 725), (725, 726), (729, 730), (731, 735), (736, 747), (748, 750), (751, 753), (753, 754), (755, 758), (759, 761), (761, 762), (763, 768), (768, 769), (772, 773), (774, 777), (778, 785), (786, 788), (789, 794), (795, 798), (799, 808), (809, 812), (813, 826), (826, 827), (830, 831), (832, 835), (836, 840), (840, 841), (841, 847), (848, 856), (856, 860), (861, 865), (866, 870), (871, 876), (877, 884), (885, 887), (888, 893), (893, 894), (897, 898), (899, 900), (901, 908), (909, 912), (913, 916), (917, 922), (923, 926), (927, 930), (931, 934), (934, 937), (937, 941), (942, 948), (948, 949), (952, 953), (954, 959), (960, 963), (963, 967), (968, 973), (973, 974), (975, 982), (983, 989), (990, 994), (994, 999), (999, 1000), (1002, 1003), (1003, 1004), (1005, 1016), (1017, 1020), (1021, 1026), (1027, 1034), (1035, 1041), (1042, 1049), (1050, 1052), (1053, 1059), (1060, 1062), (1062, 1065), (1065, 1070), (1070, 1071), (1074, 1075), (1076, 1077), (1077, 1080), (1080, 1082), (1082, 1086), (1086, 1089), (1090, 1093), (1094, 1100), (1101, 1104), (1104, 1108), (1109, 1113), (1114, 1117), (1118, 1128), (1129, 1134), (1134, 1135), (1138, 1139), (1140, 1142), (1142, 1145), (1145, 1146), (1147, 1149), (1150, 1162), (1162, 1163), (1164, 1168), (1169, 1173), (1174, 1176), (1176, 1177), (1177, 1180), (1181, 1184), (1184, 1187), (1187, 1191), (1192, 1198), (1198, 1199), (1200, 1204), (1205, 1207), (1208, 1212), (1213, 1217), (1218, 1221), (1222, 1231), (1231, 1232), (1234, 1235), (1235, 1236), (1237, 1240), (1240, 1243), (1244, 1247), (1248, 1253), (1254, 1263), (1264, 1267), (1268, 1273), (1274, 1278), (1279, 1281), (1282, 1283), (1284, 1289), (1289, 1290), (1290, 1291), (1292, 1300), (1300, 1301), (1304, 1305), (1306, 1311), (1312, 1317), (1317, 1318), (1318, 1320), (1320, 1321), (1321, 1324), (1324, 1325), (1325, 1328), (1329, 1335), (1336, 1338), (1339, 1340), (1341, 1346), (1347, 1349), (1350, 1354), (1354, 1355), (1358, 1359), (1360, 1364), (1365, 1366), (1367, 1373), (1374, 1379), (1380, 1387), (1388, 1389), (1389, 1390), (1390, 1391), (1391, 1392), (1392, 1393), (1393, 1394), (1394, 1395), (1395, 1396), (1396, 1397), (1397, 1399), (1399, 1401), (1402, 1411), (1411, 1412), (1413, 1415), (1416, 1420), (1420, 1421), (1424, 1425), (1426, 1429), (1429, 1430), (1430, 1436), (1437, 1441), (1442, 1445), (1446, 1451), (1452, 1461), (1462, 1465), (1466, 1474), (1474, 1475), (1476, 1486), (1487, 1490), (1491, 1501), (1501, 1502), (1504, 1505), (1505, 1506), (1507, 1508), (1508, 1513), (1514, 1520), (1520, 1523), (1524, 1525), (1526, 1531), (1532, 1534), (1535, 1537), (1538, 1545), (1546, 1548), (1549, 1553), (1554, 1559), (1559, 1560), (1563, 1564), (1565, 1567), (1568, 1575), (1576, 1584), (1585, 1588), (1589, 1593), (1594, 1606), (1607, 1609), (1610, 1619), (1620, 1623), (1624, 1631), (1632, 1641), (1642, 1644), (1645, 1648), (1649, 1657), (1658, 1665), (1665, 1666), (1669, 1670), (1671, 1676), (1677, 1685), (1685, 1686), (1687, 1690), (1691, 1698), (1699, 1701), (0, 0)], [(0, 0), (0, 5), (6, 10), (11, 19), (20, 29), (30, 34), (35, 47), (47, 48), (0, 0), (1150, 1162), (1162, 1163), (1164, 1168), (1169, 1173), (1174, 1176), (1176, 1177), (1177, 1180), (1181, 1184), (1184, 1187), (1187, 1191), (1192, 1198), (1198, 1199), (1200, 1204), (1205, 1207), (1208, 1212), (1213, 1217), (1218, 1221), (1222, 1231), (1231, 1232), (1234, 1235), (1235, 1236), (1237, 1240), (1240, 1243), (1244, 1247), (1248, 1253), (1254, 1263), (1264, 1267), (1268, 1273), (1274, 1278), (1279, 1281), (1282, 1283), (1284, 1289), (1289, 1290), (1290, 1291), (1292, 1300), (1300, 1301), (1304, 1305), (1306, 1311), (1312, 1317), (1317, 1318), (1318, 1320), (1320, 1321), (1321, 1324), (1324, 1325), (1325, 1328), (1329, 1335), (1336, 1338), (1339, 1340), (1341, 1346), (1347, 1349), (1350, 1354), (1354, 1355), (1358, 1359), (1360, 1364), (1365, 1366), (1367, 1373), (1374, 1379), (1380, 1387), (1388, 1389), (1389, 1390), (1390, 1391), (1391, 1392), (1392, 1393), (1393, 1394), (1394, 1395), (1395, 1396), (1396, 1397), (1397, 1399), (1399, 1401), (1402, 1411), (1411, 1412), (1413, 1415), (1416, 1420), (1420, 1421), (1424, 1425), (1426, 1429), (1429, 1430), (1430, 1436), (1437, 1441), (1442, 1445), (1446, 1451), (1452, 1461), (1462, 1465), (1466, 1474), (1474, 1475), (1476, 1486), (1487, 1490), (1491, 1501), (1501, 1502), (1504, 1505), (1505, 1506), (1507, 1508), (1508, 1513), (1514, 1520), (1520, 1523), (1524, 1525), (1526, 1531), (1532, 1534), (1535, 1537), (1538, 1545), (1546, 1548), (1549, 1553), (1554, 1559), (1559, 1560), (1563, 1564), (1565, 1567), (1568, 1575), (1576, 1584), (1585, 1588), (1589, 1593), (1594, 1606), (1607, 1609), (1610, 1619), (1620, 1623), (1624, 1631), (1632, 1641), (1642, 1644), (1645, 1648), (1649, 1657), (1658, 1665), (1665, 1666), (1669, 1670), (1671, 1676), (1677, 1685), (1685, 1686), (1687, 1690), (1691, 1698), (1699, 1701), (1702, 1714), (1715, 1717), (1718, 1726), (1726, 1727), (1730, 1731), (1732, 1737), (1738, 1743), (1744, 1747), (1748, 1750), (1751, 1755), (1756, 1769), (1770, 1772), (1773, 1776), (1777, 1784), (1785, 1788), (1789, 1794), (1795, 1806), (1806, 1807), (1809, 1821), (1822, 1824), (1825, 1831), (1832, 1834), (1835, 1838), (1839, 1844), (1845, 1849), (1850, 1857), (1858, 1862), (1863, 1871), (1872, 1881), (1882, 1883), (1884, 1887), (1887, 1888), (1889, 1890), (1890, 1891), (1891, 1892), (1892, 1894), (1894, 1896), (1897, 1900), (1901, 1904), (1904, 1907), (1907, 1908), (1908, 1911), (1912, 1913), (1914, 1918), (1919, 1920), (1921, 1924), (1924, 1925), (1925, 1929), (1930, 1941), (1942, 1949), (1950, 1954), (1954, 1955), (1956, 1958), (1958, 1959), (1959, 1960), (1961, 1976), (1977, 1979), (1980, 1985), (1986, 1990), (1991, 1997), (1998, 2002), (2003, 2006), (2007, 2013), (2014, 2021), (2022, 2026), (2027, 2030), (2031, 2033), (2033, 2040), (2041, 2045), (2046, 2049), (2050, 2055), (2055, 2056), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]    

# print(candidates)   
# [(0, 0, 0.6493712067604065), (167, 178, 0.9697459936141968)] 

for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}

print(result)    




