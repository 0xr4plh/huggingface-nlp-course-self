import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def qa_app():
    st.title("Question Answering App")

    # Sidebar for context input
    st.sidebar.title("Context")
    long_context = st.sidebar.text_area("Enter the context", height=400)

    # Sidebar for question input
    st.sidebar.title("Question")
    question = st.sidebar.text_input("Enter your question")

    if st.sidebar.button("Get Answer"):
        model_checkpoint = "distilbert-base-cased-distilled-squad"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

        inputs = tokenizer(question, long_context)
        tokens = inputs.tokens()

        inputs = tokenizer(
            question,
            long_context,
            stride=128,
            max_length=384,
            padding="longest",
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        _ = inputs.pop("overflow_to_sample_mapping")
        offsets = inputs.pop("offset_mapping")

        inputs = inputs.convert_to_tensors("pt")

        output = model(**inputs)
        start_logits = output.start_logits
        end_logits = output.end_logits

        mask = [i != 1 for i in inputs.sequence_ids()]
        mask[0] = False
        mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

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

        for candidate, offset in zip(candidates, offsets):
            start_token, end_token, score = candidate
            start_char, _ = offset[start_token]
            _, end_char = offset[end_token]
            answer = long_context[start_char:end_char]
            result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
            st.write(result)

if __name__ == "__main__":
    qa_app()