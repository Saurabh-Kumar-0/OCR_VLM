from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import streamlit as st
import torch
from PIL import Image

# Default: Load the model on the available device(s)
@st.cache_resource
def init_qwen_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor

MODEL, PROCESSOR = init_qwen_model()

# Streamlit app title
st.title("OCR Image Text Extraction")
st.subheader("I used Qwen2-VL-7B-Instruct model to get better accuracy but as it is running on CPU it takes 25-30 minutes to run it. So please have patience.")
# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add the spinner here while the model is processing
    with st.spinner("Extracting text..."):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Run Optical Character recognition on the image and don't translate Hindi to English."},
                ],
            }
        ]

        # Preparation for inference
        text = PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = PROCESSOR(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cpu")

        # Inference: Generation of the output
        generated_ids = MODEL.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        structured_output = PROCESSOR.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Convert structured output to plain text
        plain_text_output = " ".join(structured_output.split())  # Remove any extra spaces or line breaks

    # Display extracted plain text after the spinner ends
    st.subheader("Extracted Plain Text:")
    st.write(plain_text_output)

    # Keyword search functionality on plain text
    st.subheader("Keyword Search")
    search_query = st.text_input("Enter keywords to search within the extracted text")

    if search_query:
        # Check if the search query is in the plain text output
        if search_query.lower() in plain_text_output.lower():
            # Highlight the search query in the plain text
            highlighted_text = plain_text_output.replace(search_query, f"**{search_query}**", flags=re.IGNORECASE)
            st.markdown(f"Matching Text: {highlighted_text}", unsafe_allow_html=True)
        else:
            st.write("No matching text found.")
else:
    st.info("Please upload an image to extract text.")
