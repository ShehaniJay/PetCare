import streamlit as st
import random

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="AI Pet Chatbot", page_icon="💬")

st.title("💬 AI Pet Health Assistant Chatbot")
st.write("Ask me anything about pet care, health, or training!")

# ------------------------
# Initialize Chat History
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------
# Simple Knowledge Base
# ------------------------
def generate_response(user_input):
    user_input = user_input.lower()

    # Feeding
    if "food" in user_input or "eat" in user_input:
        return "Ensure your pet gets balanced nutrition based on its breed, age, and weight. Avoid chocolate, onions, and grapes."

    # Vaccination
    elif "vaccine" in user_input:
        return "Pets should follow a proper vaccination schedule. Consult your veterinarian for core and optional vaccines."

    # Fever
    elif "fever" in user_input or "temperature" in user_input:
        return "Normal body temperature for dogs is 38-39.2°C. For cats, it's 38-39°C. If higher, consult a vet."

    # Vomiting
    elif "vomit" in user_input:
        return "Occasional vomiting may be minor, but frequent vomiting requires veterinary attention."

    # Training
    elif "train" in user_input:
        return "Positive reinforcement training works best. Reward good behavior with treats and praise."

    # Grooming
    elif "groom" in user_input:
        return "Regular grooming keeps your pet healthy. Brush fur weekly and bathe only when necessary."

    # Breed info
    elif "breed" in user_input:
        return "You can use our Breed Identification tool to identify your pet's breed."

    # Disease prediction
    elif "disease" in user_input:
        return "Use the Disease Prediction tool for symptom-based condition analysis."

    # Default response
    else:
        responses = [
            "I recommend monitoring your pet closely and consulting a veterinarian if symptoms persist.",
            "Can you provide more details about your pet's symptoms?",
            "Maintaining a healthy diet and regular exercise is important for pets.",
            "That’s an interesting question! Tell me more about your pet."
        ]
        return random.choice(responses)

# ------------------------
# Display Chat History
# ------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------
# Chat Input
# ------------------------
if prompt := st.chat_input("Ask about pet health..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = generate_response(prompt)

    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)