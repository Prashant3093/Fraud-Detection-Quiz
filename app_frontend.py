import streamlit as st
import requests
import random

# Function to get a new transaction based on difficulty
def get_new_transaction(score):
    if score < 5:
        difficulty = "Easy"
        amount = round(random.uniform(10, 100), 2)  # Lower amounts for early levels
    elif score < 15:
        difficulty = "Medium"
        amount = round(random.uniform(100, 500), 2)  # Medium amounts
    elif score < 25:
        difficulty = "Hard"
        amount = round(random.uniform(500, 1000), 2)  # Larger amounts
    else:
        difficulty = "Very Hard"
        amount = round(random.uniform(1000, 5000), 2)  # Extreme amounts
        # Rare merchants for higher difficulty
    merchant_list = ["Walmart", "Amazon", "Target", "Crypto Exchange", "Unknown Merchant"]
    if score > 20:
        merchant_list = ["Black Market", "Luxury Item Dealer", "Rare Crypto Exchange", "Offshore Account"]

    return {
        "merchant": random.choice(merchant_list),
        "amount": amount,
        "category": random.choice(["Groceries", "Electronics", "Luxury Items"]),
        "city": random.choice(["New York", "Los Angeles", "Random Small Town"]),
        "state": random.choice(["NY", "CA", "TX", "Unknown"]),
        "time": f"{random.randint(0, 23)}:{random.randint(0, 59)}",
        "gender": random.choice(["Male", "Female"]),
        "difficulty": difficulty
    }

# Function to get prediction from backend
def get_prediction(transaction):
    response = requests.post("http://127.0.0.1:5000/predict", json={"features": transaction})
    return response.json()

# Initialize session state if not already done
if 'transaction' not in st.session_state:
    st.session_state.transaction = get_new_transaction(st.session_state.get("score", 0))
    st.session_state.score = 0
    st.session_state.high_score = 0
    st.session_state.game_over = False  # Initialize game_over to False

# Display transaction and difficulty
st.subheader(f"Transaction Details (Difficulty: {st.session_state.transaction['difficulty']})")
for key, value in st.session_state.transaction.items():
    if key != "difficulty":
        st.write(f"**{key.capitalize()}**: {value}")

# User input for fraud prediction
user_input = st.radio("Is this transaction a fraud?", ("Yes", "No"))

# Submit button to check answer
if st.button("Submit"):
    if not st.session_state.game_over:
        prediction = get_prediction(st.session_state.transaction)
        is_fraud = prediction.get("is_fraud")

        # Check if the user's answer is correct
        if (is_fraud and user_input == "Yes") or (not is_fraud and user_input == "No"):
            st.session_state.score += 1
            st.success("Correct! Moving on to the next transaction.")
            st.session_state.transaction = get_new_transaction(st.session_state.score)  # Load a new transaction
        else:
            st.session_state.game_over = True  # Set game over when the user is wrong
            st.error("Incorrect. Game Over! Press 'Restart' to try again.")

    # Update high score
    if st.session_state.score > st.session_state.high_score:
        st.session_state.high_score = st.session_state.score

# Display score and high score
st.write(f"Score: {st.session_state.score}")
st.write(f"High Score: {st.session_state.high_score}")

# Show "Restart" button if game is over
if st.session_state.game_over:
    if st.button("Restart"):
        st.session_state.score = 0
        st.session_state.game_over = False  # Reset the game over flag
        st.session_state.transaction = get_new_transaction(st.session_state.score)
        st.success("Game Restarted! Let's play again.")
