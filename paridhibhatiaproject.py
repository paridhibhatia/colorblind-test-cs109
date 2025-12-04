import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------
# SESSION STATE
# --------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.prior = None
    st.session_state.num_tests = 5  # Default number of tests
    st.session_state.test_num = 0
    st.session_state.results = []  # 1 for correct, 0 for wrong
    st.session_state.likelihoods_cb = []
    st.session_state.likelihoods_not_cb = []
    st.session_state.posterior_history = []

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def calculate_posterior(prior, results, likelihoods_cb, likelihoods_not_cb):
    """Calculate posterior by multiplying all likelihoods"""
    likelihood_cb = 1.0
    likelihood_not_cb = 1.0
    
    for i in range(len(results)):
        if results[i] == 1:  # Correct
            likelihood_cb *= likelihoods_cb[i]
            likelihood_not_cb *= likelihoods_not_cb[i]
        else:  # Wrong
            likelihood_cb *= (1 - likelihoods_cb[i])
            likelihood_not_cb *= (1 - likelihoods_not_cb[i])
    
    numerator = prior * likelihood_cb
    denominator = numerator + (1 - prior) * likelihood_not_cb
    
    return numerator / denominator if denominator > 0 else prior

# --------------------------
# TITLE
# --------------------------
st.title("🎨 Paridhi's Colorblindness Bayesian Test")

if st.button("🔄 Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --------------------------
# STEP 1: GET PRIOR
# --------------------------
if st.session_state.prior is None:
    st.subheader("Step 1: What is your gender?")
    gender = st.radio(
        "Select:",
        ["Male", "Female", "Other/Prefer not to say"],
        help="This affects the prior probability based on population statistics"
    )
    
    if gender == "Male":
        prior = 0.08
    elif gender == "Female":
        prior = 0.005
    else:
        prior = 0.33 * 0.08 + 0.67 * 0.005
    
    st.info(f"Prior probability of colorblindness: **{prior:.1%}** ({prior:.4f})")
    
    st.subheader("Step 2: How many tests do you want to take?")
    num_tests = st.slider("Number of tests:", min_value=3, max_value=10, value=5, step=1)
    st.session_state.num_tests = num_tests
    
    if st.button("▶️ Start Testing"):
        st.session_state.prior = prior
        st.session_state.posterior_history.append(prior)
        st.rerun()
    
    st.stop()

# --------------------------
# GENERATE TEST
# --------------------------
st.subheader(f"Test #{st.session_state.test_num + 1} of {st.session_state.num_tests}")

# Check if all tests are completed
if st.session_state.test_num >= st.session_state.num_tests:
    st.success("🎉 All tests completed!")
    st.balloons()
    
    current_posterior = st.session_state.posterior_history[-1]
    if current_posterior < 0.01:
        st.info("Based on your performance, you are very likely **NOT colorblind**!")
    elif current_posterior < 0.05:
        st.info("Based on your performance, you are probably **NOT colorblind**, but there's some uncertainty.")
    elif current_posterior < 0.15:
        st.warning("Results are uncertain. You might want to consult an eye specialist.")
    else:
        st.warning("Based on your performance, you may be colorblind. Consider consulting an eye specialist.")
    
    st.stop()

# Use test number as seed for reproducibility
np.random.seed(st.session_state.test_num * 42)

correct_value = np.random.randint(0, 100)
num_dots = 2000  # Increased from 500 to make it much harder to see

x = np.random.rand(num_dots)
y = np.random.rand(num_dots)
dot_colors = np.random.rand(num_dots, 3)
number_color = np.random.rand(3)

# Calculate contrast
bg_brightness = np.mean(dot_colors)
num_brightness = np.mean(number_color)
contrast = abs(num_brightness - bg_brightness)

# Make tests progressively harder by reducing effective contrast
difficulty_factor = 1.0 - (st.session_state.test_num * 0.10)  # Gets harder more slowly
difficulty_factor = max(difficulty_factor, 0.5)  # Keep minimum at 50% to maintain uncertainty
contrast = contrast * difficulty_factor

# Calculate likelihoods (probability of being CORRECT)
# Made harder with less separation between colorblind/non-colorblind
p_correct_if_not_cb = min(0.50 + 0.20 * contrast, 0.75)  # 50-75% range
p_correct_if_cb = min(0.30 + 0.15 * contrast, 0.60)      # 30-60% range

# --------------------------
# DISPLAY TEST
# --------------------------
fig, ax = plt.subplots(figsize=(5, 5))
circle = plt.Circle((0.5, 0.5), 0.48, color="white", zorder=1)
ax.add_patch(circle)
ax.scatter(x, y, c=dot_colors, s=30, zorder=2, alpha=0.9)  # Smaller dots, slight transparency
ax.text(0.5, 0.5, str(correct_value), fontsize=60, ha='center', va='center',
        color=number_color, weight='bold', zorder=3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
st.pyplot(fig)

st.write(f"**Test difficulty:** Contrast = {contrast:.3f}")
st.write(f"- If NOT colorblind: {p_correct_if_not_cb:.1%} chance of correct answer")
st.write(f"- If colorblind: {p_correct_if_cb:.1%} chance of correct answer")

# --------------------------
# GET USER ANSWER
# --------------------------
user_guess = st.text_input("What number do you see?", key=f"input_{st.session_state.test_num}")

if st.button("Submit Answer"):
    if not user_guess.isdigit():
        st.error("Please enter a valid number!")
    else:
        guess = int(user_guess)
        is_correct = (guess == correct_value)
        
        # Store result
        st.session_state.results.append(1 if is_correct else 0)
        st.session_state.likelihoods_cb.append(p_correct_if_cb)
        st.session_state.likelihoods_not_cb.append(p_correct_if_not_cb)
        
        # Calculate new posterior
        posterior = calculate_posterior(
            st.session_state.prior,
            st.session_state.results,
            st.session_state.likelihoods_cb,
            st.session_state.likelihoods_not_cb
        )
        st.session_state.posterior_history.append(posterior)
        st.session_state.test_num += 1
        
        # Show result
        if is_correct:
            st.success(f"✅ Correct! The number was {correct_value}")
        else:
            st.error(f"❌ Wrong. The correct number was {correct_value}")
        
        st.write("---")
        st.write("**📊 Bayesian Update:**")
        st.write(f"- Prior (before this test): {st.session_state.posterior_history[-2]:.4f}")
        st.write(f"- Posterior (after this test): {posterior:.4f}")
        st.write(f"- Change: {posterior - st.session_state.posterior_history[-2]:+.4f}")
        
        if is_correct:
            st.write(f"✓ Since you were correct and non-colorblind people are more likely to be correct ({p_correct_if_not_cb:.1%} vs {p_correct_if_cb:.1%}), your probability of being colorblind should **decrease**.")
        else:
            st.write(f"✗ Since you were wrong and colorblind people are more likely to be wrong, your probability of being colorblind should **increase**.")
        
        st.write("---")
        
        if st.button("Continue to Next Test"):
            st.rerun()

# --------------------------
# RESULTS SUMMARY
# --------------------------
st.divider()
st.subheader("📈 Current Results")

if st.session_state.test_num > 0:
    current_posterior = st.session_state.posterior_history[-1]
    st.metric(
        label="Current Probability of Colorblindness",
        value=f"{current_posterior:.1%}",
        delta=f"{current_posterior - st.session_state.prior:+.1%}"
    )
    
    # Plot posterior history
    if len(st.session_state.posterior_history) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(st.session_state.posterior_history)), 
                st.session_state.posterior_history, 
                marker='o', linewidth=2, markersize=8)
        ax.set_xlabel("Test Number")
        ax.set_ylabel("Probability of Colorblindness")
        ax.set_title("Posterior Probability Update Over Tests")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(st.session_state.posterior_history) * 1.2)
        st.pyplot(fig)
    
    # Test history
    st.write("### Test History")
    for i in range(len(st.session_state.results)):
        result = "✅ Correct" if st.session_state.results[i] == 1 else "❌ Wrong"
        st.write(f"**Test {i+1}**: {result} | "
                f"P(colorblind): {st.session_state.posterior_history[i]:.4f} → {st.session_state.posterior_history[i+1]:.4f}")

st.write(f"**Tests completed:** {st.session_state.test_num} / {st.session_state.num_tests}")
