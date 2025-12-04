import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import scipy.stats as stats

# ---------- PRIOR ----------
def get_prior():
    gender = st.selectbox("Select your gender:", ["Male", "Female", "Other"])
    if gender == "Male":
        return 0.08
    elif gender == "Female":
        return 0.005
    else:
        return 0.33 * 0.08 + 0.67 * 0.005

# ---------- COLORBLINDNESS TEST ----------
def colorblindness_test(test_index):
    # Initialize test-specific dots if not already
    if test_index not in st.session_state.test_numbers:
        num_dots = 3000
        st.session_state.dot_x = np.random.rand(num_dots)
        st.session_state.dot_y = np.random.rand(num_dots)
        st.session_state.dot_colors[test_index] = np.random.rand(num_dots, 3)
        st.session_state.test_numbers[test_index] = np.random.randint(0, 100)

    correct_value = st.session_state.test_numbers[test_index]
    colors = st.session_state.dot_colors[test_index]

    # Plot the dots and number
    fig, ax = plt.subplots(figsize=(4,4))
    circle = plt.Circle((0.5, 0.5), 0.48, color="white", zorder=1)
    ax.add_patch(circle)
    ax.scatter(st.session_state.dot_x, st.session_state.dot_y, c=colors, s=40, zorder=2)
    number_color = np.random.rand(3)
    ax.text(0.5, 0.5, str(correct_value), fontsize=55, ha='center', va='center', color=number_color, zorder=3)
    ax.axis('off')
    st.pyplot(fig)

    # Compute contrast for likelihoods
    bg_brightness = np.mean(np.mean(colors, axis=0))
    num_brightness = np.mean(number_color)
    contrast = abs(num_brightness - bg_brightness)
    p_correct_if_not_cb = 0.7 + 0.25 * contrast
    p_correct_if_cb = 0.05 + 0.1 * contrast

    # Input from user
    guess = st.text_input(f"Enter the number you see for Test {test_index+1}:", key=f"input_{test_index}")
    if guess.isdigit():
        guess = int(guess)
        correct_for_user = (guess == correct_value)
        return correct_for_user, correct_value, p_correct_if_cb, p_correct_if_not_cb
    else:
        return None, correct_value, p_correct_if_cb, p_correct_if_not_cb

# ---------- POSTERIOR CALCULATION ----------
def calculate_posterior(prior, results, likelihoods_cb, likelihoods_not_cb):
    likelihood_cb = 1.0
    likelihood_not_cb = 1.0
    for i in range(len(results)):
        r = results[i]
        p_cb = likelihoods_cb[i]
        p_not_cb = likelihoods_not_cb[i]
        if r == 1:
            likelihood_cb *= p_cb
            likelihood_not_cb *= p_not_cb
        else:
            likelihood_cb *= (1 - p_cb)
            likelihood_not_cb *= (1 - p_not_cb)
    posterior = (prior * likelihood_cb) / (prior * likelihood_cb + (1 - prior) * likelihood_not_cb)
    return posterior

# ---------- MAIN APP ----------
def main():
    st.title("Colorblindness Probability Test")
    prior = get_prior()
    st.write(f"Your prior probability of colorblindness is: {prior:.3f}")

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.likelihoods_cb = []
        st.session_state.likelihoods_not_cb = []
        st.session_state.test_numbers = {}  # store correct numbers per test
        st.session_state.dot_colors = {}    # store colors per test

    num_tests = st.slider("How many tests do you want to run?", 1, 10, 3)

    # Run tests
    for i in range(num_tests):
        correct, correct_value, p_cb, p_not_cb = colorblindness_test(i)
        if correct is not None:
            if len(st.session_state.results) <= i:
                st.session_state.results.append(1 if correct else 0)
                st.session_state.likelihoods_cb.append(p_cb)
                st.session_state.likelihoods_not_cb.append(p_not_cb)

            st.success("Correct!" if correct else f"Wrong! Correct answer: {correct_value}")

            posterior = calculate_posterior(prior, st.session_state.results, 
                                            st.session_state.likelihoods_cb, 
                                            st.session_state.likelihoods_not_cb)
            st.write(f"Updated posterior probability of being colorblind: {posterior:.3f}")

    # ---------- Beta Posterior ----------
    if st.session_state.results:
        alpha_prior = prior * 10
        beta_prior = (1 - prior) * 10
        alpha_post = alpha_prior + sum(st.session_state.results)
        beta_post = beta_prior + len(st.session_state.results) - sum(st.session_state.results)
        posterior_mean = alpha_post / (alpha_post + beta_post)
        posterior_std = np.sqrt((alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)))

        st.write(f"Posterior mean probability: {posterior_mean:.3f}")
        st.write(f"Posterior standard deviation: {posterior_std:.3f}")

        # Plot prior vs posterior
        x = np.linspace(0, 1, 100)
        prior_pdf = stats.beta.pdf(x, alpha_prior, beta_prior)
        posterior_pdf = stats.beta.pdf(x, alpha_post, beta_post)
        fig2, ax2 = plt.subplots()
        ax2.plot(x, prior_pdf, label="Prior")
        ax2.plot(x, posterior_pdf, label="Posterior")
        ax2.set_xlabel("Probability of Being Colorblind")
        ax2.set_ylabel("Density")
        ax2.set_title("Prior vs Posterior Distribution")
        ax2.legend()
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
