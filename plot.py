import math
import random
import numpy as np
import matplotlib.pyplot as plt

########################################
# Helper Functions for Binary Strings
########################################

def index_to_binary(i, n):
    """Convert index i to a binary string of exactly n bits (MSB first)."""
    return format(i, f'0{n}b')

def all_binary_strings(n):
    """
    Return a list of all binary strings of length n.
    For n=0, return the list with the empty string.
    """
    if n == 0:
        return [""]
    return [format(i, f'0{n}b') for i in range(2 ** n)]

def degree_distribution(terms):
    """
    Given a list of binary strings (terms), return a dictionary
    mapping the degree (number of 1's) to the count.
    """
    dist = {}
    for term in terms:
        d = term.count('1')
        dist[d] = dist.get(d, 0) + 1
    return dist

########################################
# Joint Probability Distribution Functions
########################################
########################################
# Uniform Distribution Functions
########################################

def standard_uniform_pdf(x, a=0, b=1):
    """Compute the Uniform PDF over [a, b]. Returns 0 if x is outside [a,b]."""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def create_joint_probabilities_uniform(n, a=0, b=1):
    """
    Create a joint probability array of length 2^n using a uniform distribution over [a, b].
    Each index i is mapped linearly to a value x in [a, b] and the PDF is evaluated at x.
    The resulting probabilities are normalized.
    """
    joint_probs = []
    for i in range(2 ** n):
        max_val = (2 ** n) - 1
        x = a + (i * (b - a) / max_val)
        val = standard_uniform_pdf(x, a, b)
        joint_probs.append(val)
    total = sum(joint_probs)
    return [p / total for p in joint_probs]

########################################
# Exponential Distribution Functions
########################################

def standard_exponential_pdf(x, lambd=1.0):
    """
    Compute the exponential PDF with rate parameter lambd.
    Returns 0 for x < 0.
    """
    if x < 0:
        return 0.0
    return lambd * math.exp(-lambd * x)

def create_joint_probabilities_exponential(n, lambd=1.0, L=10.0):
    """
    Create a joint probability array of length 2^n using an exponential PDF.
    Each index i is mapped linearly to a value x in [0, L] and evaluated by the exponential PDF.
    The results are then normalized.
    """
    joint_probs = []
    for i in range(2 ** n):
        max_val = (2 ** n) - 1
        x = (i * L) / max_val
        joint_probs.append(standard_exponential_pdf(x, lambd))
    total = sum(joint_probs)
    return [p / total for p in joint_probs]

########################################
# Poisson Distribution Functions
########################################

def standard_poisson_pmf(k, mu=3):
    """
    Compute the Poisson PMF for a nonnegative integer k with mean mu.
    """
    if k < 0:
        return 0.0
    return math.exp(-mu) * (mu ** k) / math.factorial(k)

def create_joint_probabilities_poisson(n, mu=3):
    """
    Create a joint probability array of length 2^n using a Poisson PMF.
    Here, the index i is directly used as the outcome value (i.e. k in the Poisson PMF).
    This effectively generates a truncated Poisson distribution, and the probabilities are normalized.
    """
    joint_probs = []
    for i in range(2 ** n):
        joint_probs.append(standard_poisson_pmf(i, mu))
    total = sum(joint_probs)
    return [p / total for p in joint_probs]

########################################
# Weibull Distribution Functions
########################################

def standard_weibull_pdf(x, k=1.5, lam=1.0):
    """
    Compute the Weibull PDF at x with shape parameter k and scale parameter lam.
    Returns 0 for x < 0.
    """
    if x < 0:
        return 0.0
    return (k / lam) * (x / lam)**(k - 1) * math.exp(-(x / lam)**k)

def create_joint_probabilities_weibull(n, k=1.5, lam=1.0, L=10.0):
    """
    Create a joint probability array of length 2^n using a Weibull PDF.
    Each index i is mapped linearly to a value x in [0, L] and the PDF is evaluated at x.
    The output is normalized.
    """
    joint_probs = []
    for i in range(2 ** n):
        max_val = (2 ** n) - 1
        x = (i * L) / max_val
        joint_probs.append(standard_weibull_pdf(x, k, lam))
    total = sum(joint_probs)
    return [p / total for p in joint_probs]

########################################
# Gamma Distribution Functions
########################################

def standard_gamma_pdf(x, k=2.0, theta=2.0):
    """
    Compute the Gamma PDF at x with shape parameter k and scale parameter theta.
    Returns 0 for x < 0.
    """
    if x < 0:
        return 0.0
    return (1 / (math.gamma(k) * (theta ** k))) * (x ** (k - 1)) * math.exp(-x / theta)

def create_joint_probabilities_gamma(n, k=2.0, theta=2.0, L=10.0):
    """
    Create a joint probability array of length 2^n using a Gamma PDF.
    Each index i is mapped linearly to a value x in [0, L].
    The resulting probabilities are normalized.
    """
    joint_probs = []
    for i in range(2 ** n):
        max_val = (2 ** n) - 1
        x = (i * L) / max_val
        joint_probs.append(standard_gamma_pdf(x, k, theta))
    total = sum(joint_probs)
    return [p / total for p in joint_probs]

def standard_gaussian_pdf(x):
    """Compute the standard Gaussian PDF value at x."""
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def create_joint_probabilities(n):
    """
    Create a joint probability array of length 2^n using a standard Gaussian PDF.
    Each index i is mapped to a value in [-3, 3]. The result is normalized.
    """
    joint_probs = []
    for i in range(2 ** n):
        max_val = (2 ** n) - 1
        x = (i * 6 / max_val) - 3
        val = standard_gaussian_pdf(x)
        joint_probs.append(val)
    total = sum(joint_probs)
    return [x / total for x in joint_probs]

# --- Chi-Square Distribution Functions ---

def standard_chi_square_pdf(x, df=2):
    """
    Compute the chi-square PDF with 'df' degrees of freedom at x.
    For x < 0, return 0.
    """
    if x < 0:
        return 0.0
    coeff = 1 / (2 ** (df / 2) * math.gamma(df / 2))
    return coeff * (x ** (df / 2 - 1)) * math.exp(-x / 2)

def convert_range_chi_square(i, n, L=10):
    """
    Map an index i from [0, 2^n - 1] to the range [0, L] for chi-square.
    """
    max_val = (2 ** n) - 1
    return (i * L) / max_val

def create_joint_probabilities_chi_square(n, df=2, L=10):
    """
    Create a joint probability array of length 2^n using a chi-square PDF.
    Each index i is mapped to a value in [0, L]. The result is normalized.
    """
    joint_probs = []
    for i in range(2 ** n):
        x_mapped = convert_range_chi_square(i, n, L)
        val = standard_chi_square_pdf(x_mapped, df)
        joint_probs.append(val)
    total = sum(joint_probs)
    return [x / total for x in joint_probs]

########################################
# Conditional Probability Table Functions
########################################

def generate_conditional_probability_table(n, joint_probs):
    """
    For each stage i (0-based, where stage 0 gives the unconditional probability for x0)
    and for each history h (a binary string of length i), compute:
        P(x_i = 1 | h)
    Returns a dictionary: cond_table[i][h] = probability.
    """
    cond_table = {}
    for i in range(n):
        cond_table[i] = {}
        if i == 0:
            num = 0.0
            den = 0.0
            for idx, prob in enumerate(joint_probs):
                bstr = index_to_binary(idx, n)
                den += prob
                if bstr[0] == '1':
                    num += prob
            cond_table[0][""] = num / den if den > 0 else 0.0
        else:
            for h in all_binary_strings(i):
                num = 0.0
                den = 0.0
                for idx, prob in enumerate(joint_probs):
                    bstr = index_to_binary(idx, n)
                    if bstr[:i] == h:
                        den += prob
                        if bstr[i] == '1':
                            num += prob
                cond_table[i][h] = (num / den) if den > 0 else 0.0
    return cond_table

########################################
# Compute Minterm Coefficients from Conditional Table
########################################

def compute_minterm_coeffs_from_cond_table(n, cond_table):
    """
    For each full assignment (binary string of length n), compute its coefficient
    by chaining the conditional probabilities.

    For a binary string b of length n:
      v(b) = ∏_{i=0}^{n-1} [ if b[i]=='1' then P(x_i=1 | b[:i]) else (1 - P(x_i=1 | b[:i])) ]

    Returns a dictionary mapping each binary string to its coefficient.
    """
    minterm_coeffs = {}
    for b in all_binary_strings(n):
        prob = 1.0
        for i in range(n):
            history = b[:i]
            p = cond_table[i][history]
            prob *= p if b[i] == '1' else (1 - p)
        minterm_coeffs[b] = prob
    return minterm_coeffs

########################################
# DP-Based Optimal Quantization for Stage Coefficients
########################################

def kl_divergence(p, q, eps=1e-12):
    """
    Compute the KL divergence between two Bernoulli distributions with parameters p and q.
    Values are clamped to avoid log(0).
    """
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def dp_quantize_stage_coeffs(stage_coeffs, error_bound):
    """
    Use a DP-inspired segmentation to partition the coefficient values (for a given stage)
    into clusters such that the total KL divergence error within each cluster is below error_bound.

    This method:
      1. Sorts the (history, value) pairs by value.
      2. Precomputes the cumulative KL divergence cost for every contiguous interval.
      3. Uses a greedy segmentation (DP-like) to choose the longest interval starting at index i
          whose total cost is below error_bound.
      4. Assigns each cluster the optimal representative (the average over that interval).

    Returns:
      quantized_stage: dictionary mapping each history to its quantized coefficient.
      num_clusters: number of clusters used.
    """
    # Sort items by coefficient value.
    sorted_items = sorted(stage_coeffs.items(), key=lambda x: x[1])
    n = len(sorted_items)
    A = [val for (_, val) in sorted_items]

    # Precompute cumulative cost for each interval [i, j] and centroid.
    # For each interval, cost = sum_{k=i}^{j} KL(p_k || q) with q = average over [i,j]
    cost_matrix = [[0.0] * n for _ in range(n)]
    centroid_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        cumulative = 0.0
        for j in range(i, n):
            cluster = A[i:j+1]
            q = sum(cluster) / len(cluster)
            # Recompute cost for interval [i,j]
            cost = 0.0
            for p in cluster:
                cost += kl_divergence(p, q)
            cost_matrix[i][j] = cost
            centroid_matrix[i][j] = q

    # Greedy segmentation: for each starting index i, find the farthest j with cost <= error_bound.
    clusters = []
    i = 0
    while i < n:
        j = i
        # Increase j until the cost exceeds error_bound
        while j < n and cost_matrix[i][j] <= error_bound:
            j += 1
        # Cluster covers indices i to j-1.
        clusters.append((i, j))
        i = j

    # Assign each coefficient its quantized value (the optimal representative for its cluster).
    quantized_stage = {}
    for (start, end) in clusters:
        q = centroid_matrix[start][end - 1]
        for idx in range(start, end):
            key = sorted_items[idx][0]
            quantized_stage[key] = q

    num_clusters = len(clusters)
    return quantized_stage, num_clusters

########################################
# QM Reduction Functions (Basic Implementation)
########################################

def combine_terms(terms):
    """
    Given a set of terms (binary strings, with no '-' initially), combine pairs that differ in exactly one position.
    Returns a tuple (new_terms, used_terms) where new_terms is a set of combined terms (with '-' as don't care)
    and used_terms is the set of terms that were combined.
    """
    new_terms = set()
    used = set()
    terms = list(terms)
    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            term1 = terms[i]
            term2 = terms[j]
            diff_count = 0
            diff_index = -1
            for k in range(len(term1)):
                if term1[k] != term2[k]:
                    if term1[k] == '-' or term2[k] == '-':
                        diff_count = 10
                        break
                    diff_count += 1
                    diff_index = k
            if diff_count > 1:
                break
            if diff_count == 1:
                combined = list(term1)
                combined[diff_index] = '-'
                new_terms.add("".join(combined))
                used.add(term1)
                used.add(term2)
    return new_terms, used

def qm_reduce(terms):
    """
    Perform a basic Quine-McCluskey reduction on a set of minterms (binary strings with no '-' initially).
    Returns a list of prime implicants (with '-' indicating combined positions).
    """
    current_terms = set(terms)
    prime_implicants = set()
    while True:
        new_terms, used = combine_terms(current_terms)
        prime_implicants.update(current_terms - used)
        if not new_terms:
            break
        current_terms = new_terms
    return list(prime_implicants)

def count_degrees(implicants):
    """
    Given a list of implicants (strings with '-' for don't cares),
    returns a dictionary mapping the degree (number of non-dash characters) to the count.
    For example, "1-" has degree 1.
    """
    deg_counts = {}
    for imp in implicants:
        d = sum(1 for ch in imp if ch != '-')
        deg_counts[d] = deg_counts.get(d, 0) + 1
    return deg_counts

########################################
# QM Reduction Per Stage by Coefficient Grouping
########################################

def qm_reduce_stage(quantized_stage):
    """
    For a given stage, where quantized_stage is a dictionary mapping histories (minterms)
    to quantized coefficients, group the minterms by their quantized coefficient (rounded)
    and then apply QM reduction within each group.
    Returns the combined list of prime implicants for the stage.
    """
    groups = {}
    for term, coeff in quantized_stage.items():
        key = round(coeff, 8)
        groups.setdefault(key, []).append(term)
    reduced_terms = []
    for key, terms in groups.items():
        prime_implicants = qm_reduce(terms)
        reduced_terms.extend(prime_implicants)
    return reduced_terms

########################################
# Simulation Functions for Conditional Generation
########################################

def generate_conditional_random_number(n, cond_table):
    """
    Generate an n-bit number using the conditional probability table.
    For stage 0, use cond_table[0][""] as the probability for the first bit.
    For stage i (i>=1), use the history (binary string) to pick the corresponding probability.
    """
    bits = []
    p0 = cond_table[0][""]
    x0 = 1 if random.random() < p0 else 0
    bits.append(x0)
    for i in range(1, n):
        history = "".join(str(b) for b in bits)
        p = cond_table[i][history]
        xi = 1 if random.random() < p else 0
        bits.append(xi)
    num = 0
    for b in bits:
        num = (num << 1) | b
    return num

def simulate_conditional_model(n, cond_table, iterations=10000):
    """
    Simulate the original conditional probability method.
    Returns a frequency dictionary of generated n-bit numbers.
    """
    freq = {i: 0 for i in range(2 ** n)}
    for _ in range(iterations):
        num = generate_conditional_random_number(n, cond_table)
        freq[num] += 1
    return freq

def simulate_quantized_model(n, quantized_cond_table, iterations=10000):
    """
    Simulate the conditional probability method using the quantized conditional table.
    Returns a frequency dictionary of generated n-bit numbers.
    """
    freq = {i: 0 for i in range(2 ** n)}
    for _ in range(iterations):
        num = generate_conditional_random_number(n, quantized_cond_table)
        freq[num] += 1
    return freq

########################################
# Main Execution: Build Polynomials, Quantize per Stage, Apply QM Reduction, and Report Degree Counts
########################################
n = 16  # n-bit system; stages 0 to n-1

    # Choose distribution: use_chi_square = True for chi-square; else use Gaussian.
use_chi_square = True
if use_chi_square:
    df = 20
    L = 80
    joint_probs = create_joint_probabilities_chi_square(n, df=df, L=L)
    print("Using chi-square distribution for joint probabilities.")
else:
    #joint_probs = create_joint_probabilities_gamma(n, k=2.0, theta=2.0, L=10.0)
    joint_probs = create_joint_probabilities(n)
    #joint_probs = create_joint_probabilities_exponential(n)
    #joint_probs = create_joint_probabilities_poisson(n, mu=3)
    #joint_probs = create_joint_probabilities_uniform(n, a=0, b=1)
    #joint_probs = create_joint_probabilities_weibull(n, k=1.5, lam=1.0, L=10.0)
    print("Using Gaussian distribution for joint probabilities.")

    # Build the conditional probability table.
cond_table = generate_conditional_probability_table(n, joint_probs)

    # Quantize the conditional table coefficients for each stage using DP-based optimal quantization.
    # (Try adjusting error_bound; for example, 0.01 may be too tight.)
error_bound = 1e-4  # maximum allowed total KL divergence error per cluster
quantized_cond_table = {}
total_original = 0
total_quantized = 0
for stage in range(n):
    stage_coeffs = cond_table[stage]
    total_original += len(stage_coeffs)
    #quantized_stage, num_clusters = dp_quantize_stage_coeffs(stage_coeffs, error_bound)
    quantized_stage, num_clusters = dp_quantize_stage_coeffs(stage_coeffs, error_bound/np.sqrt(stage+1))
    total_quantized += num_clusters
    print(f"\nStage {stage} (domain size = 2^{stage} = {2 ** stage}):")
    print(f"  Original distinct coefficients: {len(stage_coeffs)}")
    print(f"  Quantized distinct coefficients: {num_clusters} (error bound = {error_bound/np.sqrt(stage+1)})")
    quantized_cond_table[stage] = quantized_stage
print("\nTotal number of coefficients over all stages (original):", total_original)
print("Total number of coefficients over all stages (quantized):", total_quantized)

    # Compute full minterm coefficients from the quantized conditional table.
minterm_coeffs_quantized = compute_minterm_coeffs_from_cond_table(n, quantized_cond_table)

    # For each stage, compute the original and QM-reduced degree distributions.
overall_orig_components = 0
overall_qm_components = 0
overall_orig_deg = {}
overall_qm_deg = {}
'''
    print("\nQM Reduction Degree Counts per Stage:")
    for stage in range(n):
        orig_terms = list(quantized_cond_table[stage].keys())
        orig_deg_dist = degree_distribution(orig_terms)
        overall_orig_components += len(orig_terms)
        reduced_terms = qm_reduce_stage(quantized_cond_table[stage])
        qm_deg_dist = count_degrees(reduced_terms)
        overall_qm_components += len(reduced_terms)
        for d, count in orig_deg_dist.items():
            overall_orig_deg[d] = overall_orig_deg.get(d, 0) + count
        for d, count in qm_deg_dist.items():
            overall_qm_deg[d] = overall_qm_deg.get(d, 0) + count
        print(f"  Stage {stage}:")
        print(f"    Original Degree Distribution: {orig_deg_dist}")
        print(f"    QM Reduced Degree Distribution: {qm_deg_dist}")

    print("\nTotal number of components over all stages (original):", overall_orig_components)
    print("Total number of components over all stages (after QM reduction):", overall_qm_components)
    print("\nOverall Original Degree Distribution over all stages:")
    for d in sorted(overall_orig_deg.keys()):
        print(f"  Degree {d}: {overall_orig_deg[d]}")
    print("\nOverall QM Reduced Degree Distribution over all stages:")
    for d in sorted(overall_qm_deg.keys()):
        print(f"  Degree {d}: {overall_qm_deg[d]}")
'''

# Set the number of iterations for simulation.
iterations = 10000000

# 5. Simulate the Quantized (Optimal Partition) model.
freq_AR = simulate_quantized_model(n, quantized_cond_table, iterations)
print("\nFrequency Distribution using Optimal Partition Model (over", iterations, "iterations):")

# 6. Simulate the Conditional model.
freq_cond = simulate_conditional_model(n, cond_table, iterations)
print("\nFrequency Distribution using Conditional Model (over", iterations, "iterations):")

# Prepare data for plotting.
numbers = list(range(2 ** n))
frequencies_AR = [freq_AR[i] for i in numbers]
frequencies_cond = [freq_cond[i] for i in numbers]
def convert_range_3(x, n):
    # Map x from [0, 2^n - 1] to [-3, 3]
    max_val = (2 ** n) - 1
    return (x * 6 / max_val) - 3

def convert_range_20(x, n):
    # Map x from [0, 2^n - 1] to [0, 20]
    max_val = (2 ** n) - 1
    return (x * 80 / max_val)  # Changed from 6 to 20 and shifted range

# Compute x-values corresponding to each index (from 0 to 20)
x_vals = [convert_range_20(i, n) for i in range(2**n)]
# Compute x-values corresponding to each index (from -3 to 3)
#x_vals = [convert_range_3(i, n) for i in range(2**n)]
numbers = list(range(2**n))

# Compute probabilities from frequencies.
probabilities_AR = [freq_AR[i] / iterations for i in numbers]
probabilities_cond = [freq_cond[i] / iterations for i in numbers]

########################################
# Compute KL Divergence and MSE between Distributions
########################################

def compute_mse(true_dist, est_dist):
    """
    Compute Mean Squared Error between two discrete distributions.
    """
    true_arr = np.array(true_dist)
    est_arr = np.array(est_dist)
    mse = np.sum((true_arr - est_arr) ** 2)
    return mse * 1e5

def compute_kl(true_dist, est_dist, epsilon=1e-12):
    """
    Compute KL divergence D_KL(true_dist || est_dist) for two discrete distributions.
    """
    true_arr = np.array(true_dist) + epsilon
    est_arr = np.array(est_dist) + epsilon
    return np.sum(true_arr * np.log(true_arr / est_arr))

mse_ar = compute_mse(joint_probs, probabilities_AR)
mse_cond = compute_mse(joint_probs, probabilities_cond)
kl_ar = compute_kl(joint_probs, probabilities_AR)
kl_cond = compute_kl(joint_probs, probabilities_cond)

print("\nComparison between the original PDF and generated distributions:")
print(f"MSE Error (Optimal Partition Model vs Original): {mse_ar:.6f}")
print(f"MSE Error (Conditional Model vs Original): {mse_cond:.6f}")
print(f"KL Divergence (Optimal Partition Model vs Original): {kl_ar:.6f}")
print(f"KL Divergence (Conditional Model vs Original): {kl_cond:.6f}")

########################################
# Plot frequency distributions side-by-side
########################################

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# Plot True Distribution.
axes[0].plot(x_vals, joint_probs, label="True Distribution", color="blue", linestyle="--", linewidth=2)
axes[0].set_xlabel("x, KL divergence = 0")
axes[0].set_ylabel("Probability")
axes[0].set_title("True Distribution, Chi-squared, df=20")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)

# Plot Optimal Partition Model (Quantized).
axes[1].plot(x_vals, probabilities_AR, label="Optimal Partition Model", color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel(f"{n}bit, {iterations} , KL divergence = {kl_ar:.6f}, {total_quantized} elements")
axes[1].set_title("Optimal Partition Model")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)

# Plot Conditional Model.
axes[2].plot(x_vals, probabilities_cond, label="Conditional Model", color="green", linestyle="--", linewidth=1)
axes[2].set_xlabel(f"{n}bit, {iterations} , KL divergence = {kl_cond:.6f}, {total_original} elements")
axes[2].set_title("Conditional Model")
axes[2].legend()
axes[2].grid(True, linestyle="--", alpha=0.6)

# Set y-axis of all subplots to start at 0.
for ax in axes:
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("8_chi_df_20.png", dpi=1000)
plt.show()