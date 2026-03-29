from statsmodels.stats.power import FTestAnovaPower
import numpy as np
import matplotlib.pyplot as plt

# Study Parameters:
k_groups = 5 # e.g. White, Black, Hispanic, Asian, Null Baseline
alpha = 0.05 # accept 5% false positives
power_target = 0.80 # likeliness to detect real effect
effect_size_f = 0.25 # medium effect (Cohen's f)

# Solve for sample size:
analysis = FTestAnovaPower()

# Why ANOVA?
# - one continuous ourcome (resume score) is measured against more than two groups
# - controls false positive risk

n_total = analysis.solve_power(
    effect_size=effect_size_f,
    alpha=alpha,
    power=power_target,
    k_groups=k_groups
)
n_per_group = n_total / k_groups

print(f"Required n per group:  {np.ceil(n_per_group):.0f}")
print(f"Total resumes to send: {np.ceil(n_total):.0f}")

# Sensitivity sweep: divide each result by k_groups
effect_sizes = np.linspace(0.10, 0.50, 100)
sample_sizes_per_group = [
    np.ceil(analysis.solve_power(
        effect_size=f, alpha=alpha, power=power_target, k_groups=k_groups
    )) / k_groups
    for f in effect_sizes
]

plt.figure(figsize=(8, 4))
plt.plot(effect_sizes, sample_sizes_per_group, color='steelblue', linewidth=2)
plt.axvline(0.25, color='red', linestyle='--', label='Medium effect (f=0.25)')
plt.axvline(0.10, color='orange', linestyle='--', label='Small effect (f=0.10)')
plt.xlabel("Effect Size (Cohen's f)")
plt.ylabel("Required n per group")
plt.title("Power Analysis: Resume Bias Study")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()