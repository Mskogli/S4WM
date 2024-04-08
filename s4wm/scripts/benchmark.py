# %%
import matplotlib.pyplot as plt
import numpy as np


# Data collected from benchmarking "encode and step" across a decreasing number of S4 blocks

# 6 blocks:
# 	with XLA prealloc:
# 		Forward pass avg:  0.023689283
# 		Forward pass std:  0.0015766548
# 	without XLA prealloc:
# 		Forward pass avg:  0.024566974
# 		Forward pass std:  0.0017398251
# 5 blocks:
# 	without XLA prealloc:
# 		Forward pass avg:  0.020698579
# 		Forward pass std:  0.0014936151
# 	with XLA prealloc:
# 		Forward pass avg:  0.020779556
# 		Forward pass std:  0.0018999141
# 4 blocks:
# 	without XLA prealloc:
# 		Forward pass avg:  0.017846867
# 		Forward pass std:  0.0040340535
# 	with XLA prealloc:
# 		Forward pass avg:  0.018313168
# 		Forward pass std:  0.0012083857
# 3 blocks:
# 	without XLA prealloc:
# 		Forward pass avg:  0.014845294
# 		Forward pass std:  0.0034428607
# 	with XLA prealloc:
# 		Forward pass avg:  0.014793408
# 		Forward pass std:  0.0029217482
# 2 blocks:
# 	without XLA prealloc:
# 		Forward pass avg:  0.0123723885
# 		Forward pass std:  0.0012790103
# 	with XLA prealloc:
# 		Forward pass avg:  0.012398921
# 		Forward pass std:  0.0010495259
# 1 blocks:
# 	without XLA prealloc:
# 		Forward pass avg:  0.008537367
# 		Forward pass std:  0.0031764016
# 	with XLA prealloc:
# 		Forward pass avg:  0.009201544
# 		Forward pass std:  0.0035629526

blocks = np.array([1, 2, 3, 4, 5, 6])

avg_without_xla = np.array(
    [0.008537367, 0.0123723885, 0.014845294, 0.017846867, 0.020698579, 0.024566974]
)
std_without_xla = np.array(
    [0.0031764016, 0.0012790103, 0.0034428607, 0.0040340535, 0.0014936151, 0.0017398251]
)
avg_with_xla = np.array(
    [0.009201544, 0.012398921, 0.014793408, 0.018313168, 0.020779556, 0.023689283]
)
std_with_xla = np.array(
    [0.0035629526, 0.0010495259, 0.0029217482, 0.0012083857, 0.0018999141, 0.0015766548]
)


plt.figure(figsize=(10, 6))
plt.errorbar(
    blocks,
    avg_without_xla,
    yerr=std_without_xla,
    label="XLA_PREALLOC = False",
    fmt="-o",
    color="red",
    alpha=0.5,
)

plt.errorbar(
    blocks,
    avg_with_xla,
    yerr=std_with_xla,
    label="XLA_PREALLOC = True",
    fmt="-o",
    color="blue",
    alpha=0.5,
)

plt.xlabel("Number of S4 Encoding Blocks")
plt.ylabel("FWP Averaged Over 2000 Runs [seconds]")
plt.xticks(blocks)
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# %%
