# %%

import numpy as np
import matplotlib.pyplot as plt

import scoreEtudiant

# %%

instance = scoreEtudiant.load_instance("data/inst1")

plt.close("all")
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 10))

for city_num in instance:
    city = instance[city_num]
    print(city_num, city)
    ax1.scatter(city["x"], city["y"], marker=".", s=100)
    ax1.annotate(city_num, (city["x"] + 0.3, city["y"] + 0.3))

    ax2.plot([city["wstart"], city["wend"]], [city_num, city_num], lw=4)

fig.tight_layout()

# %%
