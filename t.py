import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("train.csv")

"""
Save the image in the folder
"""

img = df.iloc[0, 1:].values.reshape(28, 28)

plt.imshow(img)
plt.axis("off")
plt.savefig(f"{df.iloc[0,0]}.png")
