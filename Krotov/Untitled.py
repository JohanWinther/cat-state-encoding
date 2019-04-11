
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
guesses_fb = [233, 212, 425, 342, 360, 270, 69, 411, 612, 318, 200, 414, 413, 210, 253, 311, 430, 345, 213, 371, 262, 417, 378, 632, 417, 382, 385, 447, 168, 297, 333, 956, 328, 564, 420]
guesses_insta = [187, 440, 232, 260, 289, 282, 328, 337, 300, 347, 135, 267, 493, 356, 389, 312, 452, 318, 392, 324, 495, 462, 516, 440, 374, 358, 233, 338, 272, 412, 600, 574, 500, 282, 376, 210, 391, 476, 259, 493, 348, 402, 403, 473, 473]
guesses = guesses_fb + guesses_insta
g_mean = np.mean(guesses)
g_median = np.median(guesses)
plt.hist(guesses, int(1 + 3.3 * np.log(len(guesses)))*3)
plt.axvline(g_mean, color='r', linestyle='--')
plt.axvline(g_median, color='r', linestyle='--')
plt.xlabel('Antal pinnar')
plt.show()
print('Mean: {}'.format(g_mean))
print('Median: {}'.format(g_median))

