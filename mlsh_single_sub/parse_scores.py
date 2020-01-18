import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
sns.set(style="darkgrid")
envs = []
hids = []
seeds = []

d = {}
with open('merged_results') as f:
  for line in f:
    match = re.search('(.*)-v3-hid(.*)-seed(.*).txt', line)
    if match is not None:
      env = match.group(1)
      if env not in envs:
        envs.append(env)
      hid = match.group(2)
      if hid not in hids:
        hids.append(hid)
      seed = match.group(3)
      if seed not in seeds:
        seeds.append(seed)
      key = (env, hid, seed)
      d[key] = []

    match = re.search('Episode .* return: ([0-9\.\-]+)', line)
    if match is not None:
      d[key].append(float(match.group(1)))

d2 = {}
for key in d:
  d2[key] = statistics.mean(d[key])
  d[key] = statistics.median(d[key])

median_d = {}
mean_d = {}
for env in envs:
  for hid in hids:
    largest = -1000000
    largest_mean = -100000
    for seed in seeds:
      if d[(env, hid, seed)] > largest:
        largest = d[(env, hid, seed)]
      if d2[(env, hid, seed)] > largest_mean:
        largest_mean = d2[(env, hid, seed)]

    median_d[(env, hid)] = largest
    mean_d[(env, hid)] = largest_mean

for k in median_d:
  print(k, 'median: %.4f' % median_d[k], 'mean: %.4f' % mean_d[k], sep='\t')
