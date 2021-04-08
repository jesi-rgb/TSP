import os
import time

from elkai import solve_float_matrix
from sklearn.metrics.pairwise import euclidean_distances

from bullet import Bullet

data_files = [f.split('.')[0] for f in os.listdir("data")]
# assist user in selecting file
cli = Bullet(prompt = "Choose from one of the following problems to solve:", 
            choices= data_files,
            bullet = "→ ")
usr_file = cli.launch()

# read the file 
with open('data/{}.tsp'.format(usr_file)) as file:
    lines = file.readlines()

# find out where the metadata ends and the datapoints begin
start_index = [i+1 for i, x in enumerate(lines) if x.startswith("NODE_COORD_SECTION")][0]

# extract metadata
metadata = [x.strip() for x in lines[:start_index]]


# save name and dimensions of the problem
name = metadata[0].split()[2] # NAME
dimension = [int(i.strip().split()[2]) for i in lines if i.startswith("DIMENSION")][0]



# for every line, take the points and store them
nodelist = []
for i in range(start_index, dimension + start_index - 1):
    x, y = lines[i].strip().split()[1:]
    nodelist.append([float(x), float(y)])


# calculate the euclidean distances between every point.
# this creates a matrix of distances between every pair 
# of points. now we are ready to calculate the optimal path.
dists = euclidean_distances(nodelist)

print()
print("> Starting execution for file: {}".format(name))
print("> Matrix dimensions: {}".format(dists.shape))

print("> Calculating solution. \nProblems over 500 cities can take several minutes...\n")

start_time = time.time()
print("> Proposed solution: {}\n".format(solve_float_matrix(dists, runs=10)))

end_time = time.time() - start_time
hours, rem = divmod(end_time, 3600)
minutes, seconds = divmod(rem, 60)
print("------------- Process terminated in {:0>2} hours, {:0>2} minutes and {:05.2f} seconds. -------------".format(int(hours),int(minutes), seconds))
