
import random
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import pickle
import csv
# Read CSV file into a pandas DataFrame
df = pd.read_csv('lonlatcal.csv')

# Find minimum of first column
min_col1 = df.iloc[:, 0].min()
max_col1 = df.iloc[:, 0].max()
# Find minimum of second column
min_col2 = df.iloc[:, 1].min()
max_col2 = df.iloc[:, 1].max()
print("Minimum of column 1:", min_col1)
print("Minimum of column 2:", min_col2)


print("max:" ,max_col1)
print("max:" ,max_col2)
# Open CSV file and create a CSV reader object
with open('lonlatcal.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)

    # Create an empty 2D list
    data = []

    # Loop through each row in the CSV file
    for row in reader:

        # Append the row to the 2D list
        data.append(row)
print(data[:5])
for i in range(len(data)):
    data[i][0]=int((float(data[i][0])*100))-int((min_col1*100))
    data[i][1] = int((float(data[i][1]) * 100)) - int((min_col2*100))

print(data[:5])

rows=int((int(max_col1*100))-int((min_col1*100)))+1
cols=int(max_col2*100)-int(min_col2*100)+1
print("rows: ",rows)
print("cols: ",cols)


class Space():

    def __init__(self, height, width, num_hospitals):
        """Create a new state space with given dimensions."""
        self.height = height
        self.width = width
        self.num_hospitals = num_hospitals
        self.houses = set()
        self.hospitals = set()
        self.centroids = list()

    def add_house(self, row, col):
        """Add a house at a particular location in state space."""
        self.houses.add((row, col))

    def available_spaces(self):
        """Returns all cells not currently used by a house or hospital."""

        # Consider all possible cells
        candidates = set(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        )


        # Remove all houses and hospitals
        for house in self.houses:
            candidates.remove(house)
        for hospital in self.hospitals:
            candidates.remove(hospital)
        return candidates

    def hill_climb(self, maximum=None, image_prefix=None, log=False):
        """Performs hill-climbing to find a solution."""
        count = 0

        X = np.array(houses)
        kmeans = KMeans(n_clusters=n_hospitals, random_state=0).fit(X)
        cent = kmeans.cluster_centers_

        for i in range(len(cent)):
            self.centroids.append([int(cent[i][0]), int(cent[i][1])])
        # print(centroids)
        self.hospitals = set()
        spaces = list(self.available_spaces())
        for i in range(self.num_hospitals):
            if self.centroids[i] in spaces:
                self.hospitals.add(self.centroids[i])
            else:
                dist=abs(self.centroids[i][0] - spaces[0][0]) + abs(self.centroids[i][1] - spaces[0][1])
                for spc in spaces:
                    if(abs(self.centroids[i][0] - spc[0]) + abs(self.centroids[i][1] - spc[1])<dist):
                        dist=abs(self.centroids[i][0] - spc[0]) + abs(self.centroids[i][1] - spc[1])
                        pos=spc

                place=random.choice(list(self.available_spaces()))
                dist=abs(self.centroids[i][0] - place[0]) + abs(self.centroids[i][1] - place[1])
                for k in range(5):
                    a=random.choice(list(self.available_spaces()))
                    if(dist>abs(self.centroids[i][0] - a[0]) + abs(self.centroids[i][1] - a[1])):
                        dist=abs(self.centroids[i][0] - a[0]) + abs(self.centroids[i][1] - a[1])
                        place=a



                self.hospitals.add(place)

        # Continue until we reach maximum number of iterations
        while maximum is None or count < maximum:
            count += 1
            best_neighbors = []
            best_neighbor_cost = None

            # Consider all hospitals to move
            for hospital in self.hospitals:

                # Consider all neighbors for that hospital
                for replacement in self.get_neighbors(*hospital):

                    # Generate a neighboring set of hospitals
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    # Check if neighbor is best so far
                    cost = self.get_cost2(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            # None of the neighbors are better than the current state
            if best_neighbor_cost >= self.get_cost2(self.hospitals):
                return self.hospitals

            # Move to a highest-valued neighbor
            else:
                if log:
                    best_neighbor_cost2 = self.get_cost2(self.hospitals)
                    print(f"Found better neighbor: cost {best_neighbor_cost2}")
                self.hospitals = random.choice(best_neighbors)

            # Generate image

    def random_restart(self, maximum, image_prefix=None, log=False):
        """Repeats hill-climbing multiple times."""
        best_hospitals = None
        best_cost = None

        # Repeat hill-climbing a fixed number of times
        for i in range(maximum):
            hospitals = self.hill_climb()
            cost = self.get_cost2(hospitals)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_hospitals = hospitals
                if log:
                    print(f"{i}: Found new best state: cost {cost}")
            else:
                if log:
                    print(f"{i}: Found state: cost {cost}")


        return best_hospitals

    def get_cost(self, hospitals):
        """Calculates sum of distances from houses to nearest hospital."""
        cost = 0
        for centroid in self.centroids:
            cost += min(
                abs(centroid[0] - hospital[0]) + abs(centroid[1] - hospital[1])
                for hospital in hospitals
            )
        return cost

    def get_cost2(self, hospitals):
        """Calculates sum of distances from houses to nearest hospital."""
        cost = 0
        for house in self.houses:
            cost += min(
                abs(house[0] - hospital[0]) + abs(house[1] - hospital[1])
                for hospital in hospitals
            )
        return cost

    def get_neighbors(self, row, col):
        """Returns neighbors not already containing a house or hospital."""
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
            (row+1, col + 1),
            (row-1, col + 1),
            (row+1, col - 1),
            (row - 1, col - 1)
        ]
        neighbors = []
        for r, c in candidates:
            if (r, c) in self.houses or (r, c) in self.hospitals:
                continue
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append((r, c))
        return neighbors



n_hospitals=5
n_houses=len(data)
houses=data
s = Space(height=rows, width=cols, num_hospitals=n_hospitals)
for i in range(n_houses):
    s.add_house(houses[i][0], houses[i][1])

# Use local search to determine hospital placement
hospitals = s.hill_climb(image_prefix="hospitals", log=True)
print("Hospital positions:")

i=0
for hospital in hospitals:
    i+=1
    lat=(hospital[0]+int((min_col1*100)))/100
    lon=(hospital[1]+int((min_col2*100)))/100
    print("Hospital ",i,": ",lat," ",lon)
