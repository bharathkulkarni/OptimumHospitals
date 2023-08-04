# run hospitals.py file

# File name: hospitals.py


import random
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import pickle



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
            if image_prefix:
                self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")

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

            if image_prefix:
                self.output_image(f"{image_prefix}{str(i).zfill(3)}.png")

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

    def output_image(self, filename):
        """Generates image with all houses and hospitals."""
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        cost_size = 40
        padding = 10

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "white"
        )
        house = Image.open("assets/images/House.png").resize(
            (cell_size, cell_size)
        )
        hospital = Image.open("assets/images/Hospital.png").resize(
            (cell_size, cell_size)
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 30)
        draw = ImageDraw.Draw(img)

        for i in range(self.height):
            for j in range(self.width):

                # Draw cell
                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                draw.rectangle(rect, fill="black")

                if (i, j) in self.houses:
                    img.paste(house, rect[0], house)
                if (i, j) in self.hospitals:
                    img.paste(hospital, rect[0], hospital)

        # Add cost
        draw.rectangle(
            (0, self.height * cell_size, self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "black"
        )
        draw.text(
            (padding, self.height * cell_size + padding),
            f"Cost: {self.get_cost2(self.hospitals)}",
            fill="white",
            font=font
        )
        import time
        import psutil
        img.save(filename)
        img.show(filename)
        time.sleep(2)


"""
rows=int(input("Enter no. of rows in the community: "))
cols=int(input("Enter no.of columns in the community: "))
n_hospitals=int(input("Enter no. of hospitals: "))
n_houses=int(input("Enter no. of houses in this community: "))

houses=[]
for i in range(n_houses):
	x=int(input("Enter the row number of house {} : ".format(i+1)))
	y=int(input("Enter the column number of house {} : ".format(i+1)))
	houses.append([x,y])
"""
import InputBox


def getDetails():
    InputBox.ShowDialog("Enter no. of rows in the community: ")
    rows = int(InputBox.GetInput())
    InputBox.ShowDialog("Enter no. of columns in the community: ")
    cols = int(InputBox.GetInput())
    InputBox.ShowDialog("Enter no. of hospitals in the community: ")
    n_hospitals = int(InputBox.GetInput())
    InputBox.ShowDialog("Enter no. of houses in the community: ")
    n_houses = int(InputBox.GetInput())
    houses = []
    for i in range(n_houses):
        InputBox.ShowDialog("Enter x,y of house {}".format(i + 1))
        coord = InputBox.GetInput()
        coord = coord.split(",")
        houses.append([int(coord[0]), int(coord[1])])
    return rows, cols, n_hospitals, n_houses, houses


"""
rows=10
cols=20
n_hospitals=2
n_houses=5
houses=[[1,2],[8,12],[4,7],[2,18],[3,19]]
"""

rows, cols, n_hospitals, n_houses, houses = getDetails()

# Create a new space and add houses randomly
s = Space(height=rows, width=cols, num_hospitals=n_hospitals)
for i in range(n_houses):
    s.add_house(houses[i][0], houses[i][1])

# Use local search to determine hospital placement
hospitals = s.hill_climb(image_prefix="hospitals", log=True)

# File name: InputBox.py
import tkinter
from tkinter import *

root = Tk()
e1 = Entry(root, width=55)
l1 = Label(root, justify=LEFT)
result = ""


def ShowDialog(s):
    root.title("Input Box")

    l1.grid(row=0, column=1, sticky=W, padx=10, pady=5)
    l1['text'] = s

    e1.grid(row=1, column=1, padx=10, pady=5)

    button1 = Button(root, text=" OK ", command=button1_Click)
    button1.grid(row=2, column=1, sticky=E, padx=10, pady=5)

    root.mainloop()


def button1_Click():
    f = open('tttemp', 'w')
    f.write(e1.get())
    f.close()
    e1.delete(0, END)
    l1['text'] = ""
    root.quit()


def GetInput():
    f = open('tttemp', 'r')
    result = f.read()
    f.close()
    import os
    os.remove('tttemp')
    return result


# File name: MessageBox.py
import tkinter
from tkinter import *
from PIL import Image, ImageTk


def Show(stri):
    root = Tk()
    root.title('Final output')
    print(stri)

    photo = PhotoImage(file=stri)
    label = Label(root, image=photo)
    label.pack()
    root.mainloop()
