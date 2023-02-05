from __future__ import annotations

import pygame as pg
import pymunk
import numpy as np
import skimage.measure as measure
import random
import os

from typing import Tuple, List, Optional, Union


class Node:
    """
    A node object that stores its coordinates and distance to the target cell.
    The nodes are used for building a search tree.
    """

    def __init__(
        self, coordinates: Tuple[int, int], distance: Optional[float] = None
    ) -> None:
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.distance = distance

    def coordinates(self) -> Tuple[int, int]:
        """Returns the coordinates of the node."""
        return (self.x, self.y)

    def get_neighbors(self, world_array: np.ndarray) -> List[Node]:
        """Returns all valid neighboring nodes of the current node."""
        neighbors = []
        for x_delta in [-1, 0, 1]:
            for y_delta in [-1, 0, 1]:
                # calculate the coordinates of a neighboring node
                neighbor_x = self.x + x_delta
                neighbor_y = self.y + y_delta

                # make sure that the neighbor is a node in the world_array
                if (
                    neighbor_x < 0
                    or neighbor_x > 799
                    or neighbor_y < 0
                    or neighbor_y > 799
                ):  # XXX
                    continue

                # disregard if the neighbor node is a wall
                if world_array[neighbor_x, neighbor_y] == 1:
                    continue

                # exclude the origin node (self) from the neighbors
                if x_delta == 0 and y_delta == 0:
                    continue

                # check that the neighbor is in the bounds of the world array
                if neighbor_x < 0 or neighbor_x > 799:
                    continue
                if neighbor_y < 0 or neighbor_y > 799:
                    continue

                # append the valid neighbor to the neighbors list
                neighbor = Node(coordinates=(neighbor_x, neighbor_y))
                neighbors.append(neighbor)

        # return a list of all neighbor nodes
        return neighbors

    def distance_to_neighbor(self, neighbor: Node) -> float:

        """Returns the euclidian distance to a neighboring cell (node).
        From node "X", the distance is defined as follows:

        sqrt(2) | 1 | sqrt(2)
        --------|---|--------
           1    | X |   1
        --------|---|--------
        sqrt(2) | 1 | sqrt(2)

        """
        # check that both nodes are neighbors
        if not self.x in [
            neighbor.x + x_delta for x_delta in [-1, 0, 1]
        ] or not self.y in [neighbor.y + y_delta for y_delta in [-1, 0, 1]]:
            raise Exception(
                f"Error: {self.coordinates()} and {neighbor.coordinates()} are not neighbors!"
            )

        # check that they are not the same node
        if self.coordinates() == neighbor.coordinates():
            raise Exception(
                f"Error: {self.coordinates()} and {neighbor.coordinates()} are the same node!"
            )

        # check if they are direct neighbors (-> distance=1)
        if (
            self.x == neighbor.x
            and self.y in [neighbor.y + y_delta for y_delta in [-1, 0, 1]]
            or self.y == neighbor.y
            and self.x in [neighbor.x + x_delta for x_delta in [-1, 0, 1]]
        ):
            return 1

        # if they are not direct neighbors, they are diagonal neighbors (-> distance=sqrt(2))
        return np.sqrt(2)


class Queue:
    """A queue that is used for tree search."""

    def __init__(self, start_node: Node) -> None:
        """Starts the tree search from the start_node."""
        self.queue = [start_node]
        self.enqueued_coordinates = {start_node.coordinates()}

    def add_node(self, node: Node) -> None:
        """Inserts a node as first element of the queue if no node with the same corresponding cell is already enqueued"""
        if node.coordinates() not in self.enqueued_coordinates:
            self.queue = [node] + self.queue
            self.enqueued_coordinates.add(node.coordinates())

    def remove_node(self) -> Node:
        """Removes a node from the set of enqueued nodes and returns the node in front, meaning the one with highest index
        (the one that 'waited' the longest)."""
        node = self.queue.pop()
        self.enqueued_coordinates.remove(node.coordinates())
        return node

    def has_elements(self) -> bool:
        """Returns True if the Queue has elements and False otherwise."""
        return bool(self.queue)


class Pathfinder:
    def __init__(self, sim, use_precomputed_heatmaps: bool) -> None:
        """
        Initializes the Pathfinder object and creates the world array, a list of target positions
        that is used for picking a target, and loads or computes a heatmap tensor that is used for the
        vector-based pathfinding.
        """
        # create the world as a 2d-array
        self.world_array = self.create_world_array(sim)

        # create a target point for each building
        self.targets = [  # index: building_name
            (640, 510),  # 0: building 1
            (630, 460),  # 1: building 2
            (710, 560),  # 2: building 3
            (710, 450),  # 3: building 4
            (740, 385),  # 4: building 5
            (650, 320),  # 5: building 6
            (640, 380),  # 6: building 7
            (560, 340),  # 7: building 8
            (580, 440),  # 8: building 9
            (570, 500),  # 9: building 10
            (540, 570),  # 10: building 11
            (770, 310),  # 11: building 12
            (730, 540),  # 12: building 13
            (350, 560),  # 13: building 14
            (380, 540),  # 14: building 14a
            (310, 610),  # 15: building 15
            (310, 570),  # 16: building 16
            (310, 530),  # 17: building 17
            (470, 730),  # 18: building 19
            (590, 700),  # 19: building 20
            (560, 630),  # 20: building 24
            (320, 150),  # 21: building 25
            (410, 130),  # 22: building 26
            (400, 300),  # 23: building 27
            (260, 320),  # 24: building 28
            (370, 360),  # 25: building 29
            (650, 630),  # 26: building BUD
            (280, 470),  # 27: building IKMZ
            (450, 600),  # 28: building 31
            (540, 700),  # 29: building 35
        ]

        # we have 30 target buildings to pick from
        n_targets = len(self.targets)

        self.heatmap_tensor = None  # will be initialized in the following lines

        if use_precomputed_heatmaps:
            # load the precomputed heatmap tensor
            self.load_heatmap_tensor()

        else:
            # compute the heatmaps for all targets
            self.heatmap_tensor = np.empty((n_targets, 800, 800))
            print(
                "computing heatmaps with shape",
                self.heatmap_tensor.shape,
                "[this may take a while!]",
            )

            for i, target in enumerate(self.targets):
                print(f"creating heatmap... [{i+1}/{n_targets}]")

                # create the target node
                target_node = Node(coordinates=target, distance=0)

                # create the heatmap for that target node
                self.heatmap_tensor[i] = self.create_heatmap(target_node)
            self.save_heatmap_tensor()

        # add the pathfinder instance to the given simulator
        sim.pf = self

    def save_heatmap_tensor(self) -> None:
        """
        Saves the heatmap tensor as a numpy file in the 'heatmaps' directory.
        """

        # create the heatmaps directory if it doesn't exist yet
        if not os.path.exists("heatmaps"):
            os.mkdir("heatmaps")

        # save the given heatmap tensor as a numpy file
        np.save("heatmaps/heatmap_tensor.npy", self.heatmap_tensor)
        print("saved heatmap_tensor with shape", self.heatmap_tensor.shape)

    def load_heatmap_tensor(self) -> None:
        """
        Loads a precomputed heatmap tensor from the 'heatmaps' directory.
        """

        if os.path.isfile("heatmaps/heatmap_tensor.npy"):
            # load a precomputed heatmap tensor
            self.heatmap_tensor = np.load("heatmaps/heatmap_tensor.npy")
            print("using precomputed heatmaps with shape", self.heatmap_tensor.shape)
        
        else:
            # the heatmap tensor doesn't exist yet
            raise Exception(
                "Heatmap-Tensor not found Error: \n \
                The heatmap tensor doesn't exist yet. Set use_precomputed_heatmaps to True and run the code again \
                    to compute it. (this only has to be done once, set use_precomputed_heatmaps=False in all preceeding runs)"
            )

    def create_world_array(self, sim) -> np.ndarray:
        """
        Creates the world array (that is used for pathfinding) from the given simulator object.
        The world array is implemented as an array of booleans (=bitmap):
        -> 0 means a person can go to this cell
        -> 1 means that this cell is part of a wall (it's not passable)
        """
        world_array = np.zeros((800, 800), dtype=int)

        # add walls (change pixels that belong to a wall to '1')
        for building in sim.buildings:
            for wall in building:
                for pixel in wall.get_pixels():
                    world_array[pixel] = 1

        # add screen borders as walls
        world_array[0, :] = 1
        world_array[799, :] = 1
        world_array[:, 0] = 1
        world_array[:, 799] = 1

        # the following lines can reduce the map size to make pathfinding practical if the map is too large otherwise
        # to do so, you can change the filter from (1,1) to e.g. (4,4) => this would result in a 200x200 map
        world_array = measure.block_reduce(
            world_array, (1, 1), np.max
        )  # by setting the filter to (1,1), we are using the original map size of 800x800
        print("world_array shape:", world_array.shape)
        return world_array

    def create_heatmap(self, target_node: Node) -> np.ndarray:
        """Creates a heatmap starting to a target node (starting from the target coordinates)."""
        # create a Queue object that is used to store nodes while searching
        queue = Queue(target_node)

        # expand the heatmap as long as there are nodes with no distance to the target
        self.visited = dict()  # dict that maps coordinates to the corresponding node

        while queue.has_elements():
            current_node = queue.remove_node()
            self.expand_heatmap(queue, current_node)

        heatmap = np.zeros((800, 800), dtype=int)
        for coordinates, node in self.visited.items():
            heatmap[coordinates] = node.distance

        return heatmap

    def expand_heatmap(self, queue: Queue, current_node: Node) -> None:
        """
        Visits the neighbors of the current node, updates their distances
        if a shorter path to them is found and adds them to the queue.
        """

        # set the current node to visited
        self.visited[current_node.coordinates()] = current_node

        # get neighbors of node
        neighbors = current_node.get_neighbors(self.world_array)
        for i in range(len(neighbors)):
            # replace neighbors that have been visited before by the according
            # nodes to preserve the distance and other attributes
            if neighbors[i].coordinates() in self.visited.keys():
                neighbors[i] = self.visited[neighbors[i].coordinates()]

        for neighbor in neighbors:

            # check if the node was not visited before
            if neighbor.coordinates() not in self.visited.keys():

                # set the neighbor's distance
                neighbor.distance = (
                    current_node.distance + current_node.distance_to_neighbor(neighbor)
                )

                # append neighbor to queue
                queue.add_node(neighbor)

            # node has been visited before
            else:
                # check if the current path's distance is shorter
                # than the neighbors old distance to the target
                if (
                    neighbor.distance
                    > current_node.distance
                    + current_node.distance_to_neighbor(neighbor)
                ):

                    # update the distance to the shorter distance
                    neighbor.distance = (
                        current_node.distance
                        + current_node.distance_to_neighbor(neighbor)
                    )

    def get_direction(
        self, current_position: Tuple[int, int], target_building: int
    ) -> Tuple[int, int]:

        """
        Returns the (x,y) direction vector that follows the shortest path to the target building.
        (This direction is later used to update the velocity of a particle so that it finds its way
        to the target.)
        """
        current_node = Node(coordinates=current_position)
        neighbors = current_node.get_neighbors(self.world_array)

        # determine the neighbor with the shortest distance to the target
        best_neighbor = None
        best_distance = np.inf
        for neighbor in neighbors:

            neighbor_distance = self.heatmap_tensor[target_building][
                neighbor.coordinates()
            ]

            if neighbor_distance < best_distance:
                best_neighbor = neighbor
                best_distance = neighbor_distance

        # if no best neighbor was found, return a random direction
        # (this shouldn't happen and is just a protection against weird edgecases)
        if best_neighbor is None:
            return (np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]))

        # calculate the direction vector to the neighbor with the shortest distance
        direction_x = best_neighbor.x - current_position[0]
        direction_y = best_neighbor.y - current_position[1]
        return direction_x, direction_y
