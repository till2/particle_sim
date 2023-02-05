from __future__ import annotations

import pygame as pg
import pymunk
import numpy as np
import skimage.measure as measure
import random
import os

from typing import Tuple, List, Optional, Union


# constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (3, 186, 252)
LIGHT_GREY = (70, 84, 105)
YELLOW = (245, 203, 66)
RED = (252, 3, 65)


class Person:
    """
    A dynamic particle object that represents a person in the pymunk simulation.
    """

    def __init__(
        self,
        world: pymunk.Space,
        pathfinder,
        init_min: int,
        init_max: int,
        collision_radius: int = 10,
    ) -> None:
        """
        Initialize a particle (person) object with a specific position, velocity, and target building.
        The person will follow the shortest path to its target building using the pathfinder.
        """

        # pathfinder for following the shortest path to a given goal (i.e. a target building)
        self.pf = pathfinder

        # initialization attributes of a physical particle object (with a circular shape)
        self.init_min = init_min  # lower bound for init position
        self.init_max = init_max  # upper bound for init position
        self.collision_radius = collision_radius
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.shape = pymunk.Circle(self.body, self.collision_radius)
        self.shape.density = 1
        self.shape.elasticity = 1

        # initial velocity (will be overwritten by the path to the target building)
        self.body.velocity = random.uniform(-20, 20), random.uniform(-20, 20)

        # set the initial status to susceptible (=healthy)
        self.status = "susceptible"

        # set a slight bias towards going to the library and mensa
        # mensa:            building with index 3
        # library ('IKMZ'): building with index 27
        self.weights = [1 / 35 if i not in [3, 27] else 1 / 10 for i in range(30)]

        # pick a target building
        # note: we are picking the index of the building in the pf.targets list, not the number of the building
        # (the index might differ from the buildings number)
        self.target_building = np.random.choice(range(30), p=self.weights)

        # set the initial position of the particle near the target to avoid large crowds in the center
        # (when everyone needs to get to the other side of the map at once, they form a huge crowd in the center and nobody gets through)
        x_init_mean, y_init_mean = np.random.normal(
            loc=self.pf.targets[self.target_building]
        )
        x_init = int(
            np.clip(
                np.random.normal(loc=x_init_mean, scale=50),
                a_min=init_min,
                a_max=init_max,
            )
        )
        y_init = int(
            np.clip(
                np.random.normal(loc=y_init_mean, scale=50),
                a_min=init_min,
                a_max=init_max,
            )
        )
        self.body.position = (x_init, y_init)

        # set a random time until this person picks its next target
        self.time_until_next_target = np.random.randint(9_000, 72_000)

        # add the person to the simulation
        world.add(self.body, self.shape)

    def infect(self) -> None:
        """
        Set the person's status to infected and the density of the shape to 0.9
        """
        self.shape.density = 0.9
        self.status = "infected"

    def update_velocity(self, timestep: int):
        """
        Update the velocity of the person based on the person's current position and target building.
        """
        # hyperparameters
        velocity_multiplier = 30
        vel_update_rate = 0.015  # how much of the new velocity (which follows the optimal path to the target building) gets injected

        # get the current discrete position
        x, y = self.body.position
        discrete_position = (int(x), int(y))

        # get the optimal velocity to follow the path to the target (based on the current position)
        x_velocity, y_velocity = self.pf.get_direction(
            discrete_position, target_building=self.target_building
        )

        # scale the velocity by the velocity multiplier
        x_velocity = velocity_multiplier * x_velocity
        y_velocity = velocity_multiplier * y_velocity

        # add some noise to the optimal velocity (e.g. for when it gets trapped somewhere and can't get out or to resolve running into other particles)
        additive_x_noise = 4 * random.random() - 2  # [-2 to 2], mean: 0
        additive_y_noise = 4 * random.random() - 2  # [-2 to 2], mean: 0

        # calculate the new velocity
        old_velocity = self.body.velocity
        new_x_velocity = (
            (1 - vel_update_rate) * old_velocity[0]
            + vel_update_rate * x_velocity
            + additive_x_noise
        )
        new_y_velocity = (
            (1 - vel_update_rate) * old_velocity[1]
            + vel_update_rate * y_velocity
            + additive_y_noise
        )

        # update the velocity
        self.body.velocity = (new_x_velocity, new_y_velocity)

    def draw(self, screen: pg.Surface) -> None:
        """
        Draw the person on the screen.
        The persons color depends on the person's infection status.
        """
        x, y = self.body.position
        discrete_position = (int(x), int(y))
        if self.status == "infected":
            color = YELLOW
        elif self.status == "infectious":
            color = RED
        elif self.status == "removed":
            color = LIGHT_GREY
        else:
            color = BLUE
        pg.draw.circle(screen, color, discrete_position, self.collision_radius)

    def update_target(self, timestep: int) -> None:
        """
        Updates the target building for the person.
        """
        if (timestep % self.time_until_next_target) > 0 and (
            timestep % self.time_until_next_target
        ) < 50:
            # set a new random target building
            self.target_building = np.random.choice(range(30), p=self.weights)

            # set for how many timesteps the person will persue the new target building
            self.time_until_next_target = np.random.randint(9_000, 72_000)

    def update_infection_status(
        self, avg_incubation_time: int, avg_infectious_time: int, timestep: int
    ) -> None:
        """
        Updates the infection status of the person (pseudo-randomly).
        """
        incubation_rate = (
            1 / avg_incubation_time
        )  # probability that the status switches from infected to infectious (sampled each timestep)
        removed_rate = (
            1 / avg_infectious_time
        )  # probability that the status switches from infectious to removed (sampled each timestep)

        # status updates if the person is currently infected
        if self.shape.density == 0.9:

            # assure that the status is infected
            self.status = "infected"

            # change the status to infectious with a probability of incubation_rate
            if np.random.random() <= incubation_rate:
                self.status = "infectious"
                self.shape.density = 0.8

        # status updates if the person is currently infectious
        elif self.shape.density == 0.8:

            # change the status to removed with a probability of removed_rate
            if np.random.random() <= removed_rate:
                self.status = "removed"
                self.shape.density = 0.7


class Wall:
    """
    A static wall object that is used to create borders for the buildings in the pymunk simulation.
    """

    def __init__(
        self,
        world: pymunk.Space,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        thickness: int = 3,
    ) -> None:
        """
        Initializes a wall object.
        For our simulation, we added the constraint that walls have
        to be symmetrical to either the x-axis or y-axis (no arbitrary lines).
        This allows the the pathfinding setup for people to be easier.
        Walls also cannot be just a dot (both x-values and both y-values are the same).
        """
        # ensures that wall is not a dot
        if (start_pos[0] == end_pos[0]) and (start_pos[1] == end_pos[1]):
            raise Exception(
                "Value Error: Wall cannot be a dot (make it longer along one dimension)."
            )

        # ensures that the wall is symmetrical to either the x-axis or the y-axis
        if (start_pos[0] != end_pos[0]) and (start_pos[1] != end_pos[1]):
            raise Exception(
                "Value Error: Wall's position values should match along one dimension."
            )

        # ensures that the thickness is an odd number so that it can be drawn appropriately
        if thickness % 2 == 0:  # thickness is even
            raise Exception("Value Error: Wall's thickness should be an odd number.")

        # set all the wall's attributes and add the wall object to the world
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.thickness = thickness
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)  # static body
        self.shape = pymunk.Segment(
            self.body, start_pos, end_pos, radius=thickness
        )  # people might glitch through if not big enough
        self.shape.elasticity = 1
        world.add(self.body, self.shape)

    def get_pixels(self, use_buffer_px: bool = True) -> List[Tuple[int, int]]:
        """
        Returns all pixels (coordinates) between the start and end of the wall
        as (x,y) tuples. This includes the pixels to the sides of the straight line that
        stem from the wall's thickness.
        """
        # calculate how many pixels per side of the straight line (between start and endpoint)
        # belong to the wall
        # e.g. wall is 5 pixels thick -> 2px left of line + 1px middle + 2px right of line
        extra_pixels_per_side = self.thickness // 2

        # if we use a buffer of 1px, the wall will be 1px thicker in each direction
        # to avoid particles getting stuck on walls while following their path
        if use_buffer_px:
            extra_pixels_per_side += 1

        # determine all pixels of the world that the wall occupies
        wall_pixels = []
        if self.start_pos[0] == self.end_pos[0]:
            # line is up-down (parallel to y-axis)
            smaller_y = min(self.start_pos[1], self.end_pos[1])
            larger_y = max(self.start_pos[1], self.end_pos[1])
            for y in range(
                smaller_y - extra_pixels_per_side, larger_y + extra_pixels_per_side + 1
            ):
                # add all of the wall's pixels to the wall_pixels list
                # (with the according thickness)
                for x in range(
                    self.start_pos[0] - extra_pixels_per_side,
                    self.start_pos[0] + extra_pixels_per_side + 1,
                ):
                    wall_pixels.append((x, y))
        else:
            # line is left-right (parallel to x-axis)
            smaller_x = min(self.start_pos[0], self.end_pos[0])
            larger_x = max(self.start_pos[0], self.end_pos[0])
            for x in range(
                smaller_x - extra_pixels_per_side, larger_x + extra_pixels_per_side + 1
            ):
                # add all of the wall's pixels to the wall_pixels list
                # (with the according thickness)
                for y in range(
                    self.start_pos[1] - extra_pixels_per_side,
                    self.start_pos[1] + extra_pixels_per_side + 1,
                ):
                    wall_pixels.append((x, y))
        return wall_pixels

    def draw(self, screen: pg.Surface) -> None:
        """
        Draws the wall on the screen in red with the specified thickness.
        (Note: This function is only used when the simulator is in debug mode.)
        """
        pg.draw.line(screen, RED, self.start_pos, self.end_pos, self.thickness)


class Train:
    """
    A kinematic object (meaning that the object can move, but is not affected by collisions) that represents a train
    in the pymunk simulation. The train object follows a cycle so that it halts at the station, opens it's doors
    and respawns at the top of the map periodically.
    """

    def __init__(
        self,
        world: pymunk.Space,
        start_pos: Tuple[float, float],
        wall_thickness: int = 3,
    ) -> None:
        # state attributes
        self.door_is_open = False
        self.moving = True
        self.stopped_at_station = False

        # physical object initialization attributes
        self.start_pos = start_pos
        self.wall_thickness = wall_thickness

        # create the physical body object
        self.body = pymunk.Body(mass=100, body_type=pymunk.Body.KINEMATIC)

        # add segments (walls) to the physical body
        x, y = start_pos
        self.wall1 = pymunk.Segment(
            self.body, (x, y), (x + 20, y), radius=self.wall_thickness
        )  # top-right
        self.wall2 = pymunk.Segment(
            self.body, (x, y), (x, y + 80), radius=self.wall_thickness
        )  # top-down(left)
        self.wall3 = pymunk.Segment(
            self.body, (x, y + 80), (x + 20, y + 80), radius=self.wall_thickness
        )  # bot-right
        self.door = pymunk.Segment(
            self.body, (x + 20, y), (x + 20, y + 80), radius=self.wall_thickness
        )  # top-down(right)
        self.segments = [self.wall1, self.wall2, self.wall3, self.door]

        # set the elasticity (bouncyness of other objects when they collide with the train)
        for segment in self.segments:
            segment.elasticity = 0.5

        # set the initial position and velocity of the physical body
        self.body.position = x, y
        self.body.velocity = (-1.1, 30)

        # add the train object to the simulation
        world.add(self.body)
        for segment in self.segments:
            world.add(segment)

    def update_state(self, world: pymunk.Space, timestep: int):
        """
        The movement/position of the train depends on current timestep and follows a loop.
        One cycle of the loop contains the following events (t is the first timestep of a cycle):
        t+0: train drives from the top of the map towards the bottom of the map.
        t+9k: train stops at the trainstation (velocity is set to 0) and opens it's door.
        t+13k: train closes it's door and resumes moving.
        t+36k: train respawns at the top of the map and the next cycle starts.
        """
        # t+9k: stop train at trainstation and open door
        if (
            (timestep % 36_000) > 50
            and (timestep % 9_000) <= 50
            and not self.stopped_at_station
            and self.moving
        ):
            self.moving = False
            self.stopped_at_station = True
            self.open_door(world)
            self.body.velocity = (0, 0)  # stop moving

        # t+13k: resume train and close door
        if (
            (timestep % 9_000) > 4000
            and (timestep % 9_000) <= 4050
            and self.stopped_at_station
            and not self.moving
        ):
            self.moving = True
            self.close_door(world)
            self.body.velocity = (-1.1, 30)  # resume moving

        # t+36k: respawn train at top
        if (timestep % 36_000) <= 50 and self.stopped_at_station and self.moving:
            self.stopped_at_station = (
                False  # reset stopped_at_station variable to False
            )
            self.body.position = (70, 5)  # respawn train at top

    def _get_door_coordinates(self) -> Tuple[int, int, int, int]:
        """Returns the current discrete position of the train."""
        x_a, y_a = self.door.a
        x_a, y_a = int(x_a), int(y_a)
        x_b, y_b = self.door.b
        x_b, y_b = int(x_b), int(y_b)
        return x_a, y_a, x_b, y_b

    def close_door(self, world: pymunk.Space) -> None:
        """Moves the train's right wall down to simulate closing the door."""
        x_a, x_b, y_a, y_b = self._get_door_coordinates()
        self.door.unsafe_set_endpoints(a=(x_a, y_a), b=(x_b, y_b + 40))
        self.door_is_open = False

    def open_door(self, world: pymunk.Space) -> None:
        """Moves the train's right wall up to simulate opening the door."""
        x_a, x_b, y_a, y_b = self._get_door_coordinates()
        self.door.unsafe_set_endpoints(a=(x_a, y_a), b=(x_b, y_b - 40))
        self.door_is_open = True

    def draw(self, screen) -> None:
        """Draws a train image on the screen (in the train's current position)."""
        # get the train's position
        x_float, y_float = self.body.position
        x, y = int(x_float), int(y_float)

        # load and display the train image
        train_img = pg.image.load("images/train_transparent.png")
        train_img = pg.transform.scale(train_img, (20, 80))
        screen.blit(train_img, (x + 67, y))
