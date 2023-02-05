from __future__ import annotations

import pygame as pg
import pymunk
import numpy as np
import skimage.measure as measure
import random
import os

from objects import Person, Wall, Train

from typing import Tuple, List, Optional, Union


# constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (3, 186, 252)
LIGHT_GREY = (70, 84, 105)
YELLOW = (245, 203, 66)
RED = (252, 3, 65)


class CovidSim:
    """
    The CovidSim class is a simulator build on top of the pymunk particle simulation engine. It simulates people
    as particles that can infect other particles with a certain probability and go through stages of an SIR model:
    susceptible -> infected (->infectious) -> recovered/removed.
    The class has methods to set up the simulation and update the status of objects within the simulation.
    """

    def __init__(
        self,
        n_people: int,
        infection_prob: float = 0.3,
        avg_incubation_time: int = 5_000,
        avg_infectious_time: int = 10_000,
        debug_mode: bool = False,
        FPS: int = 60,
    ) -> None:
        """
        Initialize the simulation with the given parameters. This includes setting the number of people,
        the infection probability, the average incubation and infectious times, and the FPS of the visual output.
        """

        # simulator setup
        self.n_people = n_people
        self.collision_points = [] # will be filled with points where collisions occured for plotting a heatmap later
        self.status_counts = (
            []
        )  # will be filled with 4-tuples of counts for every status (susceptible/infected/infectious/removed)
        self.pf = None  # will be set later because the pathfinder needs attributes from the sim for initialization
        


        # screen setup
        self.screen_size = 800
        self.width, self.height = (self.screen_size, self.screen_size)
        self.FPS = FPS

        # useful simulator attributes
        self.draw_dots = debug_mode
        self.draw_walls = debug_mode
        self.speedup_factor = 1
        self.running = True

        # hyperparameters
        self.infection_prob = infection_prob
        self.avg_incubation_time = avg_incubation_time
        self.avg_infectious_time = avg_infectious_time

        # setup screen_borders, buildings
        self.world = None
        self.screen_borders = None
        self.buildings = None
        self.create_world()

    def create_world(self) -> None:
        """
        Create the simulated world, including the screen borders and buildings.
        """

        # create the simulated world
        self.world = pymunk.Space()

        # add screen borders as walls
        self.screen_borders = [
            Wall(
                world=self.world, start_pos=(0, 0), end_pos=(800, 0), thickness=1
            ),  # top-right
            Wall(
                world=self.world, start_pos=(0, 0), end_pos=(0, 800), thickness=1
            ),  # top-down (left)
            Wall(
                world=self.world, start_pos=(800, 0), end_pos=(800, 800), thickness=1
            ),  # top-down(right)
            # make two walls at the bottom to leave a hole for the train to pass through
            Wall(
                world=self.world, start_pos=(0, 800), end_pos=(110, 800), thickness=1
            ),  # bot-right
            Wall(
                world=self.world, start_pos=(130, 800), end_pos=(800, 800), thickness=1
            ),
        ]  # bot-right

        # create the buildings
        self.buildings = [
            self._create_tile(origin_pos=(630, 490), tile_type="building_1"),
            self._create_tile(origin_pos=(620, 440), tile_type="building_2"),
            self._create_tile(origin_pos=(700, 530), tile_type="building_3"),
            self._create_tile(origin_pos=(690, 430), tile_type="building_4"),
            self._create_tile(origin_pos=(700, 310), tile_type="building_5"),
            self._create_tile(origin_pos=(630, 310), tile_type="building_6"),
            self._create_tile(origin_pos=(630, 340), tile_type="building_7"),
            self._create_tile(origin_pos=(490, 340), tile_type="building_8"),
            self._create_tile(origin_pos=(570, 390), tile_type="building_9"),
            self._create_tile(origin_pos=(490, 480), tile_type="building_10"),
            self._create_tile(origin_pos=(490, 550), tile_type="building_11"),
            self._create_tile(origin_pos=(760, 280), tile_type="building_12"),
            self._create_tile(origin_pos=(720, 530), tile_type="building_13"),
            self._create_tile(origin_pos=(340, 520), tile_type="building_14"),
            self._create_tile(origin_pos=(370, 520), tile_type="building_14a"),
            self._create_tile(origin_pos=(280, 600), tile_type="building_15"),
            self._create_tile(origin_pos=(280, 560), tile_type="building_16"),
            self._create_tile(origin_pos=(280, 520), tile_type="building_17"),
            self._create_tile(origin_pos=(450, 700), tile_type="building_19"),
            self._create_tile(origin_pos=(570, 680), tile_type="building_20"),
            self._create_tile(origin_pos=(630, 600), tile_type="BUD"),
            self._create_tile(origin_pos=(230, 430), tile_type="IKMZ"),
            self._create_tile(origin_pos=(490, 620), tile_type="building_24"),
            self._create_tile(origin_pos=(260, 100), tile_type="building_25"),
            self._create_tile(origin_pos=(390, 100), tile_type="building_26"),
            self._create_tile(origin_pos=(350, 290), tile_type="building_27"),
            self._create_tile(origin_pos=(250, 280), tile_type="building_28"),
            self._create_tile(origin_pos=(350, 340), tile_type="building_29"),
            self._create_tile(origin_pos=(440, 550), tile_type="building_31"),
            self._create_tile(origin_pos=(520, 680), tile_type="building_35"),
            # building 36 is not a target building (because it doesn't have a number on the map)
            self._create_tile(origin_pos=(440, 410), tile_type="building_36"),
        ]

    def collision_begin(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: Tuple
    ) -> bool:
        """
        This custom pre-collision-handling method handles infection spreading
        when two persons collide.
        It also has to return True so the default PyMunk collision handler can handle
        changes of physical attributes afterwards (e.g. updating the velocity).
        """
        involves_infected_person = False

        for shape in arbiter.shapes:
            # check if the collision involves an object that is not a person (e.g. a wall)
            if shape.__class__ == pymunk.shapes.Segment:
                return True

            # check if a person is infectious
            if shape.density == 0.8:  # (0.8 is the key for infectious)
                involves_infected_person = True

        if involves_infected_person:
            for shape in arbiter.shapes:
                if shape.density == 1.0:  # (1.0 is the key for susceptible)
                    # "infection_prob" is the probability that infection status is shared when colliding
                    if random.random() < self.infection_prob:
                        # set the density to 0.9 to signal that the person is now infected
                        # -> the infection status will be set in the next timestep automatically
                        shape.density = 0.9

                        # save where the infectios collision occured
                        self.collision_points.append(shape.body.position)
        return True

    def get_status_counts(self) -> Tuple[int, int, int, int]:
        """
        Returns a 4-tuple with counts of how many people there are in each of the 4 stages
        (susceptible, infected, infectious and removed) of our model.
        """
        status_list = [person.status for person in self.people]
        susceptible_count = 0
        infected_count = 0
        infectious_count = 0
        removed_count = 0

        for status in status_list:
            if status == "susceptible":
                susceptible_count += 1
            elif status == "infected":
                infected_count += 1
            elif status == "infectious":
                infectious_count += 1
            else:
                removed_count += 1
        return susceptible_count, infected_count, infectious_count, removed_count

    def run(
        self,
        seed: int = 42,
        speedup_factor: int = 1,
        max_timestep: int = 3000,
        return_data: bool = False,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[pymunk.vec2d.Vec2d]] or None:
        """
        Runs the simulation until it is stopped. This method sets up the background and visual (pygame) of the simulation and also
        adds people and a train to the world. It defines a custom collision handler for handling infection spread
        and infects a few (3) people to start the epidemic. The method also handles mouse and keyboard events,
        updates the velocity of people, and renders the map and all simulated objects.
        It saves the status counts for all people and stops the simulation if the maximum given simulation time is reached.
        The method also has the option to return the collected data for plotting.
        """
        # setup the new run
        random.seed(seed)
        self.running = True
        self.speedup_factor = speedup_factor
        self.status_counts = []  # reset all status counts

        # create the pygame-screen
        self.screen = pg.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pg.time.Clock()

        # add the logo and caption for the window
        logo = pg.image.load("images/virus_logo.png")
        pg.display.set_icon(logo)
        pg.display.set_caption("COVID19-Sim")

        # create the particle simulation
        self.create_world()

        # add people to the simulation
        self.people = [
            Person(
                world=self.world,
                pathfinder=self.pf,
                init_min=0,
                init_max=self.screen_size,
                collision_radius=2,
            )
            for i in range(self.n_people)
        ]

        # define custom collision handler that handles infection spreading
        self.handler = self.world.add_default_collision_handler()
        self.handler.begin = (
            self.collision_begin
        )  # each time two objects collide the custom collision_begin method is called for handling infection spread

        # infect 3 random persons to start the epidemic
        for _ in range(3):
            random.choice(self.people).infect()

        # add a train to the simulation
        self.train = Train(world=self.world, start_pos=(70, 5), wall_thickness=3)

        timestep = 0
        while self.running:

            self.clock.tick(self.FPS)  # update pygame time
            self.world.step(
                self.speedup_factor / self.FPS
            )  # keeps rendered steps/s consistent (independent of self.FPS)
            timestep += 1

            # handle mouse and keyboard events (e.g. closing the window)
            self.events()

            # update the velocity of all people according to their goal-path
            for person in self.people:
                person.update_velocity(pg.time.get_ticks())

            # update the trains state and the infection-status updates for all people
            self.update()

            # render the map and all simulated objects
            if self.running:
                self.draw()

            # save status counts for all people
            (
                susceptible_count,
                infected_count,
                infectious_count,
                removed_count,
            ) = self.get_status_counts()
            self.status_counts.append(
                (susceptible_count, infected_count, infectious_count, removed_count)
            )

            # stop the simulation if the maximum given simulation time is reached
            if timestep >= max_timestep:
                break

        pg.quit()

        # return the collected data
        if return_data:
            susceptible_counts = [
                status_tuple[0] for status_tuple in self.status_counts
            ]
            infected_counts = [status_tuple[1] for status_tuple in self.status_counts]
            infectious_counts = [status_tuple[2] for status_tuple in self.status_counts]
            removed_counts = [status_tuple[3] for status_tuple in self.status_counts]

            return (
                susceptible_counts,
                infected_counts,
                infectious_counts,
                removed_counts,

                self.collision_points
            )

    def events(self) -> None:
        """
        Handles pygame events to stop the simulation via clicking the exit button or pressing the ESC-key.
        """
        for event in pg.event.get():

            # check if the exit button (top right "X") was pressed
            if event.type == pg.QUIT:
                pg.quit()
                self.running = False

            # check if a keyboard key was pressed
            if event.type == pg.KEYDOWN:
                # ESC key
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    self.running = False

    def update(self) -> None:
        """
        Updates the state (i.e. velocity updates, color updates, ...) of the train and all people (including the infection status).
        """

        # update the train's state
        self.train.update_state(world=self.world, timestep=pg.time.get_ticks())

        # update the targets for all people (which building they want to visit)
        for person in self.people:
            person.update_target(timestep=pg.time.get_ticks())

        # update the infection status for all people
        for person in self.people:
            person.update_infection_status(
                self.avg_incubation_time,
                self.avg_infectious_time,
                timestep=pg.time.get_ticks(),
            )

    def draw(self) -> None:
        """
        Renders the simulated world on the pygame screen.
        If the simulator is in debug-mode, it also draws dots (helpful for aligning walls etc.)
        and the walls of buildings.
        """
        # draw the background image of Golm
        golm_img = pg.image.load("images/golm_map.png")
        golm_img = pg.transform.scale(golm_img, (self.screen_size, self.screen_size))
        self.screen.blit(golm_img, (0, 0))

        # draw the train
        self.train.draw(self.screen)

        # draw all people
        for person in self.people:
            person.draw(self.screen)

        # draw the buildings
        if self.draw_walls:
            for building in self.buildings:
                for wall in building:
                    wall.draw(self.screen)

        # draw a grid of dots for testing & debugging
        if self.draw_dots:
            for i in range(80):
                for j in range(80):
                    if i % 10 == 0 and j % 10 == 0:
                        pg.draw.circle(self.screen, RED, (i * 10, j * 10), 2)
                    else:
                        pg.draw.circle(self.screen, LIGHT_GREY, (i * 10, j * 10), 2)

        # update the entire screen
        pg.display.flip()

    def _create_tile(self, origin_pos: Tuple[int, int], tile_type: str) -> List[Wall]:
        """
        Takes the origin (top-left) position and type of a building as input and
        returns a list of it's walls as static objects.
        """
        x, y = origin_pos

        if tile_type == "house":
            tile_walls = [
                # create main walls
                Wall(world=self.world, start_pos=(x, y), end_pos=(x + 80, y)),
                Wall(world=self.world, start_pos=(x, y), end_pos=(x, y + 80)),
                Wall(world=self.world, start_pos=(x + 80, y), end_pos=(x + 80, y + 80)),
                # create half-open wall
                Wall(world=self.world, start_pos=(x, y + 80), end_pos=(x + 20, y + 80)),
                Wall(
                    world=self.world,
                    start_pos=(x + 60, y + 80),
                    end_pos=(x + 80, y + 80),
                ),
            ]
        if tile_type == "building_1":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 20, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 80)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 80), end_pos=(x + 20, y + 80)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 20, y), end_pos=(x + 20, y + 30)
                ),  # top-down(right)
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 60),
                    end_pos=(x + 20, y + 80),
                ),  # top-down(right)
            ]
        if tile_type == "building_2":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 20, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 30), end_pos=(x + 30, y + 30)
                ),  # bot-right
            ]
        if tile_type == "building_3":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 20, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x + 20, y), end_pos=(x + 20, y + 80)
                ),  # top-down(right)
                Wall(
                    world=self.world, start_pos=(x, y + 80), end_pos=(x + 20, y + 80)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 60), end_pos=(x, y + 80)
                ),  # top-down(left)
            ]
        if tile_type == "building_4":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x + 40, y), end_pos=(x + 40, y + 50)
                ),  # top-down(right)
                Wall(
                    world=self.world, start_pos=(x, y + 50), end_pos=(x + 40, y + 50)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 20)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 40), end_pos=(x, y + 50)
                ),  # top-down(left)
            ]
        if tile_type == "building_5":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 20, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x + 20, y), end_pos=(x + 20, y + 60)
                ),  # top-down(right)
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 60), end_pos=(x, y + 90)
                ),  # top-down(left)
                # create right complex
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 60),
                    end_pos=(x + 60, y + 60),
                ),  # bot-right
                Wall(
                    world=self.world, start_pos=(x, y + 90), end_pos=(x + 60, y + 90)
                ),  # bot-right
                Wall(
                    world=self.world,
                    start_pos=(x + 60, y + 60),
                    end_pos=(x + 60, y + 90),
                ),  # top-down(right)
            ]
        if tile_type == "building_6":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 50, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 20)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x + 50, y), end_pos=(x + 50, y + 20)
                ),  # top-down(right)
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x, y + 20), end_pos=(x + 20, y + 20)
                ),  # bot-right
                Wall(
                    world=self.world,
                    start_pos=(x + 40, y + 20),
                    end_pos=(x + 50, y + 20),
                ),  # bot-right
            ]
        if tile_type == "building_7":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x + 20, y + 10)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x, y + 70)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 70), end_pos=(x + 20, y + 70)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 10),
                    end_pos=(x + 20, y + 30),
                ),  # top-down(right)
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 60),
                    end_pos=(x + 20, y + 70),
                ),  # top-down(right)
            ]
        if tile_type == "building_8":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 60, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 30), end_pos=(x + 20, y + 30)
                ),  # bot-left (left of door)
                Wall(
                    world=self.world, start_pos=(x + 60, y), end_pos=(x + 60, y - 50)
                ),  # top-up
                Wall(
                    world=self.world,
                    start_pos=(x + 60, y - 50),
                    end_pos=(x + 90, y - 50),
                ),  # top-up-right
                Wall(
                    world=self.world,
                    start_pos=(x + 90, y - 50),
                    end_pos=(x + 90, y + 30),
                ),  # top-up-right-down
                Wall(
                    world=self.world,
                    start_pos=(x + 70, y + 30),
                    end_pos=(x + 90, y + 30),
                ),  # right of door (bot-right)
            ]
        if tile_type == "building_9":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y + 20), end_pos=(x + 20, y + 20)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y + 20), end_pos=(x, y + 70)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 70), end_pos=(x + 20, y + 70)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 20),
                    end_pos=(x + 20, y + 30),
                ),  # top-down(right)
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 60),
                    end_pos=(x + 20, y + 70),
                ),  # top-down(right)
            ]
        if tile_type == "building_10":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 80, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 40)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 40), end_pos=(x + 30, y + 40)
                ),  # bot-left (left of door)
                Wall(
                    world=self.world, start_pos=(x + 80, y), end_pos=(x + 80, y - 20)
                ),  # top-up
                Wall(
                    world=self.world,
                    start_pos=(x + 80, y - 20),
                    end_pos=(x + 100, y - 20),
                ),  # top-up-right
                Wall(
                    world=self.world,
                    start_pos=(x + 100, y - 20),
                    end_pos=(x + 100, y + 40),
                ),  # top-up-right-down
                Wall(
                    world=self.world,
                    start_pos=(x + 70, y + 40),
                    end_pos=(x + 100, y + 40),
                ),  # right of door (bot-right)
            ]
        if tile_type == "building_11":
            tile_walls = [
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 30, y), end_pos=(x + 100, y)
                ),  # top-right
                # main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 40)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x + 100, y), end_pos=(x + 100, y + 40)
                ),  # top-down(right)
                Wall(
                    world=self.world, start_pos=(x, y + 40), end_pos=(x + 100, y + 40)
                ),  # bot-right
            ]
        if tile_type == "building_12":
            tile_walls = [
                Wall(
                    world=self.world, start_pos=(x, y + 40), end_pos=(x, y + 70)
                ),  # top-down(left)
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x + 20, y + 10)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y + 70), end_pos=(x + 20, y + 70)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 10),
                    end_pos=(x + 20, y + 70),
                ),  # top-down(right)
            ]
        if tile_type == "building_13":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x + 20, y), end_pos=(x + 30, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 30), end_pos=(x + 30, y + 30)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 30, y), end_pos=(x + 30, y + 30)
                ),  # top-down(right)
            ]
        if tile_type == "building_14":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 20, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x + 20, y), end_pos=(x + 20, y + 90)
                ),  # top-down(right)
                Wall(
                    world=self.world, start_pos=(x, y + 90), end_pos=(x + 20, y + 90)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 20)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 50), end_pos=(x, y + 90)
                ),  # top-down(left)
            ]
        if tile_type == "building_14a":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x + 20, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x - 10, y), end_pos=(x - 10, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world,
                    start_pos=(x - 10, y + 30),
                    end_pos=(x + 40, y + 30),
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 40, y), end_pos=(x + 40, y + 30)
                ),  # top-down(right)
            ]
        if tile_type == "building_15":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 20)
                ),  # top-down(left)
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x, y + 20), end_pos=(x + 40, y + 20)
                ),  # bot-right
            ]
        if tile_type == "building_16":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 20)
                ),  # top-down(left)
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x, y + 20), end_pos=(x + 40, y + 20)
                ),  # bot-right
            ]
        if tile_type == "building_17":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 20)
                ),  # top-down(left)
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x, y + 20), end_pos=(x + 40, y + 20)
                ),  # bot-right
            ]

        # building 18 is not on the map.

        if tile_type == "building_19":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x + 30, y), end_pos=(x + 50, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 50)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 50), end_pos=(x + 50, y + 50)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 50, y), end_pos=(x + 50, y + 50)
                ),  # top-down(right)
            ]
        if tile_type == "building_20":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x + 20, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x - 10, y), end_pos=(x - 10, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world,
                    start_pos=(x - 10, y + 30),
                    end_pos=(x + 40, y + 30),
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 40, y), end_pos=(x + 40, y + 30)
                ),  # top-down(right)
            ]
        if tile_type == "BUD":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x + 30, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 50)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 50), end_pos=(x + 40, y + 50)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 40, y), end_pos=(x + 40, y + 50)
                ),  # top-down(right)
            ]
        if tile_type == "IKMZ":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x + 30, y), end_pos=(x + 70, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 60)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 60), end_pos=(x + 70, y + 60)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 70, y), end_pos=(x + 70, y + 60)
                ),  # top-down(right)
            ]
        if tile_type == "building_24":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 80, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 30), end_pos=(x + 30, y + 30)
                ),  # bot-left (left of door)
                Wall(
                    world=self.world, start_pos=(x + 80, y), end_pos=(x + 80, y - 30)
                ),  # top-up
                Wall(
                    world=self.world,
                    start_pos=(x + 80, y - 30),
                    end_pos=(x + 100, y - 30),
                ),  # top-up-right
                Wall(
                    world=self.world,
                    start_pos=(x + 100, y - 30),
                    end_pos=(x + 100, y + 30),
                ),  # top-up-right-down
                Wall(
                    world=self.world,
                    start_pos=(x + 70, y + 30),
                    end_pos=(x + 100, y + 30),
                ),  # right of door (bot-right)
            ]
        if tile_type == "building_35":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x + 30, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 30), end_pos=(x + 40, y + 30)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 40, y), end_pos=(x + 40, y + 30)
                ),  # top-down(right)
            ]
        if tile_type == "building_25":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 90, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 150)
                ),  # top-down(left)
                Wall(
                    world=self.world,
                    start_pos=(x + 40, y + 150),
                    end_pos=(x + 90, y + 150),
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 90, y), end_pos=(x + 90, y + 150)
                ),  # top-down(right)
            ]
        if tile_type == "building_26":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x + 40, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y), end_pos=(x, y + 150)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 150), end_pos=(x + 10, y + 150)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 40, y), end_pos=(x + 40, y + 150)
                ),  # top-down(right)
            ]
        if tile_type == "building_27":
            tile_walls = [
                # create half-open wall
                Wall(
                    world=self.world, start_pos=(x + 30, y), end_pos=(x + 100, y)
                ),  # top-right
                # main walls
                Wall(
                    world=self.world, start_pos=(x - 10, y), end_pos=(x - 10, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x + 100, y), end_pos=(x + 100, y + 30)
                ),  # top-down(right)
                Wall(
                    world=self.world,
                    start_pos=(x - 10, y + 30),
                    end_pos=(x + 100, y + 30),
                ),  # bot-right
            ]
        if tile_type == "building_28":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x + 60, y + 10)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x, y + 90)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 90), end_pos=(x + 60, y + 90)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world,
                    start_pos=(x + 60, y + 10),
                    end_pos=(x + 60, y + 30),
                ),  # top-down(right)
                Wall(
                    world=self.world,
                    start_pos=(x + 60, y + 60),
                    end_pos=(x + 60, y + 90),
                ),  # top-down(right)
                # wall in the middle
                Wall(
                    world=self.world,
                    start_pos=(x + 30, y + 30),
                    end_pos=(x + 30, y + 60),
                ),  # top-down(right)
            ]
        if tile_type == "building_29":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x - 10, y), end_pos=(x + 80, y)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x - 10, y), end_pos=(x - 10, y + 30)
                ),  # top-down(left)
                Wall(
                    world=self.world,
                    start_pos=(x - 10, y + 30),
                    end_pos=(x + 30, y + 30),
                ),  # bot-left (left of door)
                Wall(
                    world=self.world, start_pos=(x + 80, y), end_pos=(x + 80, y - 20)
                ),  # top-up
                Wall(
                    world=self.world,
                    start_pos=(x + 80, y - 20),
                    end_pos=(x + 100, y - 20),
                ),  # top-up-right
                Wall(
                    world=self.world,
                    start_pos=(x + 100, y - 20),
                    end_pos=(x + 100, y + 30),
                ),  # top-up-right-down
                Wall(
                    world=self.world,
                    start_pos=(x + 70, y + 30),
                    end_pos=(x + 100, y + 30),
                ),  # right of door (bot-right)
            ]
        if tile_type == "building_31":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x + 20, y + 10)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x, y + 70)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 70), end_pos=(x + 20, y + 70)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 10),
                    end_pos=(x + 20, y + 30),
                ),  # top-down(right)
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 60),
                    end_pos=(x + 20, y + 70),
                ),  # top-down(right)
            ]
        if tile_type == "building_36":
            tile_walls = [
                # create main walls
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x + 20, y + 10)
                ),  # top-right
                Wall(
                    world=self.world, start_pos=(x, y + 10), end_pos=(x, y + 60)
                ),  # top-down(left)
                Wall(
                    world=self.world, start_pos=(x, y + 60), end_pos=(x + 20, y + 60)
                ),  # bot-right
                # create half-open wall
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 10),
                    end_pos=(x + 20, y + 30),
                ),  # top-down(right)
                Wall(
                    world=self.world,
                    start_pos=(x + 20, y + 50),
                    end_pos=(x + 20, y + 60),
                ),  # top-down(right)
            ]
        return tile_walls
