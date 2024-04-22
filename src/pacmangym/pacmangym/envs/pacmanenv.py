import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pygame.locals import *
from .constants import *
from .pacman import Pacman
from .nodes import NodeGroup
from .pellets import PelletGroup
from .ghosts import GhostGroup
from .fruit import Fruit
from .pauser import Pause
from .text import TextGroup
from .sprites import LifeSprites
from .sprites import MazeSprites
from .mazedata import MazeData

class PacManEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":30}

    def __init__(self, render_mode=None):
        pygame.init()
        self.screen = pygame.Surface((SCREENWIDTH,SCREENHEIGHT))
        self.screen_shown = False 
        self.background = None
        self.background_norm = None
        self.background_flash = None
        #self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.level = 0
        self.start_lives = 1 
        self.lives = self.start_lives 
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()

        self.max_pellets = 244 
        self.coord_shape = 2
        self.observation_space = spaces.Dict({
            'lives': spaces.Discrete(self.lives+1),
            'pacman_pos': spaces.Box(low=0, high=SCREENWIDTH, shape=(self.coord_shape,)),
            'pellet_positions': spaces.Box(low=0, high=SCREENWIDTH, shape=(self.max_pellets * self.coord_shape,)),
            'num_pellets': spaces.Discrete(self.max_pellets+1),
            'ghost_pos': spaces.Box(low=0, high=SCREENWIDTH, shape=(4 * self.coord_shape,)),
            'ghost_status': spaces.MultiDiscrete([4, 4, 4, 4]),
        })

        for key, value in self.observation_space.items():
            #print(f"Key: {key}, Value: {value}")
            pass

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: UP,
            1: DOWN,
            2: LEFT,
            3: RIGHT,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode 
        self.window = None

        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites("pacmangym/pacmangym/envs/"+self.mazedata.obj.name+".txt", "pacmangym/pacmangym/envs/"+self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("pacmangym/pacmangym/envs/"+self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart))
        self.pellets = PelletGroup("pacmangym/pacmangym/envs/"+self.mazedata.obj.name+".txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)
        self.pause.paused = False
    
    # Get the observation space of the current state
    def _get_obs(self):
        pacman_pos = np.array([self.pacman.position.x, self.pacman.position.y], dtype='float32')
        pellet_positions = np.zeros((self.max_pellets, self.coord_shape,), dtype='float32') 

        for i in range(self.max_pellets):
            pellet_positions[i] = np.array([np.float32(self.pellets.pelletList[i].position.x), np.float32(self.pellets.pelletList[i].position.y)])

        num_pellets = len(self.pellets.pelletList)
        ghost_pos = np.array([
            np.array([self.ghosts.pinky.position.x, self.ghosts.pinky.position.y]),
            np.array([self.ghosts.inky.position.x, self.ghosts.inky.position.y]),
            np.array([self.ghosts.blinky.position.x, self.ghosts.blinky.position.y]),
            np.array([self.ghosts.clyde.position.x, self.ghosts.clyde.position.y])
        ], dtype='float32')
        ghost_status = np.array([
            int(self.ghosts.pinky.mode.current),
            int(self.ghosts.inky.mode.current),
            int(self.ghosts.blinky.mode.current),
            int(self.ghosts.clyde.mode.current),
        ])

        observation = {
            'lives': self.lives,
            'pacman_pos': pacman_pos,
            'pellet_positions': pellet_positions,
            'num_pellets': num_pellets,
            'ghost_pos': ghost_pos,
            'ghost_status': ghost_status
        }
        #for key, value in observation.items():
        #    if key is 'pacman_pos':
        #        #print(f"Key: {key}, Value: {value}")
        #        pass


        return observation

    def _get_info(self):
        print(self.score)
        return {
            "score": self.score,
        }

    def reset(self, seed=None, options=None):
        # needs to reset environment
        self.restartGame()
        print('reset')
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        reward = 0
        old_lives = self.lives
        old_pellets = self.pellets.numEaten
        
        # Handling pacman movement
        direction = self._action_to_direction[action]
        if self.pacman.overshotTarget():
            self.pacman.node = self.pacman.target
            if self.pacman.node.neighbors[PORTAL] is not None:
                self.pacman.node = self.pacman.node.neighbors[PORTAL]
            self.pacman.target = self.pacman.getNewTarget(direction)
            if self.pacman.target is not self.pacman.node:
                self.pacman.direction = direction
            else:
                self.pacman.target = self.pacman.getNewTarget(self.pacman.direction)

            if self.pacman.target is self.pacman.node:
                self.pacman.direction = STOP
            self.pacman.setPosition()
        else: 
            if self.pacman.oppositeDirection(direction):
                self.pacman.reverseDirection()
        
        # Updating game state
        self.update()
        
        # Rewards
        if old_pellets < self.pellets.numEaten:
            reward += 4 
        #else:
        #    reward -= 2 

        if old_lives > self.lives:
            reward -= 50 
        else:
            reward += 1

        # Termination states
        if self.lives <= 0 or self.pellets.isEmpty():
            terminated = True 
        else:
            terminated = False

        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info


    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE)
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE)
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):      
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites("pacmangym/pacmangym/envs/"+self.mazedata.obj.name+".txt", "pacmangym/pacmangym/envs/"+self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("pacmangym/pacmangym/envs/"+self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart))
        self.pellets = PelletGroup("pacmangym/pacmangym/envs/"+self.mazedata.obj.name+".txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)
        self.pause.paused = False

    def update(self):
        #self.clock.tick(self.metadata["render_fps"]) / 1000.0
        if self.render_mode == 'human':
            dt = self.metadata["render_fps"] / 1000.0 
        else:
            dt = 0

        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)      
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt)
        else:
            self.pacman.update(dt)

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            if self.screen_shown == False:
                self.clock = pygame.time.Clock()
                self.screen_shown = True
                self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
            self.screen.blit(self.background, (0, 0))
            #self.nodes.render(self.screen)
            self.pellets.render(self.screen)
            if self.fruit is not None:
                self.fruit.render(self.screen)
            self.pacman.render(self.screen)
            self.ghosts.render(self.screen)
            self.textgroup.render(self.screen)

            for i in range(len(self.lifesprites.images)):
                x = self.lifesprites.images[i].get_width() * i
                y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
                self.screen.blit(self.lifesprites.images[i], (x, y))

            for i in range(len(self.fruitCaptured)):
                x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
                y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
                self.screen.blit(self.fruitCaptured[i], (x, y))

            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()

    # TODO: Pellet rewards here
    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            #self.pellets.pelletList.remove(pellet)
            pellet.position.x = -1
            pellet.position.y = -1
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                # TODO: End game here. Do not go to next level
                #self.pause.setPause(pauseTime=3, func=self.nextLevel)

    # TODO: Ghost rewards here
    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)                  
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=0, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            # TODO: done signal from here
                            #self.pause.setPause(pauseTime=3, func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=0, func=self.resetLevel)
    
    # TODO: Fruit rewards here
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.lives = self.start_lives 
        self.level = 0
        self.pause.paused = False 
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        #self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        #self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

if __name__ == "__main__":
    game = PacManEnv()
    game.startGame()
    while True:
        game.update()



