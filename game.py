import math
import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
# from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData

class GameController(object):
    def __init__(self, rewards=None, settings=None):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        # self.pause = Pause(True)
        self.level = 0
        self.lives = 1
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()
        self.rendernodes = False
        # AI 
        self.frame_iteration = 0
        self.setSettings(settings)

        # set rewards
        if rewards != None:
            self.rewards = rewards
        else:
            self.setDefaultRewards()

    
    def setSettings(self, settings):
        if settings != None:
            self.humaninput = False
            self.disableghosts = settings["DISABLE_GHOSTS"]
            self.disablelevels = settings["DISABLE_LEVELS"]
            self.fps = settings["FPS"]
            self.t_mult = settings["T_MULT"]
        else:
            print("Default Settings")
            self.humaninput = True
            self.disableghosts = False
            self.disablelevels = False
            self.fps = 30

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    # Start game logic
    def startGame(self):      
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart))
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")

        # print(f'Pellet Count: {len(self.pellets.pelletList)}')

        # disable ghosts
        if self.disableghosts:
            self.ghosts = False
        else:
            self.setupGhosts()

        # hide annoying text
        self.textgroup.hideText()

        self.nodes.denyHomeAccess(self.pacman)


    def setupGhosts(self):
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)
        
    # where stuff happens
    def update(self, action=None):
        self.frame_iteration += 1
        dt = self.fps / 1000.0
        # dt = self.clock.tick(self.fps) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        # if not self.pause.paused:
        if self.ghosts:
            self.ghosts.update(dt)      
        if self.fruit is not None:
            self.fruit.update(dt)

        # Reward Calculations
        reward = 0
        second = self.frame_iteration % self.fps == 0

        # calculate rewards and if gameover
        pellet_reward, gameover = self.checkPelletEvents()
        if self.ghosts:
            ate_ghost, got_eaten = self.checkGhostEvents()
            reward += ate_ghost
            gameover = gameover or got_eaten
        
        reward += self.checkFruitEvents()
        reward += pellet_reward
     
        if self.pacman.direction == STOP:
            reward += self.rewards['STOPPED']

        # encourage model not to sit around -1 every second
        # if second:
        reward += -1

        # get pacman position and direction
        self.pacman.update(dt, action, self.humaninput)

        if not self.humaninput:
            if gameover or self.frame_iteration > self.t_mult*(self.score+self.fps) and self.t_mult != None:
                gameover = True
                reward += self.rewards["DIE"]

        # print(f'Reward: {reward}')

        # self.flash(dt)
        # handle human input    
        
        self.checkEvents()
        self.render()

        self.clock.tick(self.fps)

        return reward, gameover, self.score

    def getClosestPellet(self):
        closest = 1000000
        p = None
        for pellet in self.pellets.pelletList:
            diff = self.getDistance(self.pacman.position, pellet.position)
            if diff < closest:
                closest = diff
                p = pellet
        # print(f'pellet: {pellet.position} | packman: {self.pacman.position}: diff {diff}')
        p.color = GREEN
        p.render(self.screen)
        return p

    def getDistance(self, p1, p2):
        return math.sqrt((p2.y - p1.y)**2 + (p2.x - p1.x)**2)

    # Handles pausing/quiting events
    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.textgroup.hideText()
                        self.showEntities()
                    else:
                        self.textgroup.showText(PAUSETXT)
                        #self.hideEntities()

    # update - takes reward values
    def checkPelletEvents(self):
        reward = 0
        gameover = False
        # game logic
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.updateScore(pellet.points)
            reward += self.rewards["EAT_PELLET"]
            if self.pellets.numEaten == 30 and self.ghosts:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70 and self.ghosts:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            # powerpellet event
            if pellet.name == POWERPELLET:
                if self.ghosts:
                    self.ghosts.startFreight()
                reward += self.rewards["EAT_POWER_PELLET"]
            # calculate won - restart or next level
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                reward += self.rewards["BEAT_LEVEL"]
                if not self.disablelevels:
                    self.nextLevel()
                else:
                    gameover = True
                    if self.human:
                        self.restartGame()
                    # self.pause.setPause(pauseTime=3, func=self.nextLevel)
        return reward, gameover

    def checkGhostEvents(self):
        reward = 0
        gameover = False
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)                  
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    # self.pause.setPause(pauseTime=1, func=self.showEntities)
                    self.showEntities()
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                    reward += self.rewards["EAT_GHOST"]
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        reward += self.rewards["DIE"]
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        # pacman only has one life so return gameover
                        if self.lives <= 0:
                            gameover = True
                            # self.textgroup.showText(GAMEOVERTXT)
                            if self.human:
                                self.restartGame()
                            # self.pause.setPause(pauseTime=3, func=self.restartGame)
                            # self.restartGame()
                        else:
                            self.resetLevel()
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)
        return reward, gameover

    def checkFruitEvents(self):
        reward = 0
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                # print(self.fruit)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                reward += self.rewards["EAT_FRUIT"]
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
        return reward

    def showEntities(self):
        self.pacman.visible = True
        if self.ghosts:
            self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        if self.ghosts:
            self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        # self.pause.paused = True
        self.startGame()
        # disable next level
        # self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.frame_iteration = 0
        self.lives = 1
        self.level = 0
        # self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        # self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        # self.pause.paused = True
        self.pacman.reset()
        if self.ghosts:
            self.ghosts.reset()
        self.fruit = None
        # self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def flash(self, dt):
        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

    def setDefaultRewards(self):
        self.rewards = {
            "MOVE": -1, 
            "EAT_PELLET": 2,
            "EAT_POWER_PELLET": 5,
            "EAT_GHOST": 20,
            "BEAT_LEVEL": 100,
            "DIE": -200,
            "EAT_FRUIT": 10,
            "STOPPED": -50,
        }

    def render(self):
        self.screen.blit(self.background, (0, 0))
        if self.rendernodes:
            self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        if self.ghosts:
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


if __name__ == "__main__":
    game = GameController()
    game.startGame() 
    while True:
        game.update()



