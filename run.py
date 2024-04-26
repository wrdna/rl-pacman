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
# AI imports
from enum import Enum

class GameController(object):
    def __init__(self):
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
        # AI 
        self.frame_iteration = 0

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

        self.setupGhosts()
        # self.ghosts = False

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
        
    # Main logic in loop
    def update(self, action=None):
        self.frame_iteration += 1
        dt = self.clock.tick(30) / 1000.0
        # dt = self.clock.tick(1)
        # dt = 1/1000
        self.textgroup.update(dt)
        self.pellets.update(dt)
        # if not self.pause.paused:
        if self.ghosts:
            self.ghosts.update(dt)      
        if self.fruit is not None:
            self.fruit.update(dt)

        # 
        reward = 0
        reward += self.checkPelletEvents()
        if self.ghosts:
            reward += self.checkGhostEvents()
        reward += self.checkFruitEvents()

        # get pacman position and direction
        p_pos, p_dir = self.pacman.update(dt)

        # get positions of ghosts

        # Logging
        # print(reward)

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        self.checkEvents()
        self.render()

        return p_dir

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

    # Update - takes reward values
    def checkPelletEvents(self, MOVE=-1, EAT_PELLET=2, EAT_POWER_PELLET=5, BEAT_LEVEL=100):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        reward = MOVE
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            reward += EAT_PELLET
            if self.pellets.numEaten == 30 and self.ghosts:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70 and self.ghosts:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                if self.ghosts:
                    self.ghosts.startFreight()
                reward += EAT_POWER_PELLET
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.nextLevel()
                reward += BEAT_LEVEL
                # self.pause.setPause(pauseTime=3, func=self.nextLevel)
        return reward

    def checkGhostEvents(self, EAT_GHOST=20, DIE=-200):
        reward = 0
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
                    reward += EAT_GHOST
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        reward += DIE
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.restartGame()
                            # self.pause.setPause(pauseTime=3, func=self.restartGame)
                            self.restartGame()
                        else:
                            self.resetLevel()
                            # self.pause.setPause(pauseTime=3, func=self.resetLevel)
        return reward

    def checkFruitEvents(self, EAT_FRUIT=50):
        reward = 0
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                # print(self.fruit)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                reward += EAT_FRUIT
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
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.frame_iteration = 0
        self.lives = 5
        self.level = 0
        # self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        # self.pause.paused = True
        self.pacman.reset()
        if self.ghosts:
            self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
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



