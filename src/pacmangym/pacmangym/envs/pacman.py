import pygame
from pygame.locals import *
from .vector import Vector2
from .constants import *
from .entity import Entity
from .sprites import PacmanSprites

class Pacman(Entity):
    def __init__(self, node):
        Entity.__init__(self, node )
        self.name = PACMAN    
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)
        self.debug = False

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def die(self):
        self.alive = False
        self.direction = STOP

    def update(self, dt, action=None):	
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt
        if action == None:
            direction = self.getValidKey()
        else:
            direction = action
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()

    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP  

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None    
    
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
     
    # Movement Calculations Relative To Pacman
    def MovingUpOrDown(self):
        return self.direction == UP or self.direction == DOWN

    def MovingLeftOrRight(self):
        return self.direction == RIGHT or self.direction == LEFT

    def canGoUp(self):
        if self.OnNode() and self.node.neighbors[UP] != None or self.MovingUpOrDown():
            if self.debug: print("U")
            return 1 
        if self.debug: print()
        return 0 

    def canGoDown(self):
        if self.OnNode() and self.node.neighbors[DOWN] != None or self.MovingUpOrDown():
            if self.debug: print("D")
            return 1 
        if self.debug: print()
        return 0 

    def canGoRight(self):
        if self.OnNode() and self.node.neighbors[RIGHT] != None or self.MovingLeftOrRight():
            if self.debug: print("R")
            return 1 
        if self.debug: print()
        return 0 

    def canGoLeft(self):
        if self.OnNode() and self.node.neighbors[LEFT] != None or self.MovingLeftOrRight():
            if self.debug: print("L")
            return 1 
        if self.debug: print()
        return 0 
    
    def testGo(self):
        self.canGoUp()
        self.canGoDown()
        self.canGoLeft()
        self.canGoRight()

    def OnNode(self):
        x_dif = abs(self.position.x - self.node.position.x)
        y_dif = abs(self.position.y - self.node.position.y)
        if x_dif <= self.node.position.thresh and y_dif <= self.node.position.thresh:
            #print("On Node!")
            return True
        return False

