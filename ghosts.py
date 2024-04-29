import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from modes import ModeController
from sprites import GhostSprites
from datastructures import PriorityQueue

class Ghost(Entity):
    def __init__(self, node, pacman=None, blinky=None):
        Entity.__init__(self, node)
        self.name = GHOST
        self.points = 200
        self.goal = Vector2()
        self.directionMethod = self.goalDirection
        self.pacman = pacman
        self.mode = ModeController(self)
        self.blinky = blinky
        self.homeNode = node

        # Path Drawing Functions
        self.linear = True
        self.draw = True

    def reset(self):
        Entity.reset(self)
        self.points = 200
        self.directionMethod = self.goalDirection

    def update(self, dt):
        self.sprites.update(dt)
        self.mode.update(dt)
        if self.mode.current is SCATTER:
            self.scatter()
        elif self.mode.current is CHASE:
            self.chase()
        Entity.update(self, dt)

    def scatter(self):
        self.goal = Vector2()

    def chase(self):
        self.goal = self.pacman.position

    def spawn(self):
        self.goal = self.spawnNode.position

    def setSpawnNode(self, node):
        self.spawnNode = node

    def startSpawn(self):
        self.mode.setSpawnMode()
        if self.mode.current == SPAWN:
            self.setSpeed(150)
            self.directionMethod = self.goalDirection
            self.spawn()

    def startFreight(self):
        self.mode.setFreightMode()
        if self.mode.current == FREIGHT:
            self.setSpeed(50)
            self.directionMethod = self.randomDirection         

    def normalMode(self):
        self.setSpeed(100)
        self.directionMethod = self.goalDirection
        self.homeNode.denyAccess(DOWN, self)

    def DrawPath(self, screen):
        pygame.draw.line(screen, self.color, (self.position.x, self.position.y), (self.goal.x, self.goal.y), width=2)
        pygame.display.flip()

    def drawPath(self, screen):
        if not self.draw: return
        if self.linear:
            pygame.draw.line(screen, self.color, (self.position.x, self.position.y), (self.goal.x, self.goal.y), width=2)
            pygame.display.flip()
        else:
            path = self.findNodePath()
            if path != None:
                start = path[0]
                pygame.draw.line(screen, self.color, (self.position.x, self.position.y), (start.position.x, start.position.y), width=2)
                pygame.display.flip()
                for node in path[1:]:
                    pygame.draw.line(screen, self.color, (start.position.x, start.position.y), (node.position.x, node.position.y), width=2) 
                    pygame.display.flip()

    def findNodePath(self):
        q = PriorityQueue()
        max_search = 30
        # (g, node, node[])
        q.Enqueue((0, self.node, []))
        while not q.Empty():
            node = q.Dequeue()
            # Destructure
            g = node[0]
            n = node[1]
            node[2].append(n)
            path = node[2]
            if len(path) > max_search:
                return None
            # print(f'g: {g}\nnode:{n}\npath:{path}')
            # print(f"Node x: {n.position.x} y: {n.position.y}")
            # print(f"Goal x:{self.goal.x} y: {self.goal.y}")
            # Found
            if n.position.x == self.goal.x and n.position.y == self.goal.y:
                print(f'Found! {len(path)}')
                # Draw Lines Between The Nodes
                return path
            g += 1
            if n.neighbors[UP] != None:
                q.Enqueue((g, n.neighbors[UP], path))
            if n.neighbors[DOWN] != None:
                q.Enqueue((g, n.neighbors[DOWN], path))
            if n.neighbors[RIGHT] != None:
                q.Enqueue((g, n.neighbors[RIGHT], path))
            if n.neighbors[LEFT] != None:
                q.Enqueue((g, n.neighbors[LEFT], path))
            



class Blinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = BLINKY
        self.color = RED
        self.sprites = GhostSprites(self)


class Pinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = PINKY
        self.color = PINK
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(TILEWIDTH*NCOLS, 0)

    def chase(self):
        self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILEWIDTH * 4


class Inky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = INKY
        self.color = TEAL
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(TILEWIDTH*NCOLS, TILEHEIGHT*NROWS)

    def chase(self):
        vec1 = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILEWIDTH * 2
        vec2 = (vec1 - self.blinky.position) * 2
        self.goal = self.blinky.position + vec2


class Clyde(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = CLYDE
        self.color = ORANGE
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector2(0, TILEHEIGHT*NROWS)

    def chase(self):
        d = self.pacman.position - self.position
        ds = d.magnitudeSquared()
        if ds <= (TILEWIDTH * 8)**2:
            self.scatter()
        else:
            self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * TILEWIDTH * 4


class GhostGroup(object):
    def __init__(self, node, pacman):
        self.blinky = Blinky(node, pacman)
        self.pinky = Pinky(node, pacman)
        self.inky = Inky(node, pacman, self.blinky)
        self.clyde = Clyde(node, pacman)
        self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]

    def __iter__(self):
        return iter(self.ghosts)

    def update(self, dt):
        for ghost in self:
            ghost.update(dt)

    def startFreight(self):
        for ghost in self:
            ghost.startFreight()
        self.resetPoints()

    def setSpawnNode(self, node):
        for ghost in self:
            ghost.setSpawnNode(node)

    def updatePoints(self):
        for ghost in self:
            ghost.points *= 2

    def resetPoints(self):
        for ghost in self:
            ghost.points = 200

    def hide(self):
        for ghost in self:
            ghost.visible = False

    def show(self):
        for ghost in self:
            ghost.visible = True

    def reset(self):
        for ghost in self:
            ghost.reset()

    def render(self, screen):
        for ghost in self:
            ghost.render(screen)
            ghost.drawPath(screen)

