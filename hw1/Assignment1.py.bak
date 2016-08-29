
import sys

			
class Tree:
	def __init__(self, rootkey):
		self.root = Node(rootkey, None, None, None)
		#create a new tree while setting root
			
	def checkTree(self, value, parentValue, root):
		#Recursive function that searches through tree to find
		#if parentValue exists
		
		if root == None:
			#if there is no root in tree
			return False
		if root.value == parentValue:
			if root.l == None or root.r == None:
				return root 
			else:
				print "Parent has two children, node not added."
				return False
		else:
			for child in root.getChildren():
				add_temp = self.checkTree(value, parentValue, child)
				if add_temp:
					return add_temp
		
	#your code goes here
	
	def findNodeDelete(self, value, root):
		if root == None:
			return False
		if value == root.value:
			if root.l == None and root.r == None:
				if root.p.l.value == value:
					root.p.l = None
				elif root.p.r.value == value:
					root.p.r = None
				root = None
				return True
			else:
				print "Node not deleted, has children"
				return False
		else:
			for child in root.getChildren():
				delete_node = self.findNodeDelete(value, child)
				if delete_node:
					return delete_node
			
		
		
	def delete(self, value):
		if self.root == None:
			self.root = Node(value, None, None, None)
		if value == self.root.value:
			if self.root.l == None and self.root.r == None:
				#print("Deleting Root")
				self.root = None
				return True
			else:
				print "Node not deleted, has children"
				return False
		else:
			for child in self.root.getChildren():
				delete_node = self.findNodeDelete(value, child)
				if delete_node:
					return delete_node
					
		print "Parent not found." 
		return False
		
	def printTree(self):
		if self.root != None:
			print self.root.value
			for child in self.root.getChildren():
				self.printBranch(child)
		else: 
			return
			
	
	def printBranch(self, root):
		if root == None:
			return
		else:
			print root.value
			for child in root.getChildren():
				self.printBranch(child)
					
''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
class Graph:
	def __init__(self):
		self.verticies = {}
			
	def addVertex(self, value):
		#check if value already exists
		if value in self.verticies:
			print "Vertex already exists"
		else:
			self.verticies[value] = []
			
	#your code goes here

'''''''''''''''''''''''''''''''''''''''''''''''''''
Tests
'''''''''''''''''''''''''''''''''''''''''''''''''''
	
#Tree Test

print "-------------------------------------------"
print "Tree Test"
print "add 10 ints to tree, print In-Order, delete 2 ints, print In-Order"
print ""

tree = Tree(5)
tree.add(6,5)
tree.add(4,5)
tree.add(7,4)
tree.add(3,7)
tree.add(8,4)
tree.add(2,8)
tree.add(9,7)
tree.add(1,3)
tree.add(10,3)

print ""

tree.printTree()

print ""

tree.delete(10)
tree.delete(1)

tree.add(18,3)

tree.printTree()

#Graph Test

print "-------------------------------------------"
print "Graph Test"
print "Add 10 vertecies, make 20 edges, print edges of five vertecies"
print ""

g = Graph()
g.addVertex(1)
g.addVertex(11)
g.addVertex(12)
g.addVertex(13)
g.addVertex(14)
g.addVertex(15)
g.addVertex(16)
g.addVertex(17)
g.addVertex(18)
g.addVertex(19)
g.addVertex(100)

g.addEdge(1,12)
g.addEdge(1,13)
g.addEdge(11,14)
g.addEdge(15,11)
g.addEdge(16,100)
g.addEdge(15,17)
g.addEdge(15,12)
g.addEdge(12,13)
g.addEdge(12,14)
g.addEdge(12,16)
g.addEdge(12,17)
g.addEdge(1,100)
g.addEdge(12,100)
g.addEdge(15,100)
g.addEdge(19,12)
g.addEdge(13,100)
g.addEdge(14,100)
g.addEdge(100,19)
g.addEdge(19,18)
g.addEdge(19,17)
g.addEdge(52, 53)

g.findVertex(1)
g.findVertex(12)
g.findVertex(13)
g.findVertex(14)
g.findVertex(100)
g.findVertex(52)
