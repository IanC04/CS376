class MaxHeap:
    class Node:
        def __init__(self, key, value):
            self.distance = key
            self.label = value

        def __lt__(self, other):
            return self.distance < other.distance

        def __gt__(self, other):
            return self.distance > other.distance

        def __eq__(self, other):
            return self.distance == other.distance

        def __le__(self, other):
            return self.distance <= other.distance

        def __ge__(self, other):
            return self.distance >= other.distance

        def __ne__(self, other):
            return self.distance != other.distance

        def __str__(self):
            return f"Distance: {self.distance} | Label: {self.label}"

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [MaxHeap.Node(float("inf"), -1)] * self.maxsize

    # Function to return the position of
    # parent for the node currently
    # at pos
    def parent(self, pos):

        return (pos - 1) // 2

    # Function to return the position of
    # the left child for the node currently
    # at pos
    def leftChild(self, pos):

        return 2 * pos + 1

    # Function to return the position of
    # the right child for the node currently
    # at pos
    def rightChild(self, pos):

        return 2 * pos + 2

    # Function that returns true if the passed
    # node is a leaf node
    def isLeaf(self, pos):

        if pos >= (self.size // 2) and pos <= self.size:
            return True
        return False

    # Function to swap two nodes of the heap
    def swap(self, fpos, spos):

        self.Heap[fpos], self.Heap[spos] = (self.Heap[spos],
                                            self.Heap[fpos])

    # Function to heapify the node at pos
    def maxHeapify(self, pos):

        # If the node is a non-leaf node and smaller
        # than any of its child
        if not self.isLeaf(pos):
            if (self.Heap[pos] < self.Heap[self.leftChild(pos)] or
                    self.Heap[pos] < self.Heap[self.rightChild(pos)]):

                # Swap with the left child and heapify
                # the left child
                if (self.Heap[self.leftChild(pos)] >
                        self.Heap[self.rightChild(pos)]):
                    self.swap(pos, self.leftChild(pos))
                    self.maxHeapify(self.leftChild(pos))

                # Swap with the right child and heapify
                # the right child
                else:
                    self.swap(pos, self.rightChild(pos))
                    self.maxHeapify(self.rightChild(pos))

    # Function to insert a node into the heap
    def insert(self, element):
        element = MaxHeap.Node(element[0], element[1])

        if self.size < self.maxsize:
            current = self.size
            self.Heap[current] = element
            while (self.Heap[current] >
                   self.Heap[self.parent(current)]):
                self.swap(current, self.parent(current))
                current = self.parent(current)
            self.size += 1
            return
        elif self.size == self.maxsize and element < self.Heap[0]:
            self.Heap[0] = element
            self.maxHeapify(0)

    # Function to print the contents of the heap
    def __str__(self):
        output = []
        for i in range(self.size):
            output.append(f"{self.Heap[i]} ")
        return ''.join(output)

    # Function to remove and return the maximum
    # element from the heap
    def extractMax(self):

        popped = self.Heap[0]
        self.Heap[0] = self.Heap[self.size]
        self.size -= 1
        self.maxHeapify(0)

        return popped

    def get_labels(self):
        labels = []
        for i in range(self.size):
            labels.append(self.Heap[i].label)
        return labels

    def clear(self):
        self.size = 0
        self.Heap = [MaxHeap.Node(float("inf"), -1)] * self.maxsize
