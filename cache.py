class LinkedSet(object):
    """A doubly-linked set, i.e., a set whose entries are ordered and stored
    as a doubly-linked list.

    This data structure is designed to efficiently implement a number of cache
    replacement policies such as LRU and derivatives such as Segmented LRU.

    It provides O(1) time complexity for the following operations: searching,
    remove from any position, move to top, move to bottom, insert after or
    before a given item.
    """
    class _Node(object):
        """Class implementing a node of the linked list"""

        def __init__(self, val, up=None, down=None):
            """Constructor

            Parameters
            ----------
            val : any hashable type
                The value stored by the node
            up : any hashable type, optional
                The node above in the list
            down : any hashable type, optional
                The node below in the list
            """
            self.val = val
            self.up = up
            self.down = down

    def __init__(self, iterable=[]):
        """Constructor

        Parameters
        ----------
        itaerable : iterable type
            An iterable type to inizialize the data structure.
            It must contain only one instance of each element
        """
        self._top = None
        self._bottom = None
        self._map = {}
        if iterable:
            if len(set(iterable)) < len(iterable):
                raise ValueError('The iterable parameter contains repeated '
                                 'elements')
            for i in iterable:
                self.append_bottom(i)

    def __len__(self):
        """Return the number of elements in the linked set

        Returns
        -------
        len : int
            The length of the set
        """
        return len(self._map)

    def __iter__(self):
        """Return an iterator over the set

        Returns
        -------
        reversed : iterator
            An iterator over the set
        """
        cur = self._top
        while cur:
            yield cur.val
            cur = cur.down

    def __reversed__(self):
        """Return a reverse iterator over the set

        Returns
        -------
        reversed : iterator
            A reverse iterator over the set
        """
        cur = self._bottom
        while cur:
            yield cur.val
            cur = cur.up

    def __str__(self):
        """Return a string representation of the set

        Returns
        -------
        str : str
            A string representation of the set
        """
        return self.__class__.__name__ + "([" + "".join("%s, " % str(i) for i in self)[:-2] + "])"

    def __contains__(self, k):
        """Return whether the set contains a given item

        Parameters
        ----------
        k : any hashable type
            The item to search

        Returns
        -------
        contains : bool
            *True* if the set contains the item, *False* otherwise
        """
        return k in self._map

    @property
    def top(self):
        """Return the item at the top of the set

        Returns
        -------
        top : any hashable type
            The item at the top or *None* if the set is empty
        """
        return self._top.val if self._top is not None else None

    @property
    def bottom(self):
        """Return the item at the bottom of the set

        Returns
        -------
        bottom : any hashable type
            The item at the bottom or *None* if the set is empty
        """
        return self._bottom.val if self._bottom is not None else None

    def pop_top(self):
        """Pop the item at the top of the set

        Returns
        -------
        top : any hashable type
            The item at the top or *None* if the set is empty
        """
        if self._top == None:  # No elements to pop
            return None
        k = self._top.val
        if self._top == self._bottom:  # One single element
            self._bottom = self._top = None
        else:
            self._top.down.up = None
            self._top = self._top.down
        self._map.pop(k)
        return k

    def pop_bottom(self):
        """Pop the item at the bottom of the set

        Returns
        -------
        bottom : any hashable type
            The item at the bottom or *None* if the set is empty
        """
        if self._bottom == None:  # No elements to pop
            return None
        k = self._bottom.val
        if self._bottom == self._top:  # One single element
            self._top = self._bottom = None
        else:
            self._bottom.up.down = None
            self._bottom = self._bottom.up
        self._map.pop(k)
        return k

    def append_top(self, k):
        """Append an item at the top of the set

        Parameters
        ----------
        k : any hashable type
            The item to append
        """
        if k in self._map:
            raise KeyError('The item %s is already in the set' % str(k))
        n = self._Node(val=k, up=None, down=self._top)
        if self._top == self._bottom == None:
            self._bottom = n
        else:
            self._top.up = n
        self._top = n
        self._map[k] = n

    def append_bottom(self, k):
        """Append an item at the bottom of the set

        Parameters
        ----------
        k : any hashable type
            The item to append
        """
        if k in self._map:
            raise KeyError('The item %s is already in the set' % str(k))
        n = self._Node(val=k, up=self._bottom, down=None)
        if self._top == self._bottom == None:
            self._top = n
        else:
            self._bottom.down = n
        self._bottom = n
        self._map[k] = n

    def move_up(self, k):
        """Move a specified item one position up in the set

        Parameters
        ----------
        k : any hashable type
            The item to move up
        """
        if k not in self._map:
            raise KeyError('Item %s not in the set' % str(k))
        n = self._map[k]
        if n.up == None:  # already on top or there is only one element
            return
        if n.down == None:  # bottom but not top: there are at least two elements
            self._bottom = n.up
        else:
            n.down.up = n.up
        n.up.down = n.down
        new_up = n.up.up
        new_down = n.up
        if new_up:
            new_up.down = n
        else:
            self._top = n
        new_down.up = n
        n.up = new_up
        n.down = new_down

    def move_down(self, k):
        """Move a specified item one position down in the set

        Parameters
        ----------
        k : any hashable type
            The item to move down
        """
        if k not in self._map:
            raise KeyError('Item %s not in the set' % str(k))
        n = self._map[k]
        if n.down == None:  # already at the bottom or there is only one element
            return
        if n.up == None:
            self._top = n.down
        else:
            n.up.down = n.down
        n.down.up = n.up
        new_down = n.down.down
        new_up = n.down
        new_up.down = n
        if new_down != None:
            new_down.up = n
        else:
            self._bottom = n
        n.up = new_up
        n.down = new_down

    def move_to_top(self, k):
        """Move a specified item to the top of the set

        Parameters
        ----------
        k : any hashable type
            The item to move to the top
        """
        if k not in self._map:
            raise KeyError('Item %s not in the set' % str(k))
        n = self._map[k]
        if n.up == None:  # already on top or there is only one element
            return
        if n.down == None:  # at the bottom, there are at least two elements
            self._bottom = n.up
        else:
            n.down.up = n.up
        n.up.down = n.down
        # Move to top
        n.up = None
        n.down = self._top
        self._top.up = n
        self._top = n

    def move_to_bottom(self, k):
        """Move a specified item to the bottom of the set

        Parameters
        ----------
        k : any hashable type
            The item to move to the bottom
        """
        if k not in self._map:
            raise KeyError('Item %s not in the set' % str(k))
        n = self._map[k]
        if n.down == None:  # already at bottom or there is only one element
            return
        if n.up == None:  # at the top, there are at least two elements
            self._top = n.down
        else:
            n.up.down = n.down
        n.down.up = n.up
        # Move to top
        n.down = None
        n.up = self._bottom
        self._bottom.down = n
        self._bottom = n

    def insert_above(self, i, k):
        """Insert an item one position above a given item already in the set

        Parameters
        ----------
        i : any hashable type
            The item of the set above which the new item is inserted
        k : any hashable type
            The item to insert
        """
        if k in self._map:
            raise KeyError('Item %s already in the set' % str(k))
        if i not in self._map:
            raise KeyError('Item %s not in the set' % str(i))
        n = self._map[i]
        if n.up == None:  # Insert on top
            return self.append_top(k)
        # Now I know I am inserting between two actual elements
        m = self._Node(k, up=n.up, down=n)
        n.up.down = m
        n.up = m
        self._map[k] = m

    def insert_below(self, i, k):
        """Insert an item one position below a given item already in the set

        Parameters
        ----------
        i : any hashable type
            The item of the set below which the new item is inserted
        k : any hashable type
            The item to insert
        """
        if k in self._map:
            raise KeyError('Item %s already in the set' % str(k))
        if i not in self._map:
            raise KeyError('Item %s not in the set' % str(i))
        n = self._map[i]
        if n.down == None:  # Insert on top
            return self.append_bottom(k)
        # Now I know I am inserting between two actual elements
        m = self._Node(k, up=n, down=n.down)
        n.down.up = m
        n.down = m
        self._map[k] = m

    def index(self, k):
        """Return index of a given element.

        This operation has a O(n) time complexity, with n being the size of the
        set.

        Parameters
        ----------
        k : any hashable type
            The item whose index is queried

        Returns
        -------
        index : int
            The index of the item
        """
        if not k in self._map:
            raise KeyError('The item %s is not in the set' % str(k))
        index = 0
        curr = self._top
        while curr:
            if curr.val == k:
                return index
            curr = curr.down
            index += 1
        else:
            raise KeyError('It seems that the item %s is not in the set, '
                           'but you should never see this message. '
                           'There is something wrong with the code. '
                           'Debug it or report it to the developers' % str(k))

    def remove(self, k):
        """Remove an item from the set

        Parameters
        ----------
        k : any hashable type
            The item to remove
        """
        if k not in self._map:
            raise KeyError('Item %s not in the set' % str(k))
        n = self._map[k]
        if self._bottom == n:  # I am trying to remove the last node
            self._bottom = n.up
        else:
            n.down.up = n.up
        if self._top == n:  # I am trying to remove the top node
            self._top = n.down
        else:
            n.up.down = n.down
        self._map.pop(k)

    def clear(self):
        """Empty the set"""
        self._top = None
        self._bottom = None
        self._map.clear()

class LruCache(object):
    """Least Recently Used (LRU) cache eviction policy.

    According to this policy, When a new item needs to inserted into the cache,
    it evicts the least recently requested one.
    This eviction policy is efficient for line speed operations because both
    search and replacement tasks can be performed in constant time (*O(1)*).

    This policy has been shown to perform well in the presence of temporal
    locality in the request pattern. However, its performance drops under the
    Independent Reference Model (IRM) assumption (i.e. the probability that an
    item is requested is not dependent on previous requests).
    """

    def __init__(self, maxlen, **kwargs):
        self._cache = LinkedSet()
        self._maxlen = int(maxlen)
        if self._maxlen <= 0:
            raise ValueError('maxlen must be positive')

    def __len__(self):
        return len(self._cache)

    @property
    def maxlen(self):
        return self._maxlen

    def dump(self):
        return list(iter(self._cache))

    def position(self, k, *args, **kwargs):
        """Return the current position of an item in the cache. Position *0*
        refers to the head of cache (i.e. most recently used item), while
        position *maxlen - 1* refers to the tail of the cache (i.e. the least
        recently used item).

        This method does not change the internal state of the cache.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        position : int
            The current position of the item in the cache
        """
        if not k in self._cache:
            raise ValueError('The item %s is not in the cache' % str(k))
        return self._cache.index(k)

    def has(self, k, *args, **kwargs):
        return k in self._cache

    def get(self, k, *args, **kwargs):
        # search content over the list
        # if it has it push on top, otherwise return false
        if k not in self._cache:
            return False
        self._cache.move_to_top(k)
        return True

    def put(self, k, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache, it will pushed to the
        top of the cache.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        # if content in cache, push it on top, no eviction
        if k in self._cache:
            self._cache.move_to_top(k)
            return None
        # if content not in cache append it on top
        self._cache.append_top(k)
        return self._cache.pop_bottom() if len(self._cache) > self._maxlen else None

    def remove(self, k, *args, **kwargs):
        if k not in self._cache:
            return False
        self._cache.remove(k)
        return True

    def clear(self):
        self._cache.clear()