import operator
class SegmentTree:
    def __init__(self,capacity,operation,init_value):
        #capacity should be positive and in power of 2
        assert (capacity > 0 and capacity & (capacity - 1) == 0)
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self,start,end,node,node_start,node_end):
        #to return results of operation in segments
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end)//2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(self._operate_helper(start, mid, 2 * node, node_start, mid),self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end))
    
    def operate(self, start=0, end=None):
        #results for self.operation
        if end is None:
            end = self.capacity
        if end < 0:
            end += self.capacity
        end -= 1
        return self._operate_helper(start, end, 1, 0, self.capacity - 1)
    
    def __setitem__(self,idx,val):
        #set value in tree
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2
    def __getitem__(self, idx):
        #get real value in leaf node of tree
        assert 0 <= idx < self.capacity
        return self.tree[self.capacity+idx]

#sum of segment tree
class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add, init_value=0.0)
    
    def sum(self,start=0,end=None):
        #for finding sum "arr[start]+...+arr[end]"
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound):
        #find highest index i about upper bound in the tree
        idx=1
        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity

#min of given segment tree
class MinSegmentTree(SegmentTree):
    def __init__(self,capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min, init_value=float("inf"))
    
    def min(self, start=0,end=None):
        #to find min of (arr[start],...,arr[end])
        return super(MinSegmentTree, self).operate(start, end)

