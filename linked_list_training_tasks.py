"""
Linked List Training Tasks - From Easy to Hard
==============================================

This file contains 10 linked list problems with multiple solution approaches,
progressing from basic to advanced techniques using different Python patterns.
"""

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

# =============================================================================
# LINKED LIST NODE DEFINITIONS
# =============================================================================


class ListNode:
    """Standard singly linked list node"""

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class DoublyListNode:
    """Doubly linked list node"""

    def __init__(
        self,
        val: int = 0,
        prev: Optional["DoublyListNode"] = None,
        next: Optional["DoublyListNode"] = None,
    ):
        self.val = val
        self.prev = prev
        self.next = next

    def __repr__(self):
        return f"DoublyListNode({self.val})"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Create linked list from array"""
    if not values:
        return None

    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next

    return head


def linked_list_to_array(head: Optional[ListNode]) -> List[int]:
    """Convert linked list to array for testing"""
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


def print_linked_list(head: Optional[ListNode]) -> str:
    """Print linked list in readable format"""
    if not head:
        return "[]"

    values = linked_list_to_array(head)
    return " -> ".join(map(str, values))


# =============================================================================
# TASK 1: BASIC LINKED LIST OPERATIONS (EASY)
# =============================================================================


def task1_reverse_list_v1(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Reverse a linked list
    Input: 1 -> 2 -> 3 -> 4 -> 5
    Output: 5 -> 4 -> 3 -> 2 -> 1

    Approach: Iterative with three pointers
    """
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev


def task1_reverse_list_v2(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive approach for reversing linked list
    """
    if not head or not head.next:
        return head

    # Recursively reverse the rest
    reversed_head = task1_reverse_list_v2(head.next)

    # Reverse the current connection
    head.next.next = head
    head.next = None

    return reversed_head


def task1_find_middle_v1(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find middle node of linked list
    Input: 1 -> 2 -> 3 -> 4 -> 5
    Output: Node(3)

    Approach: Two-pass (count then find)
    """
    if not head:
        return None

    # Count nodes
    count = 0
    current = head
    while current:
        count += 1
        current = current.next

    # Find middle
    middle_index = count // 2
    current = head
    for _ in range(middle_index):
        current = current.next

    return current


def task1_find_middle_v2(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Two-pointer (tortoise and hare) approach
    """
    if not head:
        return None

    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


# =============================================================================
# TASK 2: LINKED LIST CYCLE DETECTION (EASY-MEDIUM)
# =============================================================================


def task2_has_cycle_v1(head: Optional[ListNode]) -> bool:
    """
    Detect if linked list has cycle
    Input: 1 -> 2 -> 3 -> 4 -> 2 (cycle back to node 2)
    Output: True

    Approach: Hash set to track visited nodes
    """
    visited = set()
    current = head

    while current:
        if current in visited:
            return True
        visited.add(current)
        current = current.next

    return False


def task2_has_cycle_v2(head: Optional[ListNode]) -> bool:
    """
    Floyd's cycle detection (tortoise and hare)
    """
    if not head or not head.next:
        return False

    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True

    return False


def task2_find_cycle_start_v1(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find the start of cycle in linked list
    Input: 1 -> 2 -> 3 -> 4 -> 2 (cycle starts at node 2)
    Output: Node(2)

    Approach: Hash set with position tracking
    """
    visited = set()
    current = head

    while current:
        if current in visited:
            return current
        visited.add(current)
        current = current.next

    return None


def task2_find_cycle_start_v2(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Floyd's algorithm - find meeting point, then find start
    """
    if not head or not head.next:
        return None

    # Phase 1: Detect cycle
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            break
    else:
        return None  # No cycle

    # Phase 2: Find start of cycle
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow


# =============================================================================
# TASK 3: MERGE TWO SORTED LISTS (MEDIUM)
# =============================================================================


def task3_merge_sorted_v1(
    list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    """
    Merge two sorted linked lists
    Input: list1 = 1 -> 2 -> 4, list2 = 1 -> 3 -> 4
    Output: 1 -> 1 -> 2 -> 3 -> 4 -> 4

    Approach: Iterative with dummy node
    """
    dummy = ListNode(0)
    current = dummy

    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    # Attach remaining nodes
    current.next = list1 if list1 else list2

    return dummy.next


def task3_merge_sorted_v2(
    list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    """
    Recursive approach for merging sorted lists
    """
    if not list1:
        return list2
    if not list2:
        return list1

    if list1.val <= list2.val:
        list1.next = task3_merge_sorted_v2(list1.next, list2)
        return list1
    else:
        list2.next = task3_merge_sorted_v2(list1, list2.next)
        return list2


def task3_merge_k_sorted_v1(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted linked lists
    Input: [[1,4,5],[1,3,4],[2,6]]
    Output: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6

    Approach: Divide and conquer
    """
    if not lists:
        return None

    def merge_two_lists(l1, l2):
        return task3_merge_sorted_v1(l1, l2)

    while len(lists) > 1:
        merged_lists = []

        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged_lists.append(merge_two_lists(l1, l2))

        lists = merged_lists

    return lists[0]


def task3_merge_k_sorted_v2(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Using min heap for efficient merging
    """
    import heapq

    if not lists:
        return None

    # Create min heap with (value, index, node)
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next


# =============================================================================
# TASK 4: REMOVE DUPLICATES (MEDIUM)
# =============================================================================


def task4_remove_duplicates_v1(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Remove duplicates from sorted linked list
    Input: 1 -> 1 -> 2 -> 3 -> 3
    Output: 1 -> 2 -> 3

    Approach: Single pass with current pointer
    """
    if not head:
        return head

    current = head

    while current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next

    return head


def task4_remove_duplicates_v2(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Remove all duplicates (keep only unique elements)
    Input: 1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5
    Output: 1 -> 2 -> 5

    Approach: Dummy node with skip logic
    """
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    while head:
        if head.next and head.val == head.next.val:
            # Skip all duplicates
            while head.next and head.val == head.next.val:
                head = head.next
            prev.next = head.next
        else:
            prev = head
        head = head.next

    return dummy.next


def task4_remove_duplicates_unsorted_v1(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Remove duplicates from unsorted linked list
    Input: 1 -> 3 -> 2 -> 3 -> 1
    Output: 1 -> 3 -> 2

    Approach: Hash set for tracking seen values
    """
    if not head:
        return head

    seen = set()
    seen.add(head.val)
    current = head

    while current.next:
        if current.next.val in seen:
            current.next = current.next.next
        else:
            seen.add(current.next.val)
            current = current.next

    return head


def task4_remove_duplicates_unsorted_v2(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Remove duplicates without extra space (O(n²) time)
    """
    if not head:
        return head

    current = head

    while current:
        runner = current
        while runner.next:
            if runner.next.val == current.val:
                runner.next = runner.next.next
            else:
                runner = runner.next
        current = current.next

    return head


# =============================================================================
# TASK 5: LINKED LIST INTERSECTION (MEDIUM-HARD)
# =============================================================================


def task5_intersection_v1(
    headA: Optional[ListNode], headB: Optional[ListNode]
) -> Optional[ListNode]:
    """
    Find intersection point of two linked lists
    Input: A = 4 -> 1 -> 8 -> 4 -> 5, B = 5 -> 6 -> 1 -> 8 -> 4 -> 5
    Output: Node(8) - first common node

    Approach: Hash set to track nodes from first list
    """
    if not headA or not headB:
        return None

    visited = set()
    current = headA

    # Store all nodes from list A
    while current:
        visited.add(current)
        current = current.next

    # Check nodes from list B
    current = headB
    while current:
        if current in visited:
            return current
        current = current.next

    return None


def task5_intersection_v2(
    headA: Optional[ListNode], headB: Optional[ListNode]
) -> Optional[ListNode]:
    """
    Two-pointer approach with length calculation
    """
    if not headA or not headB:
        return None

    def get_length(head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length

    lenA = get_length(headA)
    lenB = get_length(headB)

    # Align starting positions
    while lenA > lenB:
        headA = headA.next
        lenA -= 1

    while lenB > lenA:
        headB = headB.next
        lenB -= 1

    # Find intersection
    while headA and headB:
        if headA == headB:
            return headA
        headA = headA.next
        headB = headB.next

    return None


def task5_intersection_v3(
    headA: Optional[ListNode], headB: Optional[ListNode]
) -> Optional[ListNode]:
    """
    Elegant two-pointer approach
    """
    if not headA or not headB:
        return None

    pointerA = headA
    pointerB = headB

    while pointerA != pointerB:
        pointerA = pointerA.next if pointerA else headB
        pointerB = pointerB.next if pointerB else headA

    return pointerA


# =============================================================================
# TASK 6: REORDER LIST (HARD)
# =============================================================================


def task6_reorder_list_v1(head: Optional[ListNode]) -> None:
    """
    Reorder list: L0 → L1 → … → Ln-1 → Ln to L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …
    Input: 1 -> 2 -> 3 -> 4 -> 5
    Output: 1 -> 5 -> 2 -> 4 -> 3

    Approach: Find middle, reverse second half, merge
    """
    if not head or not head.next:
        return

    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    second_half = slow.next
    slow.next = None

    def reverse_list(node):
        prev = None
        while node:
            next_temp = node.next
            node.next = prev
            prev = node
            node = next_temp
        return prev

    second_half = reverse_list(second_half)

    # Merge two halves
    first_half = head
    while second_half:
        temp1 = first_half.next
        temp2 = second_half.next

        first_half.next = second_half
        second_half.next = temp1

        first_half = temp1
        second_half = temp2


def task6_reorder_list_v2(head: Optional[ListNode]) -> None:
    """
    Using deque for simpler implementation
    """
    if not head or not head.next:
        return

    # Store all nodes in deque
    nodes = deque()
    current = head
    while current:
        nodes.append(current)
        current = current.next

    # Reconstruct with alternating pattern
    dummy = ListNode(0)
    current = dummy
    from_left = True

    while nodes:
        if from_left:
            current.next = nodes.popleft()
        else:
            current.next = nodes.pop()
        current = current.next
        from_left = not from_left

    current.next = None


# =============================================================================
# TASK 7: ROTATE LIST (HARD)
# =============================================================================


def task7_rotate_right_v1(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Rotate list to the right by k places
    Input: 1 -> 2 -> 3 -> 4 -> 5, k = 2
    Output: 4 -> 5 -> 1 -> 2 -> 3

    Approach: Find length, create circle, break at right point
    """
    if not head or not head.next or k == 0:
        return head

    # Find length and make circular
    length = 1
    current = head
    while current.next:
        current = current.next
        length += 1

    current.next = head  # Make circular

    # Find new tail (length - k % length - 1 steps from head)
    k = k % length
    steps_to_new_tail = length - k

    new_tail = head
    for _ in range(steps_to_new_tail - 1):
        new_tail = new_tail.next

    new_head = new_tail.next
    new_tail.next = None

    return new_head


def task7_rotate_right_v2(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Two-pass approach without circular list
    """
    if not head or not head.next or k == 0:
        return head

    # First pass: find length
    length = 0
    current = head
    while current:
        length += 1
        current = current.next

    k = k % length
    if k == 0:
        return head

    # Second pass: find new tail
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next

    new_head = new_tail.next
    new_tail.next = None

    # Find end of new list
    current = new_head
    while current.next:
        current = current.next

    current.next = head

    return new_head


# =============================================================================
# TASK 8: PALINDROME LINKED LIST (HARD)
# =============================================================================


def task8_is_palindrome_v1(head: Optional[ListNode]) -> bool:
    """
    Check if linked list is palindrome
    Input: 1 -> 2 -> 2 -> 1
    Output: True

    Approach: Convert to array and check
    """
    values = []
    current = head

    while current:
        values.append(current.val)
        current = current.next

    return values == values[::-1]


def task8_is_palindrome_v2(head: Optional[ListNode]) -> bool:
    """
    O(1) space approach: reverse second half
    """
    if not head or not head.next:
        return True

    # Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    def reverse_list(node):
        prev = None
        while node:
            next_temp = node.next
            node.next = prev
            prev = node
            node = next_temp
        return prev

    second_half = reverse_list(slow)

    # Compare halves
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True


def task8_is_palindrome_v3(head: Optional[ListNode]) -> bool:
    """
    Recursive approach with helper function
    """

    def check_palindrome(node):
        nonlocal front_pointer

        if not node:
            return True

        if not check_palindrome(node.next):
            return False

        if node.val != front_pointer.val:
            return False

        front_pointer = front_pointer.next
        return True

    front_pointer = head
    return check_palindrome(head)


# =============================================================================
# TASK 9: COPY LIST WITH RANDOM POINTER (VERY HARD)
# =============================================================================


class RandomListNode:
    def __init__(
        self,
        val: int = 0,
        next: Optional["RandomListNode"] = None,
        random: Optional["RandomListNode"] = None,
    ):
        self.val = val
        self.next = next
        self.random = random


def task9_copy_random_list_v1(
    head: Optional[RandomListNode],
) -> Optional[RandomListNode]:
    """
    Deep copy linked list with random pointers
    Input: [[7,null],[13,0],[11,4],[10,2],[1,0]]
    Output: Deep copy of the list

    Approach: HashMap to track original -> copy mapping
    """
    if not head:
        return None

    # First pass: create all nodes
    node_map = {}
    current = head

    while current:
        node_map[current] = RandomListNode(current.val)
        current = current.next

    # Second pass: set next and random pointers
    current = head
    while current:
        if current.next:
            node_map[current].next = node_map[current.next]
        if current.random:
            node_map[current].random = node_map[current.random]
        current = current.next

    return node_map[head]


def task9_copy_random_list_v2(
    head: Optional[RandomListNode],
) -> Optional[RandomListNode]:
    """
    O(1) space approach: interweave original and copy nodes
    """
    if not head:
        return None

    # Step 1: Create copy nodes interweaved with original
    current = head
    while current:
        copy_node = RandomListNode(current.val)
        copy_node.next = current.next
        current.next = copy_node
        current = copy_node.next

    # Step 2: Set random pointers for copy nodes
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next

    # Step 3: Separate original and copy lists
    dummy = RandomListNode(0)
    copy_current = dummy
    current = head

    while current:
        copy_current.next = current.next
        current.next = current.next.next
        current = current.next
        copy_current = copy_current.next

    return dummy.next


# =============================================================================
# TASK 10: LRU CACHE WITH DOUBLY LINKED LIST (VERY HARD)
# =============================================================================


class LRUCache:
    """
    Implement LRU Cache using doubly linked list + hash map
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node

        # Create dummy head and tail
        self.head = DoublyListNode(0)
        self.tail = DoublyListNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node: DoublyListNode):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DoublyListNode):
        """Remove node from list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _move_to_head(self, node: DoublyListNode):
        """Move node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> DoublyListNode:
        """Remove and return last node"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node

    def get(self, key: int) -> int:
        """Get value and mark as recently used"""
        if key in self.cache:
            node = self.cache[key]
            self._move_to_head(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.val = value
            self._move_to_head(node)
        else:
            # Add new
            new_node = DoublyListNode(value)
            new_node.key = key  # Store key in node for removal

            if len(self.cache) >= self.capacity:
                # Remove least recently used
                tail = self._pop_tail()
                del self.cache[tail.key]

            self.cache[key] = new_node
            self._add_node(new_node)


class LRUCacheV2:
    """
    Alternative implementation using collections.OrderedDict
    """

    def __init__(self, capacity: int):
        from collections import OrderedDict

        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove oldest
            self.cache.popitem(last=False)

        self.cache[key] = value


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("LINKED LIST TRAINING TASKS - TEST RESULTS")
    print("=" * 60)

    # Task 1: Basic Operations
    print("\n1. Basic Linked List Operations:")
    head = create_linked_list([1, 2, 3, 4, 5])
    print(f"Original: {print_linked_list(head)}")

    reversed_head = task1_reverse_list_v1(create_linked_list([1, 2, 3, 4, 5]))
    print(f"Reversed v1: {print_linked_list(reversed_head)}")

    reversed_head2 = task1_reverse_list_v2(create_linked_list([1, 2, 3, 4, 5]))
    print(f"Reversed v2: {print_linked_list(reversed_head2)}")

    middle = task1_find_middle_v1(create_linked_list([1, 2, 3, 4, 5]))
    print(f"Middle v1: {middle.val if middle else None}")

    middle2 = task1_find_middle_v2(create_linked_list([1, 2, 3, 4, 5]))
    print(f"Middle v2: {middle2.val if middle2 else None}")

    # Task 2: Cycle Detection
    print("\n2. Cycle Detection:")
    # Create cycle: 1 -> 2 -> 3 -> 4 -> 2
    head = create_linked_list([1, 2, 3, 4])
    head.next.next.next.next = head.next  # Create cycle

    has_cycle1 = task2_has_cycle_v1(head)
    has_cycle2 = task2_has_cycle_v2(head)
    print(f"Has cycle v1: {has_cycle1}")
    print(f"Has cycle v2: {has_cycle2}")

    # Task 3: Merge Sorted Lists
    print("\n3. Merge Sorted Lists:")
    list1 = create_linked_list([1, 2, 4])
    list2 = create_linked_list([1, 3, 4])
    merged = task3_merge_sorted_v1(list1, list2)
    print(f"Merged: {print_linked_list(merged)}")

    # Task 4: Remove Duplicates
    print("\n4. Remove Duplicates:")
    head = create_linked_list([1, 1, 2, 3, 3])
    deduplicated = task4_remove_duplicates_v1(head)
    print(f"Remove duplicates: {print_linked_list(deduplicated)}")

    head2 = create_linked_list([1, 2, 3, 3, 4, 4, 5])
    deduplicated2 = task4_remove_duplicates_v2(head2)
    print(f"Remove all duplicates: {print_linked_list(deduplicated2)}")

    # Task 6: Reorder List
    print("\n6. Reorder List:")
    head = create_linked_list([1, 2, 3, 4, 5])
    original_values = linked_list_to_array(head)
    print(f"Original: {original_values}")
    task6_reorder_list_v1(head)
    reordered_values = linked_list_to_array(head)
    print(f"Reordered: {reordered_values}")

    # Task 7: Rotate List
    print("\n7. Rotate List:")
    head = create_linked_list([1, 2, 3, 4, 5])
    rotated = task7_rotate_right_v1(head, 2)
    print(f"Rotated right by 2: {print_linked_list(rotated)}")

    # Task 8: Palindrome Check
    print("\n8. Palindrome Check:")
    head = create_linked_list([1, 2, 2, 1])
    is_palindrome1 = task8_is_palindrome_v1(head)
    print(f"Is palindrome v1: {is_palindrome1}")

    head2 = create_linked_list([1, 2, 2, 1])
    is_palindrome2 = task8_is_palindrome_v2(head2)
    print(f"Is palindrome v2: {is_palindrome2}")

    # Task 10: LRU Cache
    print("\n10. LRU Cache:")
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(f"Get 1: {cache.get(1)}")  # Should return 1
    cache.put(3, 3)  # Evicts key 2
    print(f"Get 2: {cache.get(2)}")  # Should return -1 (not found)
    cache.put(4, 4)  # Evicts key 1
    print(f"Get 1: {cache.get(1)}")  # Should return -1 (not found)
    print(f"Get 3: {cache.get(3)}")  # Should return 3
    print(f"Get 4: {cache.get(4)}")  # Should return 4

    print("\n" + "=" * 60)
    print("ALL LINKED LIST TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
