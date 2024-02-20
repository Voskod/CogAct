import doctest
import random
import re
import time
import xml.etree.ElementTree as ET
import json
import os
import sys

def p_equal(p1, p2):
    if len(p1) > 0 and p1[-1] == '$':
        p1 = p1[:-1]
    if len(p2) > 0 and p2[-1] == '$':
        p2 = p2[:-1]
    return p1 == p2

def p_match(p1, p2):
    if p1[-1:] == ['$']:
        if len(p1) != len(p2) or p2[-1:] != ['$']:
            return False
    else:
        if len(p1) > len(p2):
            return False
    return p1 == p2[:len(p1)]

def p_diff(p1, p2):
    for i in range(len(p1)):
        if i >= len(p2) or p1[i] != p2[i]:
            return p1[i:]
    return []

SIMILARITY_THRESHOLD = 5
ADD_LINK_TIME = 10000
DISCRIMINATION_TIME = 10000
FAMILIARISATION_TIME = 2000

class Stm:
    class Item:   #create an item container with head and pointer to next container                
        def __init__(self, data):    
            self.next = None
            self.data = data

    def __init__(self, maxlen): #STM constructor
        self.maxlen = maxlen
        self.size = 0
        self.head = None
        self.tail = None

    def push(self, data):
        self.remove(data)
        e = Stm.Item(data)
        if self.size == 0:
            self.head = self.tail = e
        else:
            check = self.head.data.image.copy()
            k = 0
            # print('check similarity', check, data.image)
            matches = []
            for i in data.image:
                if i in check:
                    k += 1
                    matches.append((data.idx, i, check))
                    if k == SIMILARITY_THRESHOLD:
                        self.head.data.similarity_links.add(data)
                        data.similarity_links.add(self.head.data)
#                         print('------------------------')
#                         for idx, i, check in matches:
#                             print(data.idx, 'FOUND', i, 'in', check)
#                         print('------------------------')
                        break
                    check.remove(i)
            self.tail.next = e
            self.tail = e
        self.size += 1
        if self.size > self.maxlen:
            self.pop()

    def pop(self):
        if self.size == 0:
            return None
        else:
            e = self.head.data
            self.head = self.head.next
            self.size -= 1
            return e

    def remove(self, value):
        if self.size == 0:
            return
        if self.head.data == value:
            self.head = self.head.next
            self.size -= 1
        else:
            p = self.head
            e = self.head.next
            while e:
                if e.data == value:
                    p.next = e.next
                    self.size -= 1
                    if e == self.tail:
                        self.tail = p
                    break
                p = p.next
                e = e.next

    class Iterator:
        def __init__(self, e):
            self.e = e

        def __next__(self):
            if self.e:
                e = self.e
                self.e = e.next
                return e.data
            raise StopIteration()

    def __iter__(self):
        return Stm.Iterator(self.head)
        
class Link:
    def __init__(self, test, child):
        self.test = test
        self.child = child
    
    def passes(self, pattern):
        return p_match(self.test, pattern)

class Node:
    def __init__(self, nodes, contents, image, children):
        self.contents = contents
        self.image = image
        if len(self.image) > 1 and self.image[-1] == '$':
            self.image.pop()
        self.children = children
        self.idx = len(nodes)
        nodes.append(self)
        self.label = None
        self.similarity_links = set()

class Modality:
    def __init__(self, cogact, name, stm_size=2):
        self.cogact = cogact
        self.name = name
        self.stm = Stm(stm_size)
        self.net = Node(nodes=cogact.nodes, contents=['ROOT NODE'], image=[], children=[])

    def recognise(self, pattern):
        current_node = self.net
        children = current_node.children
        sorted_pattern = pattern
        next_link = 0
        
        while next_link < len(children):
            link = children[next_link]
            if link.passes(sorted_pattern):
                current_node = link.child
                children = current_node.children
                next_link = 0
                sorted_pattern = p_diff(sorted_pattern, link.test)
            else:
                next_link += 1
        self.stm.push(current_node)
        return current_node
    
    def recognise_and_learn(self, pattern):
        current_node = self.recognise(pattern)
        if current_node.image != pattern:
            if current_node == self.net or \
               not p_match(current_node.image, pattern) or \
               current_node.image[-1:] == ['$']:
                current_node = self.discriminate(current_node, pattern)
            else:
                current_node = self.familiarise(current_node, pattern)
        self.stm.push(current_node)
        return current_node
    
    def get_first(self, pattern):
        fst = pattern[:1]
        if len(fst) == 0 or fst == ['$']:
            return []
        else:
            return fst
    
    def familiarise(self, node, pattern):
        new_info = self.get_first(p_diff(pattern, node.image))
        if len(new_info) == 0:
            return node
        
        retrieved_chunk = self.recognise(new_info)
        if retrieved_chunk == self.net:
            return self.learn_primitive(new_info)
        
        if node.image[:-1] == ['$']:
            node.image.pop()
        node.image += new_info
        self.cogact.clock += FAMILIARISATION_TIME
        return node
    
    def discriminate(self, node, pattern):
        new_info = p_diff(pattern, node.contents)
        
        if len(new_info) == 0 or new_info == ['$']:
            new_info = ['$']
            if self.recognise(new_info).contents == new_info:
                return self.add_test(node, new_info)
            else:
                child = Node(nodes=self.cogact.nodes, contents=new_info, image=new_info, children=[])
                self.net.children.append(Link(new_info, child))
                return child
        
        chunk = self.recognise(new_info)
        if chunk == self.net:
            return self.learn_primitive(self.get_first(new_info) or ['$'])
        elif p_match(chunk.contents, new_info):
            return self.add_test(node, chunk.contents.copy())
        else:
            return self.add_test(node, self.get_first(new_info))
            

    def add_test(self, node, pattern):
        for child in node.children:
            if child.test == pattern:
                return node
        if node == self.net:
            pat = pattern.copy()
        else:
            pat = node.contents + pattern
        child = Node(nodes=self.cogact.nodes, contents=pat, image=pat.copy(), children=[])
        node.children.append(Link(pattern, child))
        self.cogact.clock += DISCRIMINATION_TIME
        return child
    
    def learn_primitive(self, pattern):
        assert len(pattern) == 1
        child = Node(nodes=self.cogact.nodes, contents=pattern, image=[], children=[])
        self.net.children.append(Link(pattern, child))
        self.cogact.clock += DISCRIMINATION_TIME
        return child
    
    def print_tree(self, node=None, parent=[], level=0):
        if not node:
            node = self.net
        indent = '-------' * level
        contents = '< ' + ' '.join(p_diff(node.contents, parent)) + ' >'
        text = 'Node: ' + str(node.idx)
        image = '< '+ ' '.join(node.image) + ' >'
        
        string = [indent + contents, text, image]
        if node.label is not None:
            string.append('(' + str(node.label) + ')')
        if node.similarity_links:
            string.append({i.idx for i in node.similarity_links})
        print(*string)

        for child in node.children:
            self.print_tree(child.child, node.contents, level + 1)

    def print_stm(self):
        print('STM: ', end='')
        for node in self.stm:
            print(node.idx, '<', *node.image, '>', end=', ')
        print()

class CogAct:
    def __init__(self):
        self.nodes = []
        self.visual = Modality(self, 'visual')
        self.verbal = Modality(self, 'verbal')
        self.action = Modality(self, 'action')
        self.modalities = {
            'visual': self.visual,
            'verbal': self.verbal,
            'action': self.action,
        }
        self.clock = 0
