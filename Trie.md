# TRIE

```py
class Node:
    def __init__(self):
        self.node = {}
        self.flag = False

class Trie:

    def __init__(self):
        self.root = Node()

    def insert(self, word: str) -> None:
        temp = self.root
        for i in word:
            if not self.inNode(i,temp):
                temp.node[ord(i)-ord('a')] = Node()
            temp = temp.node[ord(i)-ord('a')]
        temp.flag = True

    def search(self, word: str) -> bool:
        temp = self.root
        for i in word:
            if not self.inNode(i,temp):
                return False
            temp = temp.node[ord(i)-ord('a')]
        return temp.flag

    def startsWith(self, prefix: str) -> bool:
        temp = self.root
        for i in prefix:
            if not self.inNode(i,temp):
                return False
            temp = temp.node[ord(i)-ord('a')]
        return True

    def delete(self,word: str) -> None:
        temp = self.root
        
    def inNode(self,i,temp):
        return ord(i)-ord('a') in temp.node
    

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```