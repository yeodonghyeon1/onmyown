##################################################
# author : YeoDongHyeon                          #
# date : 2025-04-05                              #
# revised : implementation stack python          #
#                                                #
##################################################


"""

STACK 구현
LIFO 원칙에 따라 데이터 관리 

push(x): 자료 x를 넣는 작업
pop(): 자료를 꺼내는 작업
peek(): 마지막에 넣은 자료를 확인하는 작업으로 pop과 비슷하지만, 값을 제거하지 않음
is_empty(): 빈 스택인지 확인하는 작업
check_stack(): 스택 항목 조회

"""

class Node:
    def __init__(self, data):
        self.data = data
        self.under_node = None 
class Stack:
    def __init__(self):
        self.top_data = None
        
    def push(self, data):
        if self.top_data == None:
            self.top_data = Node(data)
        else:
            node =  Node(data)
            node.under_node = self.top_data
            self.top_data = node

    def pop(self):
        if self.is_empty():
            self.__print__("empty")
        else:
            node = self.top_data
            self.__print__(self.top_data.data)
            self.top_data = node.under_node

    def is_empty(self):
        if self.top_data == None:
            return True
        else:
            return False

    def peek(self):
        if self.top_data ==  None:
            self.__print__("empty")
        else:
            self.__print__(self.top_data.data)
        
    def check_stack(self):
        if self.top_data == None:
            self.__print__("empty")
        else:
            node = self.top_data
            while(True):
                self.__print__(node.data, end=True)
                node = node.under_node
                if node == None:
                    break
    def __print__(self,data,end=False):
        if end == True:
            print(data, end = ' ')
        else:
            print(data)
        

if __name__ == "__main__":
    print("exam stack!")
    stack = Stack()
    stack.push(3)
    stack.peek()
    stack.pop()
    stack.pop()
    stack.push(4)
    stack.push(2)
    stack.push(1)
    stack.check_stack()
    print("\n")
    stack.pop()
    stack.pop()
    stack.pop()





