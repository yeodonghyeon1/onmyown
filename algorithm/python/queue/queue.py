##################################################
# author : YeoDongHyeon                          #
# date : 2025-04-10                              #
# revised : implementation queue python          #
#                                                #
##################################################

"""

QUEUE 구현
FIFO 원칙에 따라 데이터 관리 

enqueue(x) : 자료 x를 넣는 작업
dequeue() : 자료를 FIFO 방식으로 꺼내기
rear() : 가장 뒤 데이터(데이터를 넣는 쪽)
front() :가장 앞 데이터(데이터를 꺼내는 쪽)


"""


class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self,x):
        self.queue.append(x)

    def dequeue(self):
        if self.isEmpty():
            pass
        else:
            self.__print__(self.queue[0])
            del self.queue[0]

    def front(self):
        if self.isEmpty():
            pass
        else:
            self.__print__(self.queue[0])

    def rear(self):
        if self.isEmpty():
            pass
        else:
            self.__print__(self.queue[-1])

    def isEmpty(self):
        if not self.queue:
            self.__print__("Empty")
            return True
        
        return False

    def __print__(self,data,end=False):
        if end == True:
            print(data, end = ' ')
        else:
            print(data)



if __name__ == "__main__":
    q = Queue()

    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.enqueue(4)
    q.dequeue()
    q.dequeue()
    q.dequeue()
    q.dequeue()

    q.enqueue(5)
    q.enqueue(4)
    q.enqueue(3)
    q.rear()
    q.front()