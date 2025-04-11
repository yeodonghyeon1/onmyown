##################################################
# author : YeoDongHyeon                          #
# date : 2025-04-11                              #
# revised : implementation heap python           #
#                                                #
##################################################


"""

HEAP 구현
완전 이진 트리를 힙의 원칙에 따라 데이터 관리 

insert() : Heap에 새로운 노드 투입
delete() : Heap의 루트 노드 삭제
get_max() : 최대값 확인
get_min() : 최솟값 확인
Heapify() : 일반적인 이진 트리를 Heap으로 변경

"""



        
class Maxheap:
    def __init__(self):
        self.heap_list = []
        self.max = 0
        self.last_header = 0
        pass

    def insert(self, x):
        self.heap_list.append(x)
        current_node_header = len(self.heap_list) - 1
        self.last_header = current_node_header

        while True:
            parent_node_header = (current_node_header - 1) // 2
            if parent_node_header == -1:
                break
            else:
                if self.heap_list[parent_node_header] >= self.heap_list[current_node_header]:
                    break
                else:
                    temp = self.heap_list[parent_node_header]
                    self.heap_list[parent_node_header] = self.heap_list[current_node_header]
                    self.heap_list[current_node_header] = temp
                    current_node_header = parent_node_header




    def delete(self):
        if self.is_empty():
            self.__print__("Empty")
        else:
            self.heap_list[0] = self.heap_list[self.last_header]
            del self.heap_list[self.last_header]

            current_header = 0
            while True:
                
                left_node_header = current_header * 2 + 1
                right_node_header = current_header * 2 + 2

                #자식 노드가 없을 때 탐색 정지
                if self.last_header <= left_node_header:
                    break
                
                try:
                    max_num = max(self.heap_list[left_node_header], self.heap_list[right_node_header])
                except:
                    max_num = self.heap_list[left_node_header
                                             ]
                if max_num == self.heap_list[left_node_header]:
                    if self.heap_list[current_header] < self.heap_list[left_node_header]:
                        temp = self.heap_list[left_node_header]
                        self.heap_list[left_node_header] = self.heap_list[current_header]
                        self.heap_list[current_header] = temp
                        current_header = left_node_header
                    else:
                        break
                else:
                    if self.heap_list[current_header] < self.heap_list[right_node_header]:
                        temp = self.heap_list[right_node_header]
                        self.heap_list[right_node_header] = self.heap_list[current_header]
                        self.heap_list[current_header] = temp
                        current_header = right_node_header
                    else:
                        break
            self.last_header = len(self.heap_list) - 1                    
    
    def get_max(self):
        self.__print__(self.heap_list[0])
        pass

    def get_min(self):
        start_level = 0
        point = len(self.heap_list) - 1
        while True:
            end_level = start_level * 2 + 1
            if start_level <= point and point < end_level:
                self.__print__(min(self.heap_list[start_level:point+1]))
                break
            start_level = start_level * 2 + 1


    def Heapify(self, list_data):
        max_heap = Maxheap()
        for i in list_data:
            max_heap.insert(i)

        return max_heap

    
    def is_empty(self):
        if not self.heap_list:
            return True
        else:
            return False
        
    def __print__(self,data,end=False):
        if end == True:
            print(data, end = ' ')
        else:
            print(data)


if __name__ == "__main__":
    a = Maxheap()
    # print( 3 // 2)
    a.insert(2)
    a.insert(5)
    a.insert(7)
    a.insert(3)
    a.insert(2)
    print(a.heap_list)
    a.delete()
    print(a.heap_list)
    a.get_max()
    a.get_min()
    a.insert(7)
    print(a.heap_list)
    b = a.Heapify([1,4,5,6,2,3,2])
    print(b.heap_list)


   