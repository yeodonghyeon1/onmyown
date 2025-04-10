import stack
import os
def main():
    test_stack = stack.Stack()
    test_stack.push(3)
    test_stack.pop()


def solution(progresses, speeds):
    Q=[]
    for p, s in zip(progresses, speeds):
        print(Q)
        if len(Q)==0 or Q[-1][0]<-((p-100)//s):
            Q.append([-((p-100)//s),1])
        else:
            Q[-1][1]+=1
    return [q[1] for q in Q]

    
if __name__ == "__main__":
    main()
    print(solution(progresses=[2,1,4], speeds=[2,4,1]))
49,1