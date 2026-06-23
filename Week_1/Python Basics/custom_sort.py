def vowels(strin) :
    x = 0
    for c in strin :
        if c == 'a':
            x+=1
            break
    for c in strin :
        if c == 'e':
            x+=1
            break
    for c in strin :
        if c == 'i':
            x+=1
            break
    for c in strin :
        if c == 'o':
            x+=1
            break
    for c in strin :
        if c == 'u':
            x+=1
            break
    return x;

def numsort(s_list, n):
    if(n==1):
        return
    maxsofar = s_list[0];
    maxindex = 0
    for i in range(n) :
        if len(s_list[i]) > len(maxsofar) :
            maxsofar = s_list[i]
            maxindex = i
    tempstring = s_list[n-1]
    s_list[n-1] = maxsofar
    s_list[maxindex] = tempstring
    numsort(s_list, n-1)

def swap(a,b) :
    temp = a
    a = b
    b = temp

def vowsort(s_list, n) :
    for i in range(n-1) :
        for j in range(i+1, n):
            if len(s_list[i]) == len(s_list[j]) :
                if vowels(s_list[j]) > vowels(s_list[i]) :
                    swap(s_list[i], s_list[j])


def sort_strings(s_list):
    n = len(s_list)
    numsort(s_list, n)
    vowsort(s_list, n)
    return

    


data = ["apple","ample", "banana", "kiwi", "sky", "aieou", "z"]
numsort(data,7)
print(data)
vowsort(data,7)
print(data)