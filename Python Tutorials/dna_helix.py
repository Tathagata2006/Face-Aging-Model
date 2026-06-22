def print_helix(n, turns):
    letters = ['A', 'C', 'G', 'T']
    a = 0
    if n%2 == 1 :
        while a < turns :
            for i in range(2*n - 2) :
                if i<n :
                    print(i*' ' + letters[i%4] + 2*(n-i-1)*' ' + letters[i%4])
                else :
                    print((2*(n-1) - i)*' ' + letters[i%4] + 2*(i-n+1)*' ' + letters[i%4])
            a = a+1
        print('A' + 2*(n-1)*' ' + 'A')
    else :
        if turns%2 == 0 :
            while a < turns/2 :
                for i in range(4*(n-1)) :
                    if i<n :
                        print(i*' ' + letters[i%4] + 2*(n-i-1)*' ' + letters[i%4])
                    elif i<2*(n-1) :
                        print((2*(n-1) - i)*' ' + letters[i%4] + 2*(i-n+1)*' ' + letters[i%4])
                    elif i<3*(n-1) :
                        print((i-2*(n-1))*' ' + letters[i%4] + 2*(3*n-i-3)*' ' + letters[i%4])
                    else :
                        print((2*(n-1) - i + 2*(n-1))*' ' + letters[i%4] + 2*(i-3*n+3)*' ' + letters[i%4])
                a = a+1
            print('A' + 2*(n-1)*' ' + 'A') 
        else :
            while a < (turns-1)/2 :
                for i in range(4*(n-1)) :
                    if i<n :
                        print(i*' ' + letters[i%4] + 2*(n-i-1)*' ' + letters[i%4])
                    elif i<2*(n-1) :
                        print((2*(n-1) - i)*' ' + letters[i%4] + 2*(i-n+1)*' ' + letters[i%4])
                    elif i<3*(n-1) :
                        print((i-2*(n-1))*' ' + letters[i%4] + 2*(3*n-i-3)*' ' + letters[i%4])
                    else :
                        print((2*(n-1) - i + 2*(n-1))*' ' + letters[i%4] + 2*(i-3*n+3)*' ' + letters[i%4])
                a = a+1
            for i in range(2*(n-1)) :
                    if i<n :
                        print(i*' ' + letters[i%4] + 2*(n-i-1)*' ' + letters[i%4])
                    elif i<2*(n-1) :
                        print((2*(n-1) - i)*' ' + letters[i%4] + 2*(i-n+1)*' ' + letters[i%4])
            print('G' + 2*(n-1)*' ' + 'G')
    
print_helix(4,1)