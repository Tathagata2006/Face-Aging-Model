wt = input("Weight: ")
wt = float(wt)
command = input("(K)g or L(bs): ")
if command == 'l' or command == 'L' :
    print("Weight in Kg: " + str(wt*0.45))
elif command == 'K' or command == 'k':
    print ("Weight in lbs is: " + str(wt/0.45))