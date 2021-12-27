# Function which returns subset or r length from n
from itertools import combinations

def rSubset(arr, r):

	# return list of all subsets of length r
	# to deal with duplicate subsets use
	# set(list(combinations(arr, r)))
	return list(combinations(arr, r))

# Driver Function
if __name__ == "__main__":
	arr = [i for i in range(10)]
	r = 2
	print (rSubset(arr, r))
