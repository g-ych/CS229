import matplotlib.pyplot as plt
import util
import numpy as np 

def plot(x, y, save_path, abline:bool=False):
	plt.figure()
	plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
	plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

	plt.xlabel('x1')
	plt.ylabel('x2')

	if abline:
		abline_x = [0, 1]
		abline_y = [1, 0]
		plt.plot(abline_x, abline_y)

	plt.savefig(save_path)

def main():
	Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
	plot(Xa, Ya, 'data_a.png', abline=True)
	Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
	plot(Xb, Yb, 'data_b.png', abline=True)

if __name__ == '__main__':
	main()